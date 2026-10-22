package pi

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// PromptTurn starts a turn with a fresh prompt.
func (s *session) PromptTurn(ctx context.Context, options harness.PromptTurnOptions) (harness.PromptControl, error) {
	if err := validateResponseFormat(options.ResponseFormat); err != nil {
		return nil, err
	}

	prompt, err := extractPrompt(options.Prompt)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	s.liveTurn = false
	s.rerunContinue = false
	s.mu.Unlock()

	return s.runTurn(ctx, runOptions{
		prompt:   prompt,
		settings: options.TurnSettings,
		emit:     options.Emit,
	})
}

// ContinueTurn continues the in-flight turn. A turn Pi kept running in the
// sandbox streams into the new control from the cursor, without a new
// prompt. Otherwise, as upstream, an empty prompt asks Pi to continue its
// own thread from the session file, recomputing the work in flight at the
// suspension.
func (s *session) ContinueTurn(ctx context.Context, options harness.ContinueTurnOptions) (harness.PromptControl, error) {
	if err := validateResponseFormat(options.ResponseFormat); err != nil {
		return nil, err
	}

	if err := s.ensureUsable(); err != nil {
		return nil, err
	}

	s.mu.Lock()
	attach := s.liveTurn
	s.liveTurn = false
	s.rerunContinue = false
	s.mu.Unlock()

	if attach {
		return s.attachTurn(ctx, options.Emit)
	}

	return s.runTurn(ctx, runOptions{
		settings: options.TurnSettings,
		emit:     options.Emit,
	})
}

// attachTurn wires a turn into the run Pi kept going, then starts reading
// Pi's output at the cursor.
func (s *session) attachTurn(ctx context.Context, emit func(part harness.StreamPart)) (harness.PromptControl, error) {
	current := newTurn(ctx, s, emit)
	current.markPromptSent()

	s.mu.Lock()
	s.turn = current
	s.mu.Unlock()

	if err := s.conn.start(ctx); err != nil {
		current.fail(err)

		return current, nil
	}

	return current, nil
}

// runOptions configure runTurn.
type runOptions struct {
	prompt   promptContent
	settings harness.TurnSettings
	emit     func(part harness.StreamPart)
}

// runTurn prepares Pi for the turn's settings and sends the prompt.
// Failures while preparing are returned; failures of the prompt are
// reported by the turn, as upstream.
func (s *session) runTurn(ctx context.Context, options runOptions) (harness.PromptControl, error) {
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}

	if err := s.conn.start(ctx); err != nil {
		return nil, err
	}

	if err := s.waitForAbort(ctx); err != nil {
		return nil, err
	}

	if err := s.prepareTurn(ctx, options.settings); err != nil {
		return nil, err
	}

	selected, err := s.selectModel(ctx, options.settings.Model)
	if err != nil {
		return nil, err
	}

	if len(options.prompt.images) > 0 && selected != nil && !selected.supportsImages() {
		return nil, unsupported(fmt.Sprintf("Harness 'pi' model '%s' does not support image input.", selected.ID))
	}

	baseline, err := s.sessionTokens(ctx)
	if err != nil {
		return nil, err
	}

	current := s.beginTurn(ctx, beginOptions{
		emit:     options.emit,
		baseline: baseline,
		tools:    options.settings.Tools,
	})

	start := harness.StreamStartPart{}
	if selected != nil {
		start.ModelID = selected.ID
	}

	current.forward(start)
	s.flushCompactions(current)

	current.markPromptSent()

	reply, err := s.conn.call(ctx, promptCommand{
		Type:    "prompt",
		Message: options.prompt.text,
		Images:  options.prompt.images,
	})
	if err != nil {
		if ctx.Err() == nil {
			current.fail(err)
		}

		return current, nil
	}

	var disposition promptResponseData
	_ = json.Unmarshal(reply.Data, &disposition)

	if disposition.Disposition == dispositionHandled && current.beginFinish() {
		s.finishInBackground(current)
	}

	return current, nil
}

// ensureIdle fails when the session ended or a turn is running.
func (s *session) ensureIdle() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		return errSessionStopped
	}

	if s.turn != nil {
		return fmt.Errorf("pi session %s already has an active turn", s.sessionID)
	}

	return nil
}

// beginOptions configure beginTurn.
type beginOptions struct {
	emit     func(part harness.StreamPart)
	baseline sessionTokens
	tools    []harness.ToolSpec
}

// beginTurn makes a new turn the active turn with a fresh translator.
func (s *session) beginTurn(ctx context.Context, options beginOptions) *turn {
	current := newTurn(ctx, s, options.emit)

	hostToolNames := make([]string, 0, len(options.tools))
	for _, tool := range options.tools {
		hostToolNames = append(hostToolNames, tool.Name)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.turn = current
	s.translator = newTranslator(s.builtinToolNames, hostToolNames)
	s.turnBaseline = &options.baseline

	return current
}

// flushCompactions emits the compactions Pi reported between turns.
func (s *session) flushCompactions(current *turn) {
	s.mu.Lock()
	pending := s.pendingCompactions
	s.pendingCompactions = nil
	s.mu.Unlock()

	for _, part := range pending {
		current.forward(part)
	}
}

// prepareTurn writes the turn's skills, instructions and tools, and lets the
// adapter's extension apply them: changed skills reload Pi's resources,
// changed tools are registered again. Pi reads the instructions when the
// run starts.
func (s *session) prepareTurn(ctx context.Context, settings harness.TurnSettings) error {
	skills, err := harness.WriteSkills(ctx, harness.WriteSkillsOptions{
		Sandbox:   s.sandbox,
		HomePath:  s.layout.homeDir,
		SkillsDir: skillsDir,
		Skills:    settings.Skills,
	})
	if err != nil {
		return fmt.Errorf("write pi skills: %w", err)
	}

	tools := make([]toolSpec, 0, len(settings.Tools))
	for _, tool := range settings.Tools {
		tools = append(tools, toolSpec{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		})
	}

	content, err := json.Marshal(turnFile{
		Instructions: settings.Instructions,
		Tools:        tools,
	})
	if err != nil {
		return fmt.Errorf("encode pi turn: %w", err)
	}

	err = s.sandbox.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
		Path:    s.layout.turnFile(),
		Content: string(content),
	})
	if err != nil {
		return fmt.Errorf("write pi turn: %w", err)
	}

	return s.syncExtension(ctx, skills.Changed, toolsSignature(tools))
}

// syncExtension runs the extension's sync command when the skills or tools
// changed.
func (s *session) syncExtension(ctx context.Context, reload bool, signature string) error {
	s.mu.Lock()
	toolsChanged := signature != s.toolsSignature
	s.mu.Unlock()

	if !reload && !toolsChanged {
		return nil
	}

	command := syncCommand
	if reload {
		command += " reload"
	}

	_, err := s.conn.call(ctx, promptCommand{
		Type:    "prompt",
		Message: command,
	})
	if err != nil {
		return fmt.Errorf("apply pi turn configuration: %w", err)
	}

	s.mu.Lock()
	s.toolsSignature = signature
	s.mu.Unlock()

	return nil
}

// toolsSignature identifies a turn's tools.
func toolsSignature(tools []toolSpec) string {
	encoded, err := json.Marshal(tools)
	if err != nil {
		return ""
	}

	return string(encoded)
}

// selectModel selects the turn's model. A turn without one keeps the model
// of earlier turns, or uses the default gateway model when the AI Gateway
// authenticates Pi and no model was selected yet. It returns the active
// model, nil when Pi's choice is unknown.
func (s *session) selectModel(ctx context.Context, requested string) (*modelInfo, error) {
	s.mu.Lock()
	target := requested
	selected := s.modelSelected
	current := s.model
	s.mu.Unlock()

	if target == "" && !selected && s.setup.usesGateway {
		target = DefaultGatewayModel
	}

	if target == "" {
		return current, nil
	}

	models, err := s.availableModels(ctx)
	if err != nil {
		return nil, err
	}

	resolved, ok := resolveModel(models, target, s.setup.usesGateway)
	if !ok {
		if requested != "" {
			return nil, unsupported(fmt.Sprintf("Harness 'pi' has no model '%s' in its catalog.", requested))
		}

		return current, nil
	}

	if current == nil || current.Provider != resolved.Provider || current.ID != resolved.ID {
		_, err := s.conn.call(ctx, setModelCommand{
			Type:     "set_model",
			Provider: resolved.Provider,
			ModelID:  resolved.ID,
		})
		if err != nil {
			return nil, err
		}
	}

	s.mu.Lock()
	s.model = &resolved
	s.modelSelected = true
	s.mu.Unlock()

	return &resolved, nil
}

// availableModels returns the models whose provider is authenticated,
// listed once per session.
func (s *session) availableModels(ctx context.Context) ([]modelInfo, error) {
	s.mu.Lock()
	cached := s.models
	s.mu.Unlock()

	if cached != nil {
		return cached, nil
	}

	reply, err := s.conn.call(ctx, simpleCommand{
		Type: "get_available_models",
	})
	if err != nil {
		return nil, err
	}

	var available availableModelsData
	if err := json.Unmarshal(reply.Data, &available); err != nil {
		return nil, fmt.Errorf("decode pi models: %w", err)
	}

	models := available.Models
	if models == nil {
		models = []modelInfo{}
	}

	s.mu.Lock()
	s.models = models
	s.mu.Unlock()

	return models, nil
}

// promptContent is a prompt as Pi accepts it.
type promptContent struct {
	text   string
	images []imageBlock
}

// extractPrompt ports upstream extractPiPromptContent: the text parts joined
// with blank lines, followed by the images. Pi places images after the
// text, so text that follows an image cannot be preserved.
func extractPrompt(prompt model.UserMessage) (promptContent, error) {
	content, err := harness.ExtractPromptContent(prompt)
	if err != nil {
		return promptContent{}, err
	}

	if content.Parts == nil {
		return promptContent{
			text: content.Text,
		}, nil
	}

	var texts []string
	var images []imageBlock

	for _, part := range content.Parts {
		if part.Type == harness.PromptContentImage {
			images = append(images, imageBlock{
				Type:     "image",
				Data:     part.Data,
				MIMEType: part.MediaType,
			})

			continue
		}

		if len(images) > 0 {
			return promptContent{}, unsupported("Harness 'pi' cannot preserve text that follows an inline image in a user message.")
		}

		texts = append(texts, part.Text)
	}

	return promptContent{
		text:   strings.Join(texts, "\n\n"),
		images: images,
	}, nil
}

// validateResponseFormat rejects structured output, which Pi does not
// support.
func validateResponseFormat(format *model.ResponseFormat) error {
	if format != nil && format.Type == model.ResponseFormatJSON {
		return unsupported("Harness 'pi' does not support structured output.")
	}

	return nil
}

// unsupported returns a capability error of the adapter.
func unsupported(message string) error {
	return &harness.CapabilityUnsupportedError{
		Message:   message,
		HarnessID: HarnessID,
	}
}
