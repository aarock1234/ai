package claudecode

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
	"github.com/aarock1234/ai/pkg/model"
)

// continuePrompt drives a rerun turn forward. It must not be empty: the
// Anthropic API rejects an empty text block once the SDK marks it for
// caching.
const continuePrompt = "Continue."

// skillsDir is where the claude CLI discovers skills, relative to HOME.
const skillsDir = ".claude/skills"

// session is a Claude Code session over its bridge.
type session struct {
	harness   *Harness
	runtime   *bridgeadapter.Runtime
	sessionID string
	isResume  bool

	// env is the environment of the Claude Code process, with brokered
	// credentials replaced by their placeholders.
	env                   map[string]string
	credentialEnvironment map[string]string

	permissionMode       harness.PermissionMode
	builtinToolFiltering *harness.BuiltinToolFiltering
	observability        *harness.Observability

	mu sync.Mutex

	// continueOnNextPrompt asks a respawned bridge to rehydrate the working
	// directory's conversation on its first prompt when no exact
	// conversation is known.
	continueOnNextPrompt bool

	// rerunContinue makes the next ContinueTurn re-drive the conversation,
	// because the respawned bridge has no turn to attach to.
	rerunContinue bool

	// claudeSessionID is the latest Claude conversation the bridge reported.
	// Each turn may fork a new one.
	claudeSessionID string
}

// SessionID returns the harness session ID.
func (s *session) SessionID() string {
	return s.sessionID
}

// IsResume reports whether the session resumed from lifecycle state.
func (s *session) IsResume() bool {
	return s.isResume
}

// PromptTurn starts a turn with a fresh prompt.
func (s *session) PromptTurn(ctx context.Context, options harness.PromptTurnOptions) (harness.PromptControl, error) {
	prompt, err := harness.ExtractPromptContent(options.Prompt)
	if err != nil {
		return nil, err
	}

	turn, err := s.prepareTurn(ctx, options.TurnSettings, options.ResponseFormat, options.Emit)
	if err != nil {
		return nil, err
	}

	// A turn whose context was already cancelled has settled; starting it
	// would run an unattended turn nobody observes.
	if turn.Settled() {
		return turn.Control(), nil
	}

	s.mu.Lock()
	resume := s.resumeFields(s.continueOnNextPrompt)
	s.continueOnNextPrompt = false
	s.rerunContinue = false
	s.mu.Unlock()

	command := s.startCommand(startCommandOptions{
		prompt:         prompt,
		settings:       options.TurnSettings,
		responseFormat: options.ResponseFormat,
		resume:         resume,
	})

	return s.send(turn, command)
}

// ContinueTurn continues the in-flight turn. An attached or replaying
// bridge streams the turn into the wired turn without a new start command,
// which would begin a new turn. A rerun re-drives the conversation from its
// persisted state, recomputing the work in flight at the suspension.
func (s *session) ContinueTurn(ctx context.Context, options harness.ContinueTurnOptions) (harness.PromptControl, error) {
	turn, err := s.prepareTurn(ctx, options.TurnSettings, options.ResponseFormat, options.Emit)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	rerun := s.rerunContinue && !turn.Settled()
	if rerun {
		s.rerunContinue = false
		s.continueOnNextPrompt = false
	}
	resume := s.resumeFields(true)
	s.mu.Unlock()

	if !rerun {
		return turn.Control(), nil
	}

	command := s.startCommand(startCommandOptions{
		prompt: harness.PromptContent{
			Text: continuePrompt,
		},
		settings:       options.TurnSettings,
		responseFormat: options.ResponseFormat,
		resume:         resume,
	})

	return s.send(turn, command)
}

// prepareTurn validates the response format, writes the turn's skills and
// wires the turn.
func (s *session) prepareTurn(ctx context.Context, settings harness.TurnSettings, responseFormat *model.ResponseFormat, emit func(part harness.StreamPart)) (*bridgeadapter.Turn, error) {
	if err := bridgeadapter.ValidateResponseFormat(HarnessID, responseFormat); err != nil {
		return nil, err
	}

	_, err := harness.WriteSkills(ctx, harness.WriteSkillsOptions{
		Sandbox:         s.runtime.Sandbox(),
		HomePath:        s.runtime.HomeDir(),
		SkillsDir:       skillsDir,
		Skills:          settings.Skills,
		TrailingNewline: true,
	})
	if err != nil {
		return nil, fmt.Errorf("write claude code skills: %w", err)
	}

	return s.runtime.BeginTurn(ctx, bridgeadapter.TurnOptions{
		Emit: emit,
	}), nil
}

// send sends a turn's start command, failing the turn when it cannot be
// sent.
func (s *session) send(turn *bridgeadapter.Turn, command startCommand) (harness.PromptControl, error) {
	if err := s.runtime.Send(command); err != nil {
		turn.Fail(err)

		return nil, err
	}

	return turn.Control(), nil
}

// resumeFields selects how a new SDK query resumes: the exact conversation
// when it is known, and otherwise the working directory's most recent
// conversation when fallback is set. The caller holds mu.
func (s *session) resumeFields(fallback bool) resumeFields {
	if s.claudeSessionID != "" {
		return resumeFields{
			ResumeSessionID: s.claudeSessionID,
		}
	}

	return resumeFields{
		Continue: fallback,
	}
}

// Compact submits the /compact slash command, Claude Code's only compaction
// trigger. Claude reports the compaction as a CompactionPart.
func (s *session) Compact(_ context.Context, customInstructions string) error {
	text := "/compact"
	if trimmed := strings.TrimSpace(customInstructions); trimmed != "" {
		text += " " + trimmed
	}

	return s.runtime.Send(bridge.UserMessageCommand{
		Type: "user-message",
		Text: text,
	})
}

// SuspendTurn freezes the active turn and returns the state to continue it.
func (s *session) SuspendTurn(_ context.Context) (harness.ContinueTurnState, error) {
	coordinates, err := s.runtime.SuspendTurn()
	if err != nil {
		return harness.ContinueTurnState{}, err
	}

	data, err := s.liveStateData(coordinates)
	if err != nil {
		return harness.ContinueTurnState{}, err
	}

	return harness.NewContinueTurnState(HarnessID, data), nil
}

// Detach disconnects from the bridge, which keeps running.
func (s *session) Detach(_ context.Context) (harness.ResumeSessionState, error) {
	coordinates, err := s.runtime.Detach()
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	data, err := s.liveStateData(coordinates)
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	return harness.NewResumeSessionState(HarnessID, data), nil
}

// Stop stops the bridge. The conversation lives in the working directory,
// which the sandbox preserves, so the resume state needs no bridge.
func (s *session) Stop(ctx context.Context) (harness.ResumeSessionState, error) {
	reply, err := s.runtime.Stop(ctx)
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	data, err := stopStateData(reply, s.ownStateData())
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	return harness.NewResumeSessionState(HarnessID, data), nil
}

// Destroy stops the bridge without returning state.
func (s *session) Destroy(ctx context.Context) error {
	return s.runtime.Destroy(ctx)
}

// liveStateData encodes the state of a session whose bridge keeps running.
func (s *session) liveStateData(coordinates bridgeadapter.Coordinates) (json.RawMessage, error) {
	data := s.ownStateData()
	data.Bridge = &coordinates

	encoded, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("encode claude code lifecycle state: %w", err)
	}

	return encoded, nil
}

// ownStateData returns the state the adapter records itself.
func (s *session) ownStateData() stateData {
	s.mu.Lock()
	defer s.mu.Unlock()

	return stateData{
		SandboxCredentialEnvironment: s.credentialEnvironment,
		ClaudeSessionID:              s.claudeSessionID,
	}
}

// recordClaudeSession records the conversation a finish part reports.
func (s *session) recordClaudeSession(part harness.FinishPart) {
	metadata, ok := part.HarnessMetadata[HarnessID].(map[string]any)
	if !ok {
		return
	}

	sessionID, ok := metadata["sessionId"].(string)
	if !ok || sessionID == "" {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.claudeSessionID = sessionID
}
