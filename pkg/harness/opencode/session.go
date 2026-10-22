package opencode

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// continuePrompt drives a rerun turn forward instead of handing OpenCode an
// empty user message.
const continuePrompt = "Continue."

// skillsDir is where OpenCode discovers skills, relative to HOME.
const skillsDir = ".agents/skills"

// session is an OpenCode session over its bridge.
type session struct {
	harness   *Harness
	runtime   *bridgeadapter.Runtime
	sessionID string
	isResume  bool

	credentialEnvironment map[string]string
	openCodeConfig        map[string]any

	// transformModel maps turn models for a brokered GitLab login, and is
	// nil otherwise.
	transformModel func(modelName string) (string, error)

	permissionMode           harness.PermissionMode
	observability            *harness.Observability
	headers                  map[string]string
	inactiveBuiltinToolNames []string

	mu sync.Mutex

	// pendingResumeSessionID is sent on the first prompt of a respawned
	// bridge, so that it resumes the persisted OpenCode session.
	pendingResumeSessionID string

	// latestSessionID is the OpenCode session the bridge last announced,
	// seeded from the lifecycle state.
	latestSessionID string

	// selectedModel is the latest model a turn selected. Later turns and
	// compactions without a model keep using it.
	selectedModel string

	// rerunContinue makes the next ContinueTurn re-drive the OpenCode
	// session, because the respawned bridge has no turn to attach to.
	rerunContinue bool

	// activeTurn is the latest prompt turn, which runs until it settles.
	activeTurn *bridgeadapter.Turn

	// pendingCompactions are the compaction parts of a manual compaction,
	// emitted at the start of the next turn.
	pendingCompactions []harness.StreamPart
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
	if err := bridgeadapter.ValidateResponseFormat(HarnessID, options.ResponseFormat); err != nil {
		return nil, err
	}

	prompt, err := promptText(options.Prompt)
	if err != nil {
		return nil, err
	}

	turnModel, err := s.selectModel(options.Model)
	if err != nil {
		return nil, err
	}

	skillsChanged, err := s.writeSkills(ctx, options.Skills)
	if err != nil {
		return nil, err
	}

	turn := s.beginTurn(ctx, options.Emit)
	if turn.Settled() {
		return turn.Control(), nil
	}

	s.mu.Lock()
	resumeID := s.takeResumeSessionID()
	s.rerunContinue = false
	s.mu.Unlock()

	command := s.promptCommand(promptCommandOptions{
		prompt:         prompt,
		settings:       options.TurnSettings,
		responseFormat: options.ResponseFormat,
		model:          turnModel,
		skillsChanged:  skillsChanged,
		resumeID:       resumeID,
	})

	return s.send(turn, command)
}

// ContinueTurn continues the in-flight turn. An attached or replaying
// bridge streams the turn into the wired turn without a new start command,
// which would begin a new turn. A rerun re-drives the OpenCode session,
// recomputing the work in flight at the suspension.
func (s *session) ContinueTurn(ctx context.Context, options harness.ContinueTurnOptions) (harness.PromptControl, error) {
	if err := bridgeadapter.ValidateResponseFormat(HarnessID, options.ResponseFormat); err != nil {
		return nil, err
	}

	s.mu.Lock()
	rerun := s.rerunContinue
	s.mu.Unlock()

	var turnModel string

	if rerun {
		selected, err := s.selectModel(options.Model)
		if err != nil {
			return nil, err
		}

		turnModel = selected
	}

	skillsChanged, err := s.writeSkills(ctx, options.Skills)
	if err != nil {
		return nil, err
	}

	turn := s.beginTurn(ctx, options.Emit)
	if !rerun || turn.Settled() {
		return turn.Control(), nil
	}

	s.mu.Lock()
	resumeID := s.takeResumeSessionID()
	s.rerunContinue = false
	s.mu.Unlock()

	command := s.promptCommand(promptCommandOptions{
		prompt:         continuePrompt,
		settings:       options.TurnSettings,
		responseFormat: options.ResponseFormat,
		model:          turnModel,
		skillsChanged:  skillsChanged,
		resumeID:       resumeID,
	})

	return s.send(turn, command)
}

// selectModel returns the model of a turn: the requested model, which
// becomes the selected model, or the selected model of an earlier turn.
func (s *session) selectModel(requested string) (string, error) {
	s.mu.Lock()
	if requested != "" {
		s.selectedModel = requested
	}

	selected := s.selectedModel
	s.mu.Unlock()

	return s.wireModel(selected)
}

// wireModel maps a model for a brokered GitLab login.
func (s *session) wireModel(modelName string) (string, error) {
	if s.transformModel == nil {
		return modelName, nil
	}

	return s.transformModel(modelName)
}

// writeSkills writes the turn's skills and reports whether they changed,
// which makes OpenCode reload them.
func (s *session) writeSkills(ctx context.Context, skills []harness.Skill) (bool, error) {
	result, err := harness.WriteSkills(ctx, harness.WriteSkillsOptions{
		Sandbox:   s.runtime.Sandbox(),
		HomePath:  s.runtime.HomeDir(),
		SkillsDir: skillsDir,
		Skills:    skills,
	})
	if err != nil {
		return false, fmt.Errorf("write opencode skills: %w", err)
	}

	return result.Changed, nil
}

// beginTurn emits the compactions of a manual compaction that ran since the
// previous turn, then wires the turn. A turn whose context is already
// cancelled drops them, as upstream.
func (s *session) beginTurn(ctx context.Context, emit func(part harness.StreamPart)) *bridgeadapter.Turn {
	s.mu.Lock()
	compactions := s.pendingCompactions
	s.pendingCompactions = nil
	s.mu.Unlock()

	if ctx.Err() == nil {
		for _, part := range compactions {
			emit(part)
		}
	}

	turn := s.runtime.BeginTurn(ctx, bridgeadapter.TurnOptions{
		Emit: emit,
	})

	s.mu.Lock()
	s.activeTurn = turn
	s.mu.Unlock()

	return turn
}

// takeResumeSessionID returns the OpenCode session a start command resumes:
// the persisted session of a respawned bridge's first prompt, and the
// latest announced session otherwise. The caller holds mu.
func (s *session) takeResumeSessionID() string {
	resumeID := s.pendingResumeSessionID
	s.pendingResumeSessionID = ""

	if resumeID == "" {
		resumeID = s.latestSessionID
	}

	return resumeID
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

// Compact runs OpenCode's native compaction between turns and waits for it.
// OpenCode reports the compaction as a CompactionPart, which the next turn
// emits first. Ctx cancels the compaction.
func (s *session) Compact(ctx context.Context, customInstructions string) error {
	if strings.TrimSpace(customInstructions) != "" {
		return unsupported("Harness 'opencode' supports native manual compaction, but OpenCode does not expose custom compaction instructions through the supported API.")
	}

	if s.hasActiveTurn() {
		return unsupported("Harness 'opencode' supports manual compaction between turns; compacting during an active turn is not supported by the bridge transport.")
	}

	s.mu.Lock()
	selected := s.selectedModel
	resumeID := s.latestSessionID
	s.mu.Unlock()

	compactionModel, err := s.wireModel(selected)
	if err != nil {
		return err
	}

	var compactions []harness.StreamPart

	// The compaction runs as a turn of its own whose events are not
	// forwarded: its compaction parts are kept for the next turn.
	run := s.runtime.BeginTurn(ctx, bridgeadapter.TurnOptions{
		Emit: func(part harness.StreamPart) {
			if _, ok := part.(harness.CompactionPart); ok {
				compactions = append(compactions, part)
			}
		},
	})
	if run.Settled() {
		return run.Err()
	}

	command := s.baseCommand(baseCommandOptions{
		operation: operationCompact,
		model:     compactionModel,
		resumeID:  resumeID,
	})

	if _, err := s.send(run, command); err != nil {
		return err
	}

	<-run.Done()

	s.mu.Lock()
	s.pendingCompactions = append(s.pendingCompactions, compactions...)
	s.mu.Unlock()

	return run.Err()
}

// hasActiveTurn reports whether a prompt turn is still running.
func (s *session) hasActiveTurn() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.activeTurn != nil && !s.activeTurn.Settled()
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

// Stop stops the bridge, whose reply carries the OpenCode session to
// resume.
func (s *session) Stop(ctx context.Context) (harness.ResumeSessionState, error) {
	reply, err := s.runtime.Stop(ctx)
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	data, err := stopStateData(reply, s.credentialEnvironment)
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
	s.mu.Lock()
	data := stateData{
		OpenCodeSessionID:            s.latestSessionID,
		Bridge:                       &coordinates,
		SandboxCredentialEnvironment: s.credentialEnvironment,
	}
	s.mu.Unlock()

	encoded, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("encode opencode lifecycle state: %w", err)
	}

	return encoded, nil
}

// recordSession records the OpenCode session the bridge announced.
func (s *session) recordSession(sessionID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.latestSessionID = sessionID
}
