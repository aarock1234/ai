package codex

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// continuePrompt drives a rerun turn forward instead of handing the runtime
// an empty user message.
const continuePrompt = "Continue."

// skillsDir is where Codex discovers skills, relative to HOME.
const skillsDir = ".agents/skills"

// session is a Codex session over its bridge.
type session struct {
	harness   *Harness
	runtime   *bridgeadapter.Runtime
	sessionID string
	isResume  bool

	credentialEnvironment map[string]string

	permissionMode       harness.PermissionMode
	builtinToolFiltering *harness.BuiltinToolFiltering
	observability        *harness.Observability
	headers              map[string]string

	mu sync.Mutex

	// pendingResumeThreadID is sent on the first prompt of a respawned
	// bridge, so that it resumes the persisted thread.
	pendingResumeThreadID string

	// latestThreadID is the thread the bridge last announced, seeded from
	// the lifecycle state.
	latestThreadID string

	// fingerprint identifies the configuration of the current thread.
	fingerprint string

	// rerunContinue makes the next ContinueTurn re-drive the thread, because
	// the respawned bridge has no turn to attach to.
	rerunContinue bool
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

	prompt, err := harness.ExtractPromptContent(options.Prompt)
	if err != nil {
		return nil, err
	}

	restartThread, err := s.synchronizeTurnConfiguration(ctx, options.TurnSettings)
	if err != nil {
		return nil, err
	}

	turn := s.beginTurn(ctx, options.Emit)
	if turn.Settled() {
		return turn.Control(), nil
	}

	s.mu.Lock()
	resumeThreadID := s.pendingResumeThreadID
	s.pendingResumeThreadID = ""
	s.rerunContinue = false
	s.mu.Unlock()

	command := s.startCommand(startCommandOptions{
		prompt:         prompt,
		settings:       options.TurnSettings,
		responseFormat: options.ResponseFormat,
		resumeThreadID: resumeThreadID,
		restartThread:  restartThread,
	})

	return s.send(turn, command)
}

// ContinueTurn continues the in-flight turn. An attached or replaying
// bridge streams the turn into the wired turn without a new start command,
// which would begin a new turn. A rerun re-drives the thread with
// thread/resume, recomputing the work in flight at the suspension.
func (s *session) ContinueTurn(ctx context.Context, options harness.ContinueTurnOptions) (harness.PromptControl, error) {
	if err := bridgeadapter.ValidateResponseFormat(HarnessID, options.ResponseFormat); err != nil {
		return nil, err
	}

	restartThread, err := s.synchronizeTurnConfiguration(ctx, options.TurnSettings)
	if err != nil {
		return nil, err
	}

	turn := s.beginTurn(ctx, options.Emit)

	s.mu.Lock()
	rerun := s.rerunContinue
	threadID := s.pendingResumeThreadID
	if threadID == "" {
		threadID = s.latestThreadID
	}

	if rerun {
		s.rerunContinue = false
		s.pendingResumeThreadID = ""
	}
	s.mu.Unlock()

	if !rerun || turn.Settled() {
		return turn.Control(), nil
	}

	command := s.startCommand(startCommandOptions{
		prompt: harness.PromptContent{
			Text: continuePrompt,
		},
		settings:       options.TurnSettings,
		responseFormat: options.ResponseFormat,
		resumeThreadID: threadID,
		restartThread:  restartThread,
	})

	return s.send(turn, command)
}

// synchronizeTurnConfiguration writes the turn's skills and reports whether
// the thread must restart because its skills, instructions or tools
// changed. Legacy fingerprints are replaced once without a restart.
func (s *session) synchronizeTurnConfiguration(ctx context.Context, settings harness.TurnSettings) (bool, error) {
	skills, err := harness.WriteSkills(ctx, harness.WriteSkillsOptions{
		Sandbox:   s.runtime.Sandbox(),
		HomePath:  s.runtime.HomeDir(),
		SkillsDir: skillsDir,
		Skills:    settings.Skills,
	})
	if err != nil {
		return false, fmt.Errorf("write codex skills: %w", err)
	}

	next, err := fingerprintTurnConfiguration(settings.Instructions, settings.Tools)
	if err != nil {
		return false, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	configurationChanged := strings.HasPrefix(s.fingerprint, fingerprintPrefix) && s.fingerprint != next
	restartThread := s.latestThreadID != "" && (skills.Changed || configurationChanged)

	s.fingerprint = next
	if restartThread {
		s.pendingResumeThreadID = ""
	}

	return restartThread, nil
}

// beginTurn wires a turn. Codex turns always accept user messages.
func (s *session) beginTurn(ctx context.Context, emit func(part harness.StreamPart)) *bridgeadapter.Turn {
	return s.runtime.BeginTurn(ctx, bridgeadapter.TurnOptions{
		Emit:               emit,
		AcceptUserMessages: true,
	})
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

// Compact is unsupported: Codex compacts its context automatically inside
// its turn loop and exposes no manual trigger.
func (s *session) Compact(context.Context, string) error {
	return &harness.CapabilityUnsupportedError{
		Message:   "Harness 'codex' does not support manual compaction; Codex auto-compacts its context internally.",
		HarnessID: HarnessID,
	}
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

// Stop stops the bridge, whose reply carries the thread to resume.
func (s *session) Stop(ctx context.Context) (harness.ResumeSessionState, error) {
	reply, err := s.runtime.Stop(ctx)
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	s.mu.Lock()
	own := stateData{
		TurnConfigurationFingerprint: s.fingerprint,
		SandboxCredentialEnvironment: s.credentialEnvironment,
	}
	s.mu.Unlock()

	data, err := stopStateData(reply, own)
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
		ThreadID:                     s.latestThreadID,
		TurnConfigurationFingerprint: s.fingerprint,
		Bridge:                       &coordinates,
		SandboxCredentialEnvironment: s.credentialEnvironment,
	}
	s.mu.Unlock()

	encoded, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("encode codex lifecycle state: %w", err)
	}

	return encoded, nil
}

// recordThread records the thread the bridge announced.
func (s *session) recordThread(threadID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.latestThreadID = threadID
}
