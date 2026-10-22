package deepagents

import (
	"context"
	"fmt"
	"regexp"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// skillsDir is where the harness writes skills, relative to HOME, outside
// the working directory so that they cannot clash with the code in it.
const skillsDir = ".agents/skills"

// skillNamePattern accepts lowercase alphanumeric skill names with hyphens,
// 1 to 64 characters long.
var skillNamePattern = regexp.MustCompile(`^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$`)

// session is a Deep Agents session over its bridge.
type session struct {
	harness   *Harness
	runtime   *bridgeadapter.Runtime
	sessionID string
	isResume  bool

	// credentialEnvironment holds the placeholders of brokered credentials,
	// and is nil when the sandbox cannot broker them.
	credentialEnvironment map[string]string

	permissionMode       harness.PermissionMode
	builtinToolFiltering *harness.BuiltinToolFiltering
	headers              map[string]string

	// skillsPaths are the working directory's skills, for repository
	// skills, and the harness's skills under HOME, which take precedence.
	skillsPaths []string
}

// SessionID returns the harness session ID.
func (s *session) SessionID() string {
	return s.sessionID
}

// IsResume reports whether the session resumed from lifecycle state.
func (s *session) IsResume() bool {
	return s.isResume
}

// PromptTurn writes the turn's skills and starts a turn with a fresh
// prompt.
func (s *session) PromptTurn(ctx context.Context, options harness.PromptTurnOptions) (harness.PromptControl, error) {
	if err := bridgeadapter.ValidateResponseFormat(HarnessID, options.ResponseFormat); err != nil {
		return nil, err
	}

	prompt, err := promptText(options.Prompt)
	if err != nil {
		return nil, err
	}

	skills, err := harness.WriteSkills(ctx, harness.WriteSkillsOptions{
		Sandbox:          s.runtime.Sandbox(),
		HomePath:         s.runtime.HomeDir(),
		SkillsDir:        skillsDir,
		Skills:           options.Skills,
		SkillNamePattern: skillNamePattern,
		FilePathMode:     harness.SkillPathsStripLeadingSlashes,
	})
	if err != nil {
		return nil, fmt.Errorf("write deepagents skills: %w", err)
	}

	turn := s.beginTurn(ctx, options.Emit)
	if turn.Settled() {
		return turn.Control(), nil
	}

	command := s.startCommand(startCommandOptions{
		prompt:         prompt,
		settings:       options.TurnSettings,
		responseFormat: options.ResponseFormat,
		skillsChanged:  skills.Changed,
	})

	if err := s.runtime.Send(command); err != nil {
		turn.Fail(err)

		return nil, err
	}

	return turn.Control(), nil
}

// ContinueTurn wires the in-flight turn. The attached or replaying bridge
// streams the rest of the turn; a start command would begin a new turn and
// clear the bridge's replay log.
func (s *session) ContinueTurn(ctx context.Context, options harness.ContinueTurnOptions) (harness.PromptControl, error) {
	if err := bridgeadapter.ValidateResponseFormat(HarnessID, options.ResponseFormat); err != nil {
		return nil, err
	}

	return s.beginTurn(ctx, options.Emit).Control(), nil
}

// beginTurn wires a turn into the bridge's frames.
func (s *session) beginTurn(ctx context.Context, emit func(part harness.StreamPart)) *bridgeadapter.Turn {
	return s.runtime.BeginTurn(ctx, bridgeadapter.TurnOptions{
		Emit: emit,
	})
}

// Compact is unsupported.
func (s *session) Compact(context.Context, string) error {
	return &harness.CapabilityUnsupportedError{
		Message:   "Harness 'deepagents' does not support manual compaction yet.",
		HarnessID: HarnessID,
	}
}

// SuspendTurn freezes the active turn at the channel's cursor, leaving the
// bridge running so that the next process replays the rest of the turn.
func (s *session) SuspendTurn(_ context.Context) (harness.ContinueTurnState, error) {
	coordinates, err := s.runtime.SuspendTurn()
	if err != nil {
		return harness.ContinueTurnState{}, err
	}

	data, err := encodeStateData(s.liveStateData(coordinates))
	if err != nil {
		return harness.ContinueTurnState{}, err
	}

	return harness.NewContinueTurnState(HarnessID, data), nil
}

// Detach parks the session between turns: it closes the host connection and
// leaves the bridge running for a later reattach.
func (s *session) Detach(_ context.Context) (harness.ResumeSessionState, error) {
	coordinates, err := s.runtime.Detach()
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	data, err := encodeStateData(s.liveStateData(coordinates))
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	return harness.NewResumeSessionState(HarnessID, data), nil
}

// Stop stops the bridge, which saves the conversation checkpoint a resumed
// bridge loads.
func (s *session) Stop(ctx context.Context) (harness.ResumeSessionState, error) {
	if _, err := s.runtime.Stop(ctx); err != nil {
		return harness.ResumeSessionState{}, err
	}

	data, err := encodeStateData(stateData{
		SandboxCredentialEnvironment: s.credentialEnvironment,
	})
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	return harness.NewResumeSessionState(HarnessID, data), nil
}

// Destroy stops the bridge, which deletes the conversation checkpoint,
// without returning state.
func (s *session) Destroy(ctx context.Context) error {
	return s.runtime.Destroy(ctx)
}

// liveStateData is the state of a session whose bridge keeps running.
func (s *session) liveStateData(coordinates bridgeadapter.Coordinates) stateData {
	return stateData{
		Bridge:                       &coordinates,
		SandboxCredentialEnvironment: s.credentialEnvironment,
	}
}
