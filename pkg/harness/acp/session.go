package acp

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// terminateGracePeriod bounds the wait for a bridge to exit after a stop or
// destroy command.
const terminateGracePeriod = 5 * time.Second

// sessionLaunch describes the bridge a session talks to.
type sessionLaunch struct {
	channel *bridge.Channel

	// process is the spawned bridge, or nil when the session reattached to
	// a running bridge.
	process *bridgeadapter.BridgeProcess

	bridgePort  int
	bridgeToken string

	// turnInFlight reports a turn the session must continue rather than
	// start a fresh prompt.
	turnInFlight bool

	// replayOnly reports a bridge that only replays a recovered turn from
	// disk; lossyRerun a bridge that reruns the interrupted prompt.
	replayOnly bool
	lossyRerun bool

	turnStartConfig *turnStartConfig
	recovery        *recoveryStatus
	restoration     *restoration
	acpSessionID    string
}

// sessionConfig is everything a session needs from its startup.
type sessionConfig struct {
	harness   *Harness
	sessionID string
	isResume  bool
	launch    sessionLaunch

	configuration          sessionConfiguration
	implementationIdentity string
	permissionMode         harness.PermissionMode
	debug                  *harness.DebugConfig

	sandbox   sandbox.Session
	sandboxID string
	paths     sessionPaths

	sandboxCredentialEnv    map[string]string
	initialGuidanceApplied  bool
	instructionsFingerprint string
}

// sessionConfig collects the configuration of a session over a launched or
// reattached bridge.
func (s *sessionStartup) sessionConfig(launch sessionLaunch) sessionConfig {
	config := sessionConfig{
		harness:                s.harness,
		sessionID:              s.options.SessionID,
		isResume:               s.isResume(),
		launch:                 launch,
		configuration:          s.configuration,
		implementationIdentity: s.implementationIdentity,
		permissionMode:         s.permissionMode,
		debug:                  s.debug,
		sandbox:                s.sandbox,
		sandboxID:              s.sandboxID,
		paths:                  s.paths,
		sandboxCredentialEnv:   s.credentials.sandboxCredentialEnv,
		initialGuidanceApplied: s.lifecycle.initialGuidanceApplied(),
	}

	if s.lifecycle != nil {
		config.instructionsFingerprint = s.lifecycle.InstructionsFingerprint
	}

	return config
}

// session is an ACP harness session over one bridge.
type session struct {
	config  sessionConfig
	channel *bridge.Channel
	process *bridgeadapter.BridgeProcess

	// turns tracks the goroutines that read the channel for a turn.
	turns sync.WaitGroup

	mu                      sync.Mutex
	stopped                 bool
	turnInFlight            bool
	initialGuidanceApplied  bool
	instructionsFingerprint string
	acpSessionID            string
	turnStartConfig         *turnStartConfig
	bufferedQuestionResults []bufferedQuestionResult
}

// newSession returns a session over a connected bridge.
func newSession(config sessionConfig) *session {
	return &session{
		config:                  config,
		channel:                 config.launch.channel,
		process:                 config.launch.process,
		turnInFlight:            config.launch.turnInFlight,
		initialGuidanceApplied:  config.initialGuidanceApplied,
		instructionsFingerprint: config.instructionsFingerprint,
		acpSessionID:            config.launch.acpSessionID,
		turnStartConfig:         config.launch.turnStartConfig,
	}
}

// SessionID returns the harness session ID.
func (s *session) SessionID() string {
	return s.config.sessionID
}

// IsResume reports a session started from lifecycle state.
func (s *session) IsResume() bool {
	return s.config.isResume
}

// PromptTurn starts a prompt turn.
func (s *session) PromptTurn(ctx context.Context, options harness.PromptTurnOptions) (harness.PromptControl, error) {
	if err := s.prepareTurn(ctx, options.TurnSettings, options.ResponseFormat); err != nil {
		return nil, err
	}

	harnessID := s.harnessID()

	if s.config.launch.replayOnly {
		return nil, fmt.Errorf("%s recovered this turn through disk replay only and has no restored acp process for a subsequent prompt", harnessID)
	}

	if ctx.Err() != nil {
		return nil, context.Cause(ctx)
	}

	prompt, err := promptTextBlocks(options.Prompt, harnessID)
	if err != nil {
		return nil, err
	}

	config, err := s.config.configuration.newTurnStartConfig(turnStartInput{
		prompt:         prompt,
		tools:          nonNil(options.Tools),
		permissionMode: s.config.permissionMode,
		debug:          s.config.debug,
		responseFormat: responseFormatWire(options.ResponseFormat),
		model:          options.Model,
	})
	if err != nil {
		return nil, err
	}

	instructionsFingerprint, err := instructionsFingerprintOf(options.Instructions)
	if err != nil {
		return nil, err
	}

	err = s.startPrompt(promptStart{
		prompt:                  prompt,
		options:                 options,
		config:                  config,
		instructionsFingerprint: instructionsFingerprint,
	})
	if err != nil {
		return nil, err
	}

	return s.startTurn(ctx, options.Emit), nil
}

// promptStart is a prompt turn ready to be sent.
type promptStart struct {
	prompt                  []textBlock
	options                 harness.PromptTurnOptions
	config                  turnStartConfig
	instructionsFingerprint string
}

// startPrompt sends the start command of a prompt turn and records the
// turn's configuration.
func (s *session) startPrompt(start promptStart) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.turnInFlight {
		return fmt.Errorf("%s cannot start a new acp prompt while a turn is in flight", s.harnessID())
	}

	settings := s.config.harness.settings
	mapping := settings.InstructionMapping

	prompt := start.prompt
	if s.shouldPrependGuidance(start.instructionsFingerprint) {
		prompt = prependInstructionGuidance(prompt, start.options.Instructions)
	}

	command := s.baseStartCommand()
	command.Prompt = prompt
	command.ResponseFormat = start.config.ResponseFormat
	command.OutputSchemaMapping = start.config.OutputSchemaMapping
	command.TurnStartConfig = start.config

	if mapping != nil {
		command.Instructions = start.options.Instructions
	}

	if start.options.Model != "" {
		command.Model = start.options.Model
		command.ModelMapping = &settings.ModelMapping
	}

	if start.options.Tools != nil {
		command.Tools = start.config.Tools
	}

	if err := s.channel.Send(command); err != nil {
		return err
	}

	s.turnInFlight = true
	s.turnStartConfig = &start.config
	s.initialGuidanceApplied = true
	s.instructionsFingerprint = start.instructionsFingerprint

	return nil
}

// shouldPrependGuidance reports whether changed instructions are prepended
// to the prompt: always without an instruction mapping, and after the
// first prompt with a session or launch mapping, which only apply when the
// ACP session starts. The caller holds mu.
func (s *session) shouldPrependGuidance(instructionsFingerprint string) bool {
	mapping := s.config.harness.settings.InstructionMapping

	if mapping != nil && mapping.Type == InstructionsFilesystem {
		return false
	}

	if s.instructionsFingerprint == instructionsFingerprint {
		return false
	}

	return mapping == nil || s.initialGuidanceApplied
}

// baseStartCommand returns the session-level fields of a start command.
func (s *session) baseStartCommand() startCommand {
	settings := s.config.harness.settings

	return startCommand{
		Type:                  "start",
		Debug:                 s.config.debug,
		BuiltinTools:          s.config.configuration.builtinTools,
		PermissionMode:        s.config.permissionMode,
		PermissionModeMapping: settings.PermissionModeMapping,
		InstructionMapping:    settings.InstructionMapping,
		MCPServers:            settings.MCPServers,
	}
}

// ContinueTurn continues the in-flight turn: it attaches to a live or
// replaying bridge, or reruns the persisted prompt after a lossy recovery.
func (s *session) ContinueTurn(ctx context.Context, options harness.ContinueTurnOptions) (harness.PromptControl, error) {
	if err := s.prepareTurn(ctx, options.TurnSettings, options.ResponseFormat); err != nil {
		return nil, err
	}

	if err := s.resumeTurn(options); err != nil {
		return nil, err
	}

	return s.startTurn(ctx, options.Emit), nil
}

// resumeTurn checks the in-flight turn and, after a lossy recovery, sends
// the rerun.
func (s *session) resumeTurn(options harness.ContinueTurnOptions) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	harnessID := s.harnessID()

	if !s.turnInFlight {
		return fmt.Errorf("%s has no in-flight acp turn to continue", harnessID)
	}

	if !s.config.launch.lossyRerun {
		return nil
	}

	if s.turnStartConfig == nil || s.acpSessionID == "" {
		return fmt.Errorf("%s cannot perform lossy acp rerun without persisted start configuration and an acp session identifier", harnessID)
	}

	if err := assertRecoveryToolCatalog(s.turnStartConfig.Tools, nonNil(options.Tools)); err != nil {
		return err
	}

	return s.channel.Send(s.rerunCommand(options.Instructions))
}

// rerunCommand returns the start command that reruns the persisted prompt
// against the recovered ACP session. The caller holds mu.
func (s *session) rerunCommand(instructions string) startCommand {
	config := *s.turnStartConfig

	reason := "bridge process loss"
	if s.config.launch.recovery != nil {
		reason = s.config.launch.recovery.Reason
	}

	command := s.baseStartCommand()
	command.Prompt = config.Prompt
	command.Debug = config.Debug
	command.BuiltinTools = config.BuiltinTools
	command.PermissionMode = config.PermissionMode
	command.PermissionModeMapping = config.PermissionModeMapping
	command.Model = config.Model
	command.ModelMapping = config.ModelMapping
	command.ResponseFormat = config.ResponseFormat
	command.OutputSchemaMapping = config.OutputSchemaMapping
	command.Tools = config.Tools
	command.TurnStartConfig = config
	command.RecoveryMode = &recoveryMode{
		Type:         recoveryLossyRerun,
		ACPSessionID: s.acpSessionID,
		Reason:       reason,
	}

	if command.InstructionMapping != nil {
		command.Instructions = instructions
	}

	return command
}

// prepareTurn writes the turn's skills and instructions and checks its
// response format.
func (s *session) prepareTurn(ctx context.Context, settings harness.TurnSettings, responseFormat *model.ResponseFormat) error {
	if err := s.synchronizeSkills(ctx, settings.Skills); err != nil {
		return err
	}

	if err := s.synchronizeInstructions(ctx, settings.Instructions); err != nil {
		return err
	}

	return s.checkResponseFormat(responseFormat)
}

// synchronizeSkills writes the turn's skills into the native skills
// directory.
func (s *session) synchronizeSkills(ctx context.Context, skills []harness.Skill) error {
	if err := validateSkills(skills); err != nil {
		return err
	}

	_, err := harness.WriteSkills(ctx, harness.WriteSkillsOptions{
		Sandbox:          s.config.sandbox,
		HomePath:         s.config.paths.homeDir,
		SkillsDir:        s.config.paths.skillsDir,
		Skills:           skills,
		SkillNamePattern: skillNamePattern,
	})

	return err
}

// synchronizeInstructions writes the instructions of a filesystem
// instruction mapping.
func (s *session) synchronizeInstructions(ctx context.Context, instructions string) error {
	mapping := s.config.harness.settings.InstructionMapping
	if mapping == nil || mapping.Type != InstructionsFilesystem {
		return nil
	}

	_, err := harness.WriteInstructions(ctx, harness.WriteInstructionsOptions{
		Sandbox:          s.config.sandbox,
		HomePath:         s.config.paths.homeDir,
		InstructionsFile: mapping.File,
		Instructions:     instructions,
	})

	return err
}

// checkResponseFormat rejects structured output the implementation cannot
// produce.
func (s *session) checkResponseFormat(responseFormat *model.ResponseFormat) error {
	if responseFormat == nil || responseFormat.Type != model.ResponseFormatJSON {
		return nil
	}

	harnessID := s.harnessID()

	if len(responseFormat.Schema) == 0 {
		return s.config.harness.unsupported(fmt.Sprintf("%s requires a JSON schema for structured output.", harnessID), nil)
	}

	if s.config.harness.settings.OutputSchemaMapping == nil {
		return s.config.harness.unsupported(fmt.Sprintf("%s does not support structured output through ACP.", harnessID), nil)
	}

	return nil
}

// Compact reports that ACP v1 has no manual compaction.
func (s *session) Compact(context.Context, string) error {
	return s.config.harness.unsupported("ACP v1 does not define manual session compaction.", nil)
}

// SuspendTurn freezes the active turn and returns the state to continue it
// in a later process.
func (s *session) SuspendTurn(context.Context) (harness.ContinueTurnState, error) {
	if err := s.markStopped("suspend", false); err != nil {
		return harness.ContinueTurnState{}, err
	}

	coords := s.bridgeCoordinates(s.channel.Suspend())

	return newContinueTurnState(s.harnessID(), s.lifecycleData(&coords, true))
}

// Detach disconnects from the idle bridge, which keeps running, and
// returns the state to resume the session.
func (s *session) Detach(context.Context) (harness.ResumeSessionState, error) {
	if err := s.markStopped("detach", true); err != nil {
		return harness.ResumeSessionState{}, err
	}

	coords := s.bridgeCoordinates(s.channel.Suspend())

	return newResumeSessionState(s.harnessID(), s.lifecycleData(&coords, true))
}

// Stop stops the bridge and returns the state to cold-restore the session.
func (s *session) Stop(ctx context.Context) (harness.ResumeSessionState, error) {
	data := s.lifecycleData(nil, false)

	s.terminate(ctx, "stop")

	return newResumeSessionState(s.harnessID(), data)
}

// Destroy stops the bridge without returning state.
func (s *session) Destroy(ctx context.Context) error {
	s.terminate(ctx, "destroy")

	return nil
}

// markStopped ends the local session before a suspension or detach. A
// detach also requires an idle session.
func (s *session) markStopped(operation string, requireIdle bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		return fmt.Errorf("%s acp session %s is stopped; cannot %s", s.harnessID(), s.config.sessionID, operation)
	}

	if requireIdle && s.turnInFlight {
		return fmt.Errorf("%s acp session %s has an in-flight turn; suspend it instead", s.harnessID(), s.config.sessionID)
	}

	s.stopped = true

	return nil
}

// terminate sends a stop or destroy command, waits briefly for the bridge
// to exit, then kills it and closes the channel.
func (s *session) terminate(ctx context.Context, command string) {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()

		return
	}

	s.stopped = true
	s.mu.Unlock()

	s.channel.BeginClose()

	if !s.channel.Closed() {
		_ = s.channel.Send(bridge.SimpleCommand{
			Type: command,
		})
	}

	waitCtx, cancel := context.WithTimeout(ctx, terminateGracePeriod)
	defer cancel()

	if s.process != nil {
		_, _ = s.process.Process().Wait(waitCtx)
		s.process.Kill(ctx)
		s.process.WaitOutput()
	} else {
		select {
		case <-s.channel.Done():
		case <-waitCtx.Done():
		}
	}

	_ = s.channel.Close()
	s.turns.Wait()
}

// bridgeCoordinates returns the coordinates to reattach to the bridge.
func (s *session) bridgeCoordinates(lastSeenEventID int64) bridgeCoordinates {
	return bridgeCoordinates{
		Port:            s.config.launch.bridgePort,
		Token:           s.config.launch.bridgeToken,
		LastSeenEventID: lastSeenEventID,
		SandboxID:       s.config.sandboxID,
		StateDir:        s.config.paths.bridgeStateDir,
	}
}

// lifecycleData returns the session's lifecycle state data.
func (s *session) lifecycleData(coords *bridgeCoordinates, includeTurnStartConfig bool) lifecycleData {
	s.mu.Lock()
	defer s.mu.Unlock()

	profile := s.config.configuration.authenticationProfile
	initialGuidanceApplied := s.initialGuidanceApplied

	data := lifecycleData{
		ImplementationIdentity:       s.config.implementationIdentity,
		AuthenticationProfile:        &profile,
		SandboxCredentialEnvironment: s.config.sandboxCredentialEnv,
		ACPSessionID:                 s.acpSessionID,
		Bridge:                       coords,
		Recovery:                     s.config.launch.recovery,
		Restoration:                  s.config.launch.restoration,
		InitialGuidanceApplied:       &initialGuidanceApplied,
		InstructionsFingerprint:      s.instructionsFingerprint,
		SkillsDirectory:              s.config.paths.skillsDirectory,
	}

	if s.turnStartConfig == nil {
		return data
	}

	coldSession := s.turnStartConfig.coldSession()
	data.ColdSession = &coldSession

	if includeTurnStartConfig {
		data.TurnStartConfig = s.turnStartConfig
	}

	return data
}

// setACPSessionID records the ACP session the bridge announced.
func (s *session) setACPSessionID(acpSessionID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.acpSessionID = acpSessionID
}

// finishTurn records that the in-flight turn ended.
func (s *session) finishTurn() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.turnInFlight = false
}

// harnessID returns the harness ID.
func (s *session) harnessID() string {
	return s.config.harness.settings.HarnessID
}

// instructionsFingerprintOf digests a turn's instructions; empty
// instructions digest as JSON null.
func instructionsFingerprintOf(instructions string) (string, error) {
	if instructions == "" {
		return plainFingerprint(nil)
	}

	return plainFingerprint(instructions)
}

// assertRecoveryToolCatalog requires a lossy rerun to offer the same
// application tools as the original turn.
func assertRecoveryToolCatalog(persisted []harness.ToolSpec, current []harness.ToolSpec) error {
	persistedFingerprint, err := plainFingerprint(persisted)
	if err != nil {
		return err
	}

	currentFingerprint, err := plainFingerprint(current)
	if err != nil {
		return err
	}

	if persistedFingerprint != currentFingerprint {
		return errors.New("acp lossy rerun requires the same active host tool catalog as the original turn")
	}

	return nil
}
