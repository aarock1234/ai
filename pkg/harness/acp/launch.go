package acp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/shellquote"
	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// launch starts a bridge, optionally recovering a session, and returns the
// connected session.
func (s *sessionStartup) launch(ctx context.Context, strategy *respawnStrategy) (*session, error) {
	environment, err := s.bridgeProcessEnvironment(ctx, strategy)
	if err != nil {
		return nil, err
	}

	if err := s.writeAuthenticationFiles(ctx, environment.forwarded); err != nil {
		return nil, err
	}

	if err := s.prepareDirectories(ctx); err != nil {
		return nil, err
	}

	process, err := s.spawnBridge(ctx, environment.process)
	if err != nil {
		return nil, err
	}

	launched, err := s.connectLaunchedBridge(ctx, connectInput{
		process:  process,
		token:    environment.token,
		strategy: strategy,
	})
	if err != nil {
		process.Kill(ctx)
		process.WaitOutput()

		return nil, err
	}

	return launched, nil
}

// launchEnvironment is the environment of a new bridge process.
type launchEnvironment struct {
	// forwarded is the implementation's environment after credential
	// forwarding.
	forwarded map[string]string

	// process is the complete bridge process environment.
	process map[string]string

	token string
}

// bridgeProcessEnvironment builds the bridge process environment: the
// implementation's environment, the bridge configuration, the provider
// authentication and the channel settings.
func (s *sessionStartup) bridgeProcessEnvironment(ctx context.Context, strategy *respawnStrategy) (launchEnvironment, error) {
	settings := s.harness.settings

	forwarded, err := s.forwardedEnvironment(ctx)
	if err != nil {
		return launchEnvironment{}, err
	}

	port, err := s.bridgePort()
	if err != nil {
		return launchEnvironment{}, err
	}

	configuration := bridgeConfiguration{
		Authentication:         settings.Authentication,
		ProviderAuthentication: s.credentials.providerAuthentication.provider,
		SessionMeta:            settings.Session.Meta,
		ClientCapabilities:     settings.ClientCapabilities,
		HostToolMCPTransport:   settings.HostToolMCPTransport,
		NonTextContentTypes:    settings.NonTextContentTypes,
	}

	if settings.AskUserQuestions != nil {
		configuration.AskUserQuestionsRequestMethod = settings.AskUserQuestions.RequestMethod
	}

	if s.credentials.providerEnvInjected {
		configuration.ProviderEnvironment = map[string]string{}
	}

	configurationEnv, err := bridgeEnvironment(configuration)
	if err != nil {
		return launchEnvironment{}, err
	}

	token := bridge.NewToken()
	if settings.MintBridgeToken != nil {
		token = settings.MintBridgeToken(s.sandboxID)
	}

	channelEnv := map[string]string{
		"BRIDGE_CHANNEL_TOKEN": token,
		"BRIDGE_WS_PORT":       strconv.Itoa(port),
	}

	if strategy != nil && strategy.mode == recoveryDiskReplay {
		channelEnv["BRIDGE_REPLAY_FROM_DISK"] = "1"
	}

	return launchEnvironment{
		forwarded: forwarded,
		process:   mergeEnvironments(forwarded, configurationEnv, s.credentials.sandboxProviderEnv, channelEnv),
		token:     token,
	}, nil
}

// writeAuthenticationFiles writes the private authentication files below
// the implementation's home directory, readable only by their owner.
func (s *sessionStartup) writeAuthenticationFiles(ctx context.Context, forwarded map[string]string) error {
	produce := s.harness.settings.AuthenticationFiles
	if produce == nil {
		return nil
	}

	files, err := produce(AuthenticationFilesEvent{
		Env:                          s.credentials.implementationEnv,
		SandboxEnv:                   forwarded,
		CredentialBrokeringAvailable: s.credentials.sandboxCredentialEnv != nil,
	})
	if err != nil {
		return fmt.Errorf("acp authentication files: %w", err)
	}

	targets := make([]string, 0, len(files))

	for _, file := range files {
		if !isAuthenticationFilePath(file.Path) {
			return fmt.Errorf("acp authentication file path must be relative without traversal: %q", file.Path)
		}

		targets = append(targets, path.Join(s.paths.homeDir, file.Path))
	}

	if len(targets) == 0 {
		return nil
	}

	if err := s.createPrivateFiles(ctx, targets); err != nil {
		return err
	}

	for index, file := range files {
		err := s.sandbox.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
			Path:    targets[index],
			Content: file.Content,
		})
		if err != nil {
			return fmt.Errorf("write acp authentication file %s: %w", file.Path, err)
		}
	}

	// A sandbox may replace a file when it writes it, so the permissions are
	// restricted again.
	return bridgeadapter.RunChecked(ctx, s.sandbox, bridgeadapter.CheckedCommand{
		Command: "chmod 600 -- " + quotedPaths(targets),
		Failure: "secure acp authentication files",
	})
}

// createPrivateFiles creates empty files, and their missing directories,
// readable only by their owner before any credential is written to them.
func (s *sessionStartup) createPrivateFiles(ctx context.Context, targets []string) error {
	directories := make([]string, 0, len(targets))
	for _, target := range targets {
		directories = append(directories, path.Dir(target))
	}

	command := strings.Join([]string{
		"umask 077",
		"mkdir -p -- " + quotedPaths(directories),
		"touch -- " + quotedPaths(targets),
		"chmod 600 -- " + quotedPaths(targets),
	}, " && ")

	return bridgeadapter.RunChecked(ctx, s.sandbox, bridgeadapter.CheckedCommand{
		Command: command,
		Failure: "create acp authentication files",
	})
}

// quotedPaths joins paths as POSIX shell words.
func quotedPaths(paths []string) string {
	quoted := make([]string, 0, len(paths))
	for _, item := range paths {
		quoted = append(quoted, shellquote.Quote(item))
	}

	return strings.Join(quoted, " ")
}

// isAuthenticationFilePath reports a relative path without traversal.
func isAuthenticationFilePath(filePath string) bool {
	return filePath != "" && !path.IsAbs(filePath) && !hasParentSegment(filePath)
}

// prepareDirectories creates the working and bridge state directories and
// marks the bridge as starting.
func (s *sessionStartup) prepareDirectories(ctx context.Context) error {
	command := fmt.Sprintf(
		"mkdir -p %s %s",
		shellquote.Quote(s.options.SessionWorkDir),
		shellquote.Quote(s.paths.bridgeStateDir),
	)

	err := bridgeadapter.RunChecked(ctx, s.sandbox, bridgeadapter.CheckedCommand{
		Command: command,
		Failure: "create acp session directories",
	})
	if err != nil {
		return err
	}

	bridge.MarkStarting(ctx, bridge.MarkStartingOptions{
		Sandbox:    s.sandbox,
		StateDir:   s.paths.bridgeStateDir,
		BridgeType: s.harness.settings.HarnessID,
	})

	return nil
}

// spawnBridge starts the bridge process and waits until it listens. The
// process outlives ctx; it is killed when the startup fails.
func (s *sessionStartup) spawnBridge(ctx context.Context, environment map[string]string) (*bridgeadapter.BridgeProcess, error) {
	harnessID := s.harness.settings.HarnessID

	command := strings.Join([]string{
		"node " + shellquote.Quote(path.Join(s.paths.bridgeDir, bridgeFileName)),
		"--workdir " + shellquote.Quote(s.options.SessionWorkDir),
		"--bridge-state-dir " + shellquote.Quote(s.paths.bridgeStateDir),
		"--implementation-dir " + shellquote.Quote(s.paths.implementationDir),
		"--bridge-type " + shellquote.Quote(harnessID),
	}, " ")

	return bridgeadapter.SpawnBridge(ctx, bridgeadapter.SpawnOptions{
		Sandbox:        s.sandbox,
		Command:        command,
		Env:            environment,
		StateDir:       s.paths.bridgeStateDir,
		BridgeType:     harnessID,
		Name:           harnessID + " ACP",
		StartupTimeout: s.startupTimeout(),
	})
}

// startupTimeout returns the bridge startup timeout.
func (s *sessionStartup) startupTimeout() time.Duration {
	if s.harness.settings.StartupTimeout > 0 {
		return s.harness.settings.StartupTimeout
	}

	return defaultStartupTimeout
}

// connectInput describes a launched bridge to connect to.
type connectInput struct {
	process  *bridgeadapter.BridgeProcess
	token    string
	strategy *respawnStrategy
}

// connectLaunchedBridge opens the channel to a launched bridge, restores a
// cold session, and returns the session.
func (s *sessionStartup) connectLaunchedBridge(ctx context.Context, input connectInput) (*session, error) {
	endpoint, err := s.bridgeEndpoint(ctx, input.process.Port())
	if err != nil {
		return nil, err
	}

	strategy := input.strategy
	isDiskReplay := strategy != nil && strategy.mode == recoveryDiskReplay

	var lastSeenEventID int64
	if isDiskReplay {
		lastSeenEventID = strategy.afterSeq
	}

	channel, err := s.newChannel(endpoint, input.token, lastSeenEventID)
	if err != nil {
		return nil, err
	}

	err = channel.Open(ctx, bridge.OpenOptions{
		Resume: isDiskReplay,
	})
	if err != nil {
		return nil, errors.Join(err, channel.Close())
	}

	launch := sessionLaunch{
		channel:     channel,
		process:     input.process,
		bridgePort:  input.process.Port(),
		bridgeToken: input.token,
	}

	if s.lifecycle != nil {
		launch.turnStartConfig = s.lifecycle.TurnStartConfig
		launch.restoration = s.lifecycle.Restoration
		launch.acpSessionID = s.lifecycle.ACPSessionID
	}

	if strategy != nil {
		if err := s.applyRespawnStrategy(ctx, &launch, *strategy); err != nil {
			bridgeadapter.AbandonChannel(channel)

			return nil, err
		}
	}

	return newSession(s.sessionConfig(launch)), nil
}

// applyRespawnStrategy records the recovery of a respawned bridge and
// restores a cold session.
func (s *sessionStartup) applyRespawnStrategy(ctx context.Context, launch *sessionLaunch, strategy respawnStrategy) error {
	switch strategy.mode {
	case recoveryDiskReplay:
		launch.turnInFlight = true
		launch.replayOnly = true
		launch.recovery = &recoveryStatus{
			Mode:   strategy.mode,
			Reason: strategy.reason,
		}
	case recoveryLossyRerun:
		launch.turnInFlight = true
		launch.lossyRerun = true
		launch.turnStartConfig = strategy.turnStartConfig
		launch.recovery = &recoveryStatus{
			Mode:   strategy.mode,
			Reason: strategy.reason,
		}
	case recoveryColdRestore:
		launch.turnStartConfig = strategy.turnStartConfig

		restored, err := s.restoreColdSession(ctx, launch.channel, strategy)
		if err != nil {
			return err
		}

		launch.restoration = &restoration{
			Method: restored.method,
		}

		if restored.acpSessionID != "" {
			launch.acpSessionID = restored.acpSessionID
		}
	}

	return nil
}

// coldRestoreResult is the outcome of a cold session restoration.
type coldRestoreResult struct {
	method       string
	acpSessionID string
}

// restoreColdSession asks a replacement bridge to restore the ACP session
// without prompting, and waits for the restoration to finish.
func (s *sessionStartup) restoreColdSession(ctx context.Context, channel *bridge.Channel, strategy respawnStrategy) (coldRestoreResult, error) {
	settings := s.harness.settings
	config := *strategy.turnStartConfig

	command := startCommand{
		Type:                  "start",
		Prompt:                []textBlock{},
		Debug:                 s.debug,
		BuiltinTools:          s.configuration.builtinTools,
		PermissionMode:        s.permissionMode,
		PermissionModeMapping: settings.PermissionModeMapping,
		InstructionMapping:    settings.InstructionMapping,
		MCPServers:            settings.MCPServers,
		Tools:                 config.Tools,
		TurnStartConfig:       config,
		RecoveryMode: &recoveryMode{
			Type:         recoveryColdRestore,
			ACPSessionID: strategy.acpSessionID,
		},
	}

	if err := channel.Send(command); err != nil {
		return coldRestoreResult{}, err
	}

	var result coldRestoreResult

	for {
		frame, err := channel.Next(ctx)
		if err != nil {
			if closed, ok := errors.AsType[*bridge.ClosedError](err); ok {
				return coldRestoreResult{}, fmt.Errorf("%s acp bridge closed during cold restoration: %s", settings.HarnessID, closed.Reason)
			}

			return coldRestoreResult{}, err
		}

		if thread, ok := frame.Message.(bridge.Thread); ok {
			result.acpSessionID = thread.ThreadID

			continue
		}

		stream, ok := frame.Message.(bridge.StreamMessage)
		if !ok {
			continue
		}

		switch part := stream.Part.(type) {
		case harness.RawPart:
			if method, ok := restoredMethod(part.Value); ok {
				result.method = method
			}
		case harness.ErrorPart:
			return coldRestoreResult{}, s.harness.bridgeError(part.Err)
		case harness.FinishPart:
			if result.method == "" {
				return coldRestoreResult{}, fmt.Errorf("%s acp cold restoration completed without identifying the negotiated method", settings.HarnessID)
			}

			return result, nil
		}
	}
}

// restoredMethod reads the restoration method of an acp-session-restored
// marker.
func restoredMethod(value json.RawMessage) (string, bool) {
	var marker struct {
		Type   string `json:"type"`
		Method string `json:"method"`
	}

	if err := json.Unmarshal(value, &marker); err != nil || marker.Type != "acp-session-restored" {
		return "", false
	}

	if marker.Method != restoreByResume && marker.Method != restoreByLoad {
		return "", false
	}

	return marker.Method, true
}
