package acp

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"net/http"
	"os"
	"path"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// defaultStartupTimeout bounds the bridge startup.
const defaultStartupTimeout = 2 * time.Minute

// eventLogFileName is the bridge's event log in its state directory.
const eventLogFileName = "event-log.ndjson"

// Start starts a fresh session, reattaches to a parked bridge, or respawns
// the bridge to replay, rerun or cold-restore the session.
func (h *Harness) Start(ctx context.Context, options harness.StartOptions) (harness.Session, error) {
	if options.BuiltinToolFiltering != nil {
		return nil, h.unsupported("ACP built-in tool filtering is not available in this initial ACP v1 implementation.", nil)
	}

	startup, err := h.newStartup(ctx, options)
	if err != nil {
		return nil, err
	}

	strategy, err := startup.resume(ctx)
	if err != nil {
		return nil, err
	}

	if strategy.attached != nil {
		return strategy.attached, nil
	}

	launched, err := startup.launch(ctx, strategy.respawn)
	if err != nil {
		return nil, err
	}

	return launched, nil
}

// sessionStartup holds everything one Start call resolved.
type sessionStartup struct {
	harness *Harness
	options harness.StartOptions

	permissionMode harness.PermissionMode
	debug          *harness.DebugConfig

	// sandbox is the restricted view of the sandbox; network is the network
	// session, or nil for a basic sandbox session.
	sandbox   sandbox.Session
	network   harness.NetworkSession
	sandboxID string

	lifecycle      *lifecycleData
	stateHarnessID string
	isContinue     bool

	implementationIdentity string
	configuration          sessionConfiguration
	credentials            credentialPlan
	paths                  sessionPaths
}

// sessionPaths are the sandbox paths of a session.
type sessionPaths struct {
	bridgeDir         string
	implementationDir string
	bridgeStateDir    string

	// homeDir is the implementation's effective HOME: a private home for an
	// install command, and the sandbox's HOME otherwise.
	homeDir string

	// skillsDir is the configured relative skills directory and
	// skillsDirectory its absolute form.
	skillsDir       string
	skillsDirectory string
}

// newStartup validates the start options and resolves the credentials,
// identities and paths of the session.
func (h *Harness) newStartup(ctx context.Context, options harness.StartOptions) (*sessionStartup, error) {
	startup := &sessionStartup{
		harness:        h,
		options:        options,
		permissionMode: options.PermissionMode,
		sandbox:        harness.RestrictedSession(options.SandboxSession),
	}

	if startup.permissionMode == "" {
		startup.permissionMode = harness.PermissionAllowAll
	}

	if options.Observability != nil {
		debug := options.Observability.Debug
		startup.debug = &debug
	}

	if network, ok := options.SandboxSession.(harness.NetworkSession); ok {
		startup.network = network
		startup.sandboxID = network.ID()
	}

	if err := startup.validateSandbox(); err != nil {
		return nil, err
	}

	if err := startup.resolveLifecycle(); err != nil {
		return nil, err
	}

	if err := startup.resolveCredentials(ctx); err != nil {
		return nil, err
	}

	if err := startup.resolvePaths(ctx); err != nil {
		return nil, err
	}

	return startup, nil
}

// validateSandbox requires an explicit port and endpoint for basic sandbox
// sessions, and a sandbox ID for minted bridge tokens.
func (s *sessionStartup) validateSandbox() error {
	settings := s.harness.settings

	if s.network == nil && settings.Port == 0 {
		return s.harness.unsupported(fmt.Sprintf("The %s ACP harness requires an explicit `port` when using a basic sandbox session.", settings.HarnessID), nil)
	}

	if s.network == nil && settings.PortEndpoint == nil {
		return s.harness.unsupported(fmt.Sprintf("The %s ACP harness requires an explicit `portEndpoint` when using a basic sandbox session.", settings.HarnessID), nil)
	}

	if settings.MintBridgeToken != nil && s.sandboxID == "" {
		return s.harness.unsupported(
			fmt.Sprintf("The %s ACP harness cannot use `mintBridgeToken` with a sandbox session that does not expose an id.", settings.HarnessID),
			nil,
		)
	}

	return nil
}

// resolveLifecycle decodes the lifecycle state of a resumed or continued
// session.
func (s *sessionStartup) resolveLifecycle() error {
	continueFrom := s.options.ContinueFrom
	if continueFrom == nil && s.options.ResumeFrom != nil {
		continueFrom = s.options.ResumeFrom.ContinueFrom
	}

	switch {
	case continueFrom != nil:
		s.isContinue = true
		s.stateHarnessID = continueFrom.HarnessID

		data, err := decodeLifecycleData(continueFrom.Data)
		if err != nil {
			return err
		}

		s.lifecycle = &data
	case s.options.ResumeFrom != nil:
		s.stateHarnessID = s.options.ResumeFrom.HarnessID

		data, err := decodeLifecycleData(s.options.ResumeFrom.Data)
		if err != nil {
			return err
		}

		s.lifecycle = &data
	}

	return nil
}

// isResume reports a session started from lifecycle state.
func (s *sessionStartup) isResume() bool {
	return s.lifecycle != nil
}

// resolvePaths resolves the sandbox paths. Harness state always lives under
// the sandbox's own HOME, never the working directory.
func (s *sessionStartup) resolvePaths(ctx context.Context) error {
	sandboxHomeDir, err := harness.ResolveSandboxHomeDir(ctx, s.sandbox)
	if err != nil {
		return err
	}

	stateDirectory := harness.StateDirectoryPath(sandboxHomeDir)
	bridgeDir := path.Join(stateDirectory, bootstrapDirectory(s.harness.settings.HarnessID))
	implementationDir := path.Join(bridgeDir, implementationDirectory)

	homeDir := sandboxHomeDir
	if s.harness.implementation.isInstallCommand() {
		homeDir = path.Join(implementationDir, "home")
	}

	skillsDir, skillsDirectory, err := resolveSkillsDirectory(homeDir, s.harness.settings.SkillsDirectory)
	if err != nil {
		return err
	}

	s.paths = sessionPaths{
		bridgeDir:         bridgeDir,
		implementationDir: implementationDir,
		bridgeStateDir:    path.Join(harness.SessionDataDirectoryPath(stateDirectory, s.options.SessionID), "bridge"),
		homeDir:           homeDir,
		skillsDir:         skillsDir,
		skillsDirectory:   skillsDirectory,
	}

	return nil
}

// resumeStrategy is how a start proceeds: with a reattached session, or by
// launching a bridge, optionally respawned for a recovery.
type resumeStrategy struct {
	attached *session
	respawn  *respawnStrategy
}

// respawnStrategy describes how a replacement bridge recovers a session.
type respawnStrategy struct {
	// mode is recoveryDiskReplay, recoveryLossyRerun or recoveryColdRestore.
	mode   string
	reason string

	// afterSeq is the replay cursor of a disk replay.
	afterSeq int64

	// turnStartConfig and acpSessionID describe the rerun or restored turn.
	turnStartConfig *turnStartConfig
	acpSessionID    string
}

// resume reattaches to the bridge of lifecycle state, or decides how a
// replacement bridge recovers the session. A fresh start launches a bridge
// without recovery.
func (s *sessionStartup) resume(ctx context.Context) (resumeStrategy, error) {
	if !s.isResume() {
		return resumeStrategy{}, nil
	}

	coords := s.lifecycle.Bridge
	if coords == nil && s.isContinue {
		return resumeStrategy{}, s.harness.unsupported(
			"ACP continuation state does not contain bridge coordinates required for replay or process-loss rerun.",
			nil,
		)
	}

	if coords != nil {
		attached, err := s.attach(ctx, *coords)
		if err == nil {
			return resumeStrategy{
				attached: attached,
			}, nil
		}

		if s.isContinue {
			respawn, err := s.recoverContinuation(ctx, *coords, err)
			if err != nil {
				return resumeStrategy{}, err
			}

			return resumeStrategy{
				respawn: respawn,
			}, nil
		}
	}

	respawn, err := s.coldRestoreStrategy()
	if err != nil {
		return resumeStrategy{}, err
	}

	return resumeStrategy{
		respawn: respawn,
	}, nil
}

// recoverContinuation decides how to recover a continued turn whose bridge
// is gone: replay a complete event log from disk, or rerun the turn.
func (s *sessionStartup) recoverContinuation(ctx context.Context, coords bridgeCoordinates, attachErr error) (*respawnStrategy, error) {
	eventLog, err := s.sandbox.ReadTextFile(ctx, sandbox.ReadTextFileOptions{
		Path: path.Join(s.paths.bridgeStateDir, eventLogFileName),
	})
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("read acp bridge event log: %w", err)
	}

	if bridge.ClassifyDiskLog(eventLog) == bridge.RecoverByReplay {
		return &respawnStrategy{
			mode:     recoveryDiskReplay,
			reason:   "completed coherent event log",
			afterSeq: coords.LastSeenEventID,
		}, nil
	}

	config := s.lifecycle.TurnStartConfig
	acpSessionID := s.lifecycle.ACPSessionID

	if config == nil || acpSessionID == "" {
		return nil, s.harness.unsupported(
			"ACP process-loss recovery is unavailable because the lifecycle state does not contain the persisted turn start configuration and ACP session identifier.",
			attachErr,
		)
	}

	if err := s.validateFingerprint(config.ConfigurationFingerprint); err != nil {
		return nil, fmt.Errorf("the persisted acp turn start configuration is incompatible with the current non-secret start configuration: %w", err)
	}

	return &respawnStrategy{
		mode:            recoveryLossyRerun,
		reason:          "event log not replayable",
		turnStartConfig: config,
		acpSessionID:    acpSessionID,
	}, nil
}

// coldRestoreStrategy validates the persisted cold session and returns the
// strategy that restores it in a replacement bridge.
func (s *sessionStartup) coldRestoreStrategy() (*respawnStrategy, error) {
	coldSession := s.lifecycle.ColdSession
	acpSessionID := s.lifecycle.ACPSessionID

	if coldSession == nil || acpSessionID == "" {
		return nil, s.harness.unsupported(
			"Cold ACP session restoration requires persisted cold-session configuration and an ACP session identifier.",
			nil,
		)
	}

	if err := s.validateFingerprint(coldSession.ConfigurationFingerprint); err != nil {
		return nil, fmt.Errorf("acp cold-session state is incompatible with the current non-secret session configuration: %w", err)
	}

	if coldSession.PermissionMode != s.permissionMode {
		return nil, fmt.Errorf("acp cold-session state is incompatible with the current non-secret session configuration: permission mode %q differs from %q", coldSession.PermissionMode, s.permissionMode)
	}

	config, err := s.configuration.newTurnStartConfig(turnStartInput{
		prompt:         []textBlock{},
		tools:          coldSession.Tools,
		permissionMode: s.permissionMode,
		debug:          s.debug,
		responseFormat: coldSession.ResponseFormat,
	})
	if err != nil {
		return nil, err
	}

	return &respawnStrategy{
		mode:            recoveryColdRestore,
		turnStartConfig: &config,
		acpSessionID:    acpSessionID,
	}, nil
}

// validateFingerprint compares a persisted configuration fingerprint with
// the current session configuration.
func (s *sessionStartup) validateFingerprint(persisted string) error {
	current, err := s.configuration.fingerprint()
	if err != nil {
		return err
	}

	if current != persisted {
		return errors.New("configuration fingerprint mismatch")
	}

	return nil
}

// attach reconnects to the bridge of lifecycle state.
func (s *sessionStartup) attach(ctx context.Context, coords bridgeCoordinates) (*session, error) {
	endpoint, err := s.bridgeEndpoint(ctx, coords.Port)
	if err != nil {
		return nil, err
	}

	channel, err := s.newChannel(endpoint, coords.Token, coords.LastSeenEventID)
	if err != nil {
		return nil, err
	}

	err = channel.Open(ctx, bridge.OpenOptions{
		Resume: s.isContinue,
	})
	if err != nil {
		return nil, errors.Join(err, channel.Close())
	}

	return newSession(s.sessionConfig(sessionLaunch{
		channel:         channel,
		bridgePort:      coords.Port,
		bridgeToken:     coords.Token,
		turnInFlight:    s.isContinue,
		turnStartConfig: s.lifecycle.TurnStartConfig,
		recovery:        s.lifecycle.Recovery,
		restoration:     s.lifecycle.Restoration,
		acpSessionID:    s.lifecycle.ACPSessionID,
	})), nil
}

// bridgePort returns the sandbox port of a new bridge.
func (s *sessionStartup) bridgePort() (int, error) {
	if s.harness.settings.Port != 0 {
		return s.harness.settings.Port, nil
	}

	if s.network != nil && len(s.network.Ports()) > 0 {
		return s.network.Ports()[0], nil
	}

	harnessID := s.harness.settings.HarnessID

	return 0, s.harness.unsupported(
		fmt.Sprintf(
			"The %s ACP harness needs a TCP port exposed by the sandbox. Create the sandbox with `ports: [<port>]` or configure the harness `port`.",
			harnessID,
		),
		nil,
	)
}

// bridgeEndpoint resolves the host endpoint of a bridge port.
func (s *sessionStartup) bridgeEndpoint(ctx context.Context, port int) (harness.PortEndpoint, error) {
	if s.harness.settings.PortEndpoint != nil {
		return *s.harness.settings.PortEndpoint, nil
	}

	if s.network == nil {
		return harness.PortEndpoint{}, s.harness.unsupported(
			fmt.Sprintf("The %s ACP harness requires an explicit `portEndpoint` when using a basic sandbox session.", s.harness.settings.HarnessID),
			nil,
		)
	}

	return s.network.PortEndpoint(ctx, harness.PortEndpointOptions{
		Port:     port,
		Protocol: harness.PortWebSocket,
	})
}

// newChannel returns an unopened channel to a bridge endpoint.
func (s *sessionStartup) newChannel(endpoint harness.PortEndpoint, token string, lastSeenEventID int64) (*bridge.Channel, error) {
	authorized, err := bridge.WithToken(endpoint, token)
	if err != nil {
		return nil, err
	}

	harnessID := s.harness.settings.HarnessID
	sessionID := s.options.SessionID

	options := bridge.ChannelOptions{
		Connect: func(ctx context.Context) (bridge.Conn, error) {
			return bridge.Dial(ctx, authorized, bridge.DialOptions{})
		},
		Reconnect: s.harness.settings.Reconnect,
		OnBridgeError: func(err error) {
			bridge.LogError(context.Background(), bridge.LogErrorOptions{
				HarnessID: harnessID,
				SessionID: sessionID,
				Message:   "bridge emitted an error frame",
				Err:       err,
			})
		},
		InitialLastSeenEventID: lastSeenEventID,
		CustomFrameTypes:       customFrameTypes,
	}

	if observability := s.options.Observability; observability != nil && observability.Report != nil {
		options.OnDiagnostic = func(message bridge.Message) {
			diagnostic, ok := bridge.DiagnosticFromFrame(message, sessionID, time.Now())
			if ok {
				observability.Report(diagnostic)
			}
		}
	}

	return bridge.NewChannel(options), nil
}

// hostEnvironment returns the host process environment.
func hostEnvironment() map[string]string {
	environment := make(map[string]string)

	for _, entry := range os.Environ() {
		name, value, ok := strings.Cut(entry, "=")
		if ok && name != "" {
			environment[name] = value
		}
	}

	return environment
}

// headerValues flattens request headers, joining repeated values. It
// returns nil when there are none.
func headerValues(header http.Header) map[string]string {
	if len(header) == 0 {
		return nil
	}

	values := make(map[string]string, len(header))
	for name, value := range header {
		values[name] = strings.Join(value, ", ")
	}

	return values
}

// mergeEnvironments merges environments; later values take precedence.
func mergeEnvironments(environments ...map[string]string) map[string]string {
	merged := make(map[string]string)
	for _, environment := range environments {
		maps.Copy(merged, environment)
	}

	return merged
}

// uniqueStrings removes repeated values, keeping the first occurrence.
func uniqueStrings(values ...[]string) []string {
	var unique []string

	for _, value := range slices.Concat(values...) {
		if !slices.Contains(unique, value) {
			unique = append(unique, value)
		}
	}

	return unique
}

// unsupported returns a capability error of the harness.
func (h *Harness) unsupported(message string, cause error) error {
	return &harness.CapabilityUnsupportedError{
		Message:   message,
		HarnessID: h.settings.HarnessID,
		Cause:     cause,
	}
}
