package bridgeadapter

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"path"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/aarock1234/ai/internal/shellquote"
	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// LaunchScript is the bootstrap-relative path of the bundled bridge script
// that adapters write during bootstrap and Launch runs.
const LaunchScript = "bridge.mjs"

// eventLogFileName is the bridge's event log in its state directory.
const eventLogFileName = "event-log.ndjson"

// LaunchOptions configure Launch.
type LaunchOptions struct {
	HarnessID string

	// DisplayName names the adapter in messages, such as "Claude Code".
	DisplayName string

	// BootstrapDir is the adapter's bootstrap directory relative to the
	// harness state directory.
	BootstrapDir string

	Settings bridge.ConnectionSettings
	Start    harness.StartOptions

	// Coordinates locate the live bridge recorded in the lifecycle state,
	// when there is one.
	Coordinates *Coordinates

	// WaitForHello makes every connection wait for the bridge-hello frame.
	WaitForHello bool

	// SpawnEnv returns the adapter's environment of a spawned bridge. It is
	// only called when a bridge is spawned. Nil adds nothing.
	SpawnEnv func(ctx context.Context) (map[string]string, error)

	// ExtraArguments returns the bridge command-line arguments that the
	// adapter adds after --workdir and --bridge-state-dir, as upstream's
	// command line does. Nil adds none.
	ExtraArguments func(launch LaunchContext) []string

	// Hooks observe the frames of the session.
	Hooks Hooks
}

// LaunchContext describes a bridge launch to an adapter that adds
// command-line arguments.
type LaunchContext struct {
	// HomeDir is the sandbox user's home directory.
	HomeDir string

	// BootstrapDir is the absolute bootstrap directory of the adapter.
	BootstrapDir string
}

// launch is one Launch call.
type launch struct {
	options  LaunchOptions
	identity adapterIdentity

	sandbox        sandbox.Session
	sandboxID      string
	homeDir        string
	bootstrapDir   string
	bridgeStateDir string

	userMessageResponses *atomic.Bool
}

// Launch connects a session to its bridge, following upstream's recovery
// ladder: it reattaches to the live bridge recorded in the lifecycle state,
// and otherwise spawns a bridge that replays the suspended turn from its
// event log or reruns the runtime's thread. A fresh session spawns a new
// bridge. Ctx bounds the startup; the returned runtime outlives it.
func Launch(ctx context.Context, options LaunchOptions) (*Runtime, error) {
	identity := adapterIdentity{
		harnessID:   options.HarnessID,
		displayName: options.DisplayName,
	}

	session := options.Start.SandboxSession
	if err := identity.validateBasicSandbox(session, options.Settings); err != nil {
		return nil, err
	}

	id := sandboxID(session)
	if options.Settings.MintBridgeToken != nil && id == "" {
		return nil, identity.unsupported(fmt.Sprintf(
			"The %s harness cannot use a bridge token minter with a sandbox session that does not expose an id.",
			identity.displayName,
		))
	}

	restricted := harness.RestrictedSession(session)

	homeDir, err := harness.ResolveSandboxHomeDir(ctx, restricted)
	if err != nil {
		return nil, err
	}

	stateDir := harness.StateDirectoryPath(homeDir)
	sessionDataDir := harness.SessionDataDirectoryPath(stateDir, options.Start.SessionID)

	current := &launch{
		options:              options,
		identity:             identity,
		sandbox:              restricted,
		sandboxID:            id,
		homeDir:              homeDir,
		bootstrapDir:         path.Join(stateDir, options.BootstrapDir),
		bridgeStateDir:       path.Join(sessionDataDir, "bridge"),
		userMessageResponses: &atomic.Bool{},
	}

	return current.run(ctx)
}

// run walks the recovery ladder. A failed reattach is reported and recorded
// in the error of a spawn that fails too.
func (l *launch) run(ctx context.Context) (*Runtime, error) {
	coordinates := l.options.Coordinates
	isContinue := l.options.Start.ContinueFrom != nil

	var attachErr error

	if coordinates != nil {
		runtime, err := l.attach(ctx, *coordinates, isContinue)
		if err == nil {
			return runtime, nil
		}

		if ctx.Err() != nil {
			return nil, context.Cause(ctx)
		}

		attachErr = fmt.Errorf("reattach %s bridge: %w", l.options.HarnessID, err)
		l.reportAttachFailure(attachErr)
	}

	strategy := l.respawnStrategy(ctx, coordinates != nil && isContinue)

	runtime, err := l.spawn(ctx, strategy)
	if err != nil && attachErr != nil {
		return nil, errors.Join(err, attachErr)
	}

	return runtime, err
}

// reportAttachFailure logs a reattach that failed before the bridge is
// respawned.
func (l *launch) reportAttachFailure(err error) {
	bridge.LogError(context.Background(), bridge.LogErrorOptions{
		HarnessID: l.options.HarnessID,
		SessionID: l.options.Start.SessionID,
		Message:   "bridge reattach failed; spawning a new bridge",
		Err:       err,
	})
}

// respawnStrategy chooses how a spawned bridge recovers. Replaying is only
// sound for a suspended turn, whose cursor the event log is replayed from; a
// between-turn resume reruns, because replaying the previous turn would
// deliver stale events into the next one.
func (l *launch) respawnStrategy(ctx context.Context, canReplay bool) Strategy {
	isResume := l.options.Start.ResumeFrom != nil || l.options.Start.ContinueFrom != nil
	if !isResume {
		return StrategyFresh
	}

	if !canReplay {
		return StrategyRerun
	}

	eventLog, err := l.sandbox.ReadTextFile(ctx, sandbox.ReadTextFileOptions{
		Path: path.Join(l.bridgeStateDir, eventLogFileName),
	})
	if err != nil {
		eventLog = ""
	}

	if bridge.ClassifyDiskLog(eventLog) == bridge.RecoverByReplay {
		return StrategyReplay
	}

	return StrategyRerun
}

// attach reconnects to a bridge that is still running. Parked sessions wait
// for the next start command; suspended turns ask for the events after the
// persisted cursor.
func (l *launch) attach(ctx context.Context, coordinates Coordinates, isContinue bool) (*Runtime, error) {
	endpoint, err := l.identity.resolveEndpoint(ctx, l.options.Start.SandboxSession, l.options.Settings, coordinates.Port)
	if err != nil {
		return nil, err
	}

	channel, err := l.openChannel(ctx, channelSetup{
		endpoint:        endpoint,
		token:           coordinates.Token,
		lastSeenEventID: coordinates.LastSeenEventID,
		resume:          isContinue,
	})
	if err != nil {
		return nil, err
	}

	return l.newRuntime(runtimeSetup{
		channel:    channel,
		strategy:   StrategyAttach,
		port:       coordinates.Port,
		token:      coordinates.Token,
		holdFrames: isContinue,
	}), nil
}

// spawn starts a new bridge process and connects to it.
func (l *launch) spawn(ctx context.Context, strategy Strategy) (*Runtime, error) {
	port, err := l.identity.resolvePort(l.options.Start.SandboxSession, l.options.Settings)
	if err != nil {
		return nil, err
	}

	token := l.mintToken()

	env, err := l.spawnEnv(ctx, strategy, port, token)
	if err != nil {
		return nil, err
	}

	if strategy == StrategyFresh {
		if err := l.createDirectories(ctx); err != nil {
			return nil, err
		}
	}

	bridge.MarkStarting(ctx, bridge.MarkStartingOptions{
		Sandbox:    l.sandbox,
		StateDir:   l.bridgeStateDir,
		BridgeType: l.options.HarnessID,
	})

	process, err := SpawnBridge(ctx, SpawnOptions{
		Sandbox:        l.sandbox,
		Command:        l.launchCommand(),
		Env:            env,
		StateDir:       l.bridgeStateDir,
		BridgeType:     l.options.HarnessID,
		Name:           l.options.HarnessID,
		StartupTimeout: startupTimeout(l.options.Settings),
	})
	if err != nil {
		return nil, err
	}

	runtime, err := l.connectSpawned(ctx, spawnedBridge{
		process:  process,
		strategy: strategy,
		token:    token,
	})
	if err != nil {
		process.Kill(ctx)
		process.WaitOutput()

		return nil, err
	}

	return runtime, nil
}

// spawnedBridge is a spawned bridge process waiting for its connection.
type spawnedBridge struct {
	process  *BridgeProcess
	strategy Strategy
	token    string
}

// connectSpawned opens the channel of a bridge that became ready.
func (l *launch) connectSpawned(ctx context.Context, spawned spawnedBridge) (*Runtime, error) {
	port := spawned.process.Port()

	endpoint, err := l.identity.resolveEndpoint(ctx, l.options.Start.SandboxSession, l.options.Settings, port)
	if err != nil {
		return nil, err
	}

	setup := channelSetup{
		endpoint: endpoint,
		token:    spawned.token,
	}

	// A replaying bridge reloaded the finished turn from disk, so the cursor
	// is seeded and the channel resumes to stream the rest of the turn.
	replaying := spawned.strategy == StrategyReplay
	if replaying {
		setup.lastSeenEventID = l.options.Coordinates.LastSeenEventID
		setup.resume = true
	}

	channel, err := l.openChannel(ctx, setup)
	if err != nil {
		return nil, err
	}

	runtime := l.newRuntime(runtimeSetup{
		channel:    channel,
		strategy:   spawned.strategy,
		port:       port,
		token:      spawned.token,
		holdFrames: replaying,
	})
	runtime.bridgeProcess = spawned.process

	return runtime, nil
}

// spawnEnv returns the environment of a spawned bridge: the adapter's
// environment and the bridge's port and token, which rotate on every spawn.
func (l *launch) spawnEnv(ctx context.Context, strategy Strategy, port int, token string) (map[string]string, error) {
	env := make(map[string]string)

	if l.options.SpawnEnv != nil {
		adapterEnv, err := l.options.SpawnEnv(ctx)
		if err != nil {
			return nil, err
		}

		maps.Copy(env, adapterEnv)
	}

	env["BRIDGE_CHANNEL_TOKEN"] = token
	env["BRIDGE_WS_PORT"] = strconv.Itoa(port)

	if strategy == StrategyReplay {
		env["BRIDGE_REPLAY_FROM_DISK"] = "1"
	}

	return env, nil
}

// createDirectories creates the working directory and the bridge state
// directory of a fresh session.
func (l *launch) createDirectories(ctx context.Context) error {
	command := "mkdir -p " + shellquote.Quote(l.options.Start.SessionWorkDir) + " " + shellquote.Quote(l.bridgeStateDir)

	return RunChecked(ctx, l.sandbox, CheckedCommand{
		Command: command,
		Failure: "create " + l.options.HarnessID + " bridge directories",
	})
}

// launchCommand returns the command that runs the bridge, as upstream.
func (l *launch) launchCommand() string {
	command := "node " +
		shellquote.Quote(path.Join(l.bootstrapDir, LaunchScript)) +
		" --workdir " + shellquote.Quote(l.options.Start.SessionWorkDir) +
		" --bridge-state-dir " + shellquote.Quote(l.bridgeStateDir)

	if l.options.ExtraArguments == nil {
		return command
	}

	arguments := l.options.ExtraArguments(LaunchContext{
		HomeDir:      l.homeDir,
		BootstrapDir: l.bootstrapDir,
	})

	for _, argument := range arguments {
		command += " " + shellquote.Quote(argument)
	}

	return command
}

// mintToken returns the token of a spawned bridge.
func (l *launch) mintToken() string {
	if l.options.Settings.MintBridgeToken != nil {
		return l.options.Settings.MintBridgeToken(l.sandboxID)
	}

	return bridge.NewToken()
}

// channelSetup configures openChannel.
type channelSetup struct {
	endpoint        harness.PortEndpoint
	token           string
	lastSeenEventID int64
	resume          bool
}

// openChannel opens a reconnecting channel to the bridge.
func (l *launch) openChannel(ctx context.Context, setup channelSetup) (*bridge.Channel, error) {
	endpoint, err := bridge.WithToken(setup.endpoint, setup.token)
	if err != nil {
		return nil, err
	}

	connect := connector{
		identity:             l.identity,
		endpoint:             endpoint,
		client:               l.options.Settings.HTTPClient,
		waitForHello:         l.options.WaitForHello,
		handshakeBudget:      startupTimeout(l.options.Settings),
		userMessageResponses: l.userMessageResponses,
	}

	channel := bridge.NewChannel(bridge.ChannelOptions{
		Connect:                connect.connect,
		Reconnect:              l.options.Settings.Reconnect,
		OnDiagnostic:           l.diagnosticReporter(),
		OnBridgeError:          l.bridgeErrorLogger(),
		InitialLastSeenEventID: setup.lastSeenEventID,
	})

	err = channel.Open(ctx, bridge.OpenOptions{
		Resume: setup.resume,
	})
	if err != nil {
		_ = channel.Close()

		return nil, err
	}

	return channel, nil
}

// diagnosticReporter forwards the bridge's diagnostics frames to the
// observability reporter, or returns nil when diagnostics are disabled.
func (l *launch) diagnosticReporter() func(message bridge.Message) {
	observability := l.options.Start.Observability
	if observability == nil || observability.Report == nil {
		return nil
	}

	sessionID := l.options.Start.SessionID

	return func(message bridge.Message) {
		diagnostic, ok := bridge.DiagnosticFromFrame(message, sessionID, time.Now())
		if ok {
			observability.Report(diagnostic)
		}
	}
}

// bridgeErrorLogger logs the errors the bridge reports.
func (l *launch) bridgeErrorLogger() func(err error) {
	harnessID := l.options.HarnessID
	sessionID := l.options.Start.SessionID

	return func(err error) {
		bridge.LogError(context.Background(), bridge.LogErrorOptions{
			HarnessID: harnessID,
			SessionID: sessionID,
			Message:   "bridge reported an error",
			Err:       err,
		})
	}
}

// runtimeSetup are the connection details of a new runtime.
type runtimeSetup struct {
	channel    *bridge.Channel
	strategy   Strategy
	port       int
	token      string
	holdFrames bool
}

// newRuntime returns the runtime of a connected bridge and starts routing
// its frames.
func (l *launch) newRuntime(setup runtimeSetup) *Runtime {
	runtime := &Runtime{
		harnessID:            l.options.HarnessID,
		sessionID:            l.options.Start.SessionID,
		channel:              setup.channel,
		hooks:                l.options.Hooks,
		strategy:             setup.strategy,
		port:                 setup.port,
		token:                setup.token,
		sandboxID:            l.sandboxID,
		sandbox:              l.sandbox,
		homeDir:              l.homeDir,
		userMessageResponses: l.userMessageResponses,
		holding:              setup.holdFrames,
		stopReplies:          make(chan bridge.Stop, 1),
		routed:               make(chan struct{}),
	}

	runtime.start()

	return runtime
}
