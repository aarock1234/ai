package deepagents

import (
	"context"
	"maps"
	"path"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// Start starts a Deep Agents session: it resolves the Anthropic client
// credentials, brokers them when the sandbox can inject them outside its
// boundary, and connects to the session's bridge, reattaching to a live
// bridge or respawning one when the session resumes. A respawned bridge of
// a resumed session loads the conversation checkpoint its last stop saved.
func (h *Harness) Start(ctx context.Context, options harness.StartOptions) (harness.Session, error) {
	data, err := bridgeadapter.DecodeStateData[stateData](options, "deepagents")
	if err != nil {
		return nil, err
	}

	processEnv := bridgeadapter.HostEnvironment()
	mode := resolveMode(h.settings.Auth, processEnv)
	authEnv := resolveEnvironment(h.settings.Auth, processEnv)

	credentials, err := h.sandboxCredentials(ctx, sandboxCredentialsOptions{
		start:    options,
		env:      authEnv,
		mode:     mode,
		previous: data.SandboxCredentialEnvironment,
	})
	if err != nil {
		return nil, err
	}

	launch := bridgeLaunch{
		isResume: isResumeStart(options),
	}

	runtime, err := bridgeadapter.Launch(ctx, bridgeadapter.LaunchOptions{
		HarnessID:      HarnessID,
		DisplayName:    HarnessID,
		BootstrapDir:   bootstrapDir,
		Settings:       h.settings.ConnectionSettings,
		Start:          options,
		Coordinates:    data.Bridge,
		SpawnEnv:       credentials.spawnEnv,
		ExtraArguments: launch.arguments,
	})
	if err != nil {
		return nil, err
	}

	homeDir := runtime.HomeDir()

	return &session{
		harness:               h,
		runtime:               runtime,
		sessionID:             options.SessionID,
		isResume:              runtime.Strategy() != bridgeadapter.StrategyFresh,
		credentialEnvironment: credentials.credentialEnv,
		permissionMode:        options.PermissionMode,
		builtinToolFiltering:  options.BuiltinToolFiltering,
		headers:               bridgeadapter.HeaderMap(options.Headers),
		skillsPaths: []string{
			path.Join(options.SessionWorkDir, skillsDir),
			path.Join(homeDir, skillsDir),
		},
	}, nil
}

// sandboxCredentialsOptions configure sandboxCredentials.
type sandboxCredentialsOptions struct {
	start    harness.StartOptions
	env      map[string]string
	mode     resolvedMode
	previous map[string]string
}

// sandboxCredentials are the credentials the bridge receives.
type sandboxCredentials struct {
	// credentialEnv holds the placeholders of brokered credentials, and is
	// nil when the sandbox cannot broker them.
	credentialEnv map[string]string

	// spawnEnv returns the environment of a spawned bridge.
	spawnEnv func(ctx context.Context) (map[string]string, error)
}

// sandboxCredentials brokers the credentials when the sandbox can add
// request transformations. Otherwise the credentials are forwarded, once a
// bridge is spawned.
func (h *Harness) sandboxCredentials(ctx context.Context, options sandboxCredentialsOptions) (sandboxCredentials, error) {
	brokering := bridgeadapter.BrokeringOptions{
		Sandbox:                        options.start.SandboxSession,
		Environment:                    options.env,
		CredentialEnvironmentVariables: credentialEnvironmentVariables,
		Forwarding:                     h.settings.CredentialForwarding,
		Previous:                       options.previous,
		Transformations: func(sandboxEnv map[string]string) ([]harness.RequestTransformation, error) {
			return requestTransformations(transformationSources{
				env:        options.env,
				sandboxEnv: sandboxEnv,
				mode:       options.mode,
			})
		},
	}

	brokered, ok, err := bridgeadapter.BrokerCredentials(ctx, brokering)
	if err != nil {
		return sandboxCredentials{}, err
	}

	if !ok {
		return sandboxCredentials{
			spawnEnv: func(ctx context.Context) (map[string]string, error) {
				forwarded, err := bridgeadapter.ForwardCredentials(ctx, brokering)
				if err != nil {
					return nil, err
				}

				return bridgeEnvironment(forwarded), nil
			},
		}, nil
	}

	return sandboxCredentials{
		credentialEnv: brokered.CredentialEnvironment,
		spawnEnv: func(context.Context) (map[string]string, error) {
			return bridgeEnvironment(brokered.SandboxEnvironment), nil
		},
	}, nil
}

// bridgeEnvironment returns the bridge environment: the credentials and the
// client application attribution.
func bridgeEnvironment(credentials map[string]string) map[string]string {
	env := maps.Clone(credentials)
	if env == nil {
		env = make(map[string]string)
	}

	env["AI_SDK_HARNESS_CLIENT_APP"] = clientApp

	return env
}

// bridgeLaunch describes how a spawned bridge starts.
type bridgeLaunch struct {
	// isResume makes the bridge load the session's conversation checkpoint,
	// as for every bridge spawned for a session with lifecycle state.
	isResume bool
}

// arguments returns the bridge arguments upstream's command line adds after
// the shared ones: the bootstrap directory and, for a session with
// lifecycle state, --resume true, without which the bridge deletes its saved
// conversation.
func (l bridgeLaunch) arguments(launch bridgeadapter.LaunchContext) []string {
	arguments := []string{
		"--bootstrap-dir",
		launch.BootstrapDir,
	}

	if l.isResume {
		arguments = append(arguments, "--resume", "true")
	}

	return arguments
}
