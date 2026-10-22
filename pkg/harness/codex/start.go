package codex

import (
	"context"
	"maps"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// credentialsStoreConfigKey is the Codex configuration key that selects
// where the Codex CLI stores its credentials.
const credentialsStoreConfigKey = "cli_auth_credentials_store"

// Start starts a Codex session: it resolves the credentials, brokers them
// when the sandbox can inject them outside its boundary, and connects to the
// session's bridge, reattaching to a live bridge or respawning one when the
// session resumes. Codex cannot ask for approvals of its built-in tools, so
// every permission mode other than harness.PermissionAllowAll is rejected.
func (h *Harness) Start(ctx context.Context, options harness.StartOptions) (harness.Session, error) {
	if options.PermissionMode != "" && options.PermissionMode != harness.PermissionAllowAll {
		return nil, &harness.CapabilityUnsupportedError{
			Message:   "Harness 'codex' does not support built-in tool approval requests; use the allow-all permission mode.",
			HarnessID: HarnessID,
		}
	}

	data, err := bridgeadapter.DecodeStateData[stateData](options, "codex")
	if err != nil {
		return nil, err
	}

	processEnv := bridgeadapter.HostEnvironment()
	mode := resolveMode(h.settings.Auth, processEnv)

	authentication, err := resolveAuthentication(ctx, authenticationOptions{
		auth:         h.settings.Auth,
		processEnv:   processEnv,
		storeSetting: h.settings.CodexConfig[credentialsStoreConfigKey],
		client:       h.settings.HTTPClient,
	})
	if err != nil {
		return nil, err
	}

	credentials, err := h.sandboxCredentials(ctx, sandboxCredentialsOptions{
		start:          options,
		authentication: authentication,
		mode:           mode,
		previous:       data.SandboxCredentialEnvironment,
	})
	if err != nil {
		return nil, err
	}

	current := &session{
		harness:               h,
		sessionID:             options.SessionID,
		credentialEnvironment: credentials.credentialEnv,
		permissionMode:        options.PermissionMode,
		builtinToolFiltering:  options.BuiltinToolFiltering,
		observability:         options.Observability,
		headers:               bridgeadapter.HeaderMap(options.Headers),
		latestThreadID:        data.ThreadID,
		fingerprint:           data.TurnConfigurationFingerprint,
	}

	runtime, err := bridgeadapter.Launch(ctx, bridgeadapter.LaunchOptions{
		HarnessID:    HarnessID,
		DisplayName:  HarnessID,
		BootstrapDir: bootstrapDir,
		Settings:     h.settings.ConnectionSettings,
		Start:        options,
		Coordinates:  data.Bridge,
		SpawnEnv:     credentials.spawnEnv,
		Hooks: bridgeadapter.Hooks{
			OnThread: current.recordThread,
		},
	})
	if err != nil {
		return nil, err
	}

	strategy := runtime.Strategy()

	current.runtime = runtime
	current.isResume = strategy != bridgeadapter.StrategyFresh
	current.rerunContinue = strategy == bridgeadapter.StrategyRerun

	// A respawned bridge resumes the persisted thread on its first prompt; an
	// attached bridge still holds the thread itself.
	if strategy.IsRespawn() {
		current.pendingResumeThreadID = data.ThreadID
	}

	return current, nil
}

// sandboxCredentialsOptions configure sandboxCredentials.
type sandboxCredentialsOptions struct {
	start          harness.StartOptions
	authentication resolvedAuthentication
	mode           resolvedMode
	previous       map[string]string
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
	authEnv := options.authentication.environment

	brokering := bridgeadapter.BrokeringOptions{
		Sandbox:                        options.start.SandboxSession,
		Environment:                    authEnv,
		CredentialEnvironmentVariables: credentialEnvironmentVariables,
		Forwarding:                     h.settings.CredentialForwarding,
		Previous:                       options.previous,
		Transformations: func(sandboxEnv map[string]string) ([]harness.RequestTransformation, error) {
			return requestTransformations(transformationSources{
				env:            authEnv,
				sandboxEnv:     sandboxEnv,
				mode:           options.mode,
				requestHeaders: options.authentication.requestHeaders,
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

				return withClientApp(forwarded), nil
			},
		}, nil
	}

	sandboxEnv := brokered.SandboxEnvironment

	// Request transformations only apply to HTTP traffic. Naming the standard
	// OpenAI endpoint makes the bridge use its custom provider, which does not
	// use WebSockets, while an unbrokered session keeps Codex's built-in
	// provider.
	_, hasBaseURL := authEnv["OPENAI_BASE_URL"]
	if brokered.TransformationCount > 0 && options.mode == resolvedDirect && !hasBaseURL {
		sandboxEnv["OPENAI_BASE_URL"] = defaultOpenAIBaseURL
	}

	return sandboxCredentials{
		credentialEnv: brokered.CredentialEnvironment,
		spawnEnv: func(context.Context) (map[string]string, error) {
			return withClientApp(sandboxEnv), nil
		},
	}, nil
}

// withClientApp returns env with the client application attribution.
func withClientApp(env map[string]string) map[string]string {
	attributed := maps.Clone(env)
	if attributed == nil {
		attributed = make(map[string]string)
	}

	attributed["AI_SDK_HARNESS_CLIENT_APP"] = clientApp

	return attributed
}
