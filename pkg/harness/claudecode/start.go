package claudecode

import (
	"context"
	"maps"
	"net/http"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// Start starts a Claude Code session: it resolves the credentials, brokers
// them when the sandbox can inject them outside its boundary, and connects
// to the session's bridge, reattaching to a live bridge or respawning one
// when the session resumes.
func (h *Harness) Start(ctx context.Context, options harness.StartOptions) (harness.Session, error) {
	data, err := bridgeadapter.DecodeStateData[stateData](options, "claude code")
	if err != nil {
		return nil, err
	}

	processEnv := bridgeadapter.HostEnvironment()
	mode := resolveMode(h.settings.Auth, processEnv)

	authEnv, err := resolveAuthentication(ctx, h.settings.Auth, processEnv, h.settings.HTTPClient)
	if err != nil {
		return nil, err
	}

	claudeEnv := h.claudeEnvironment(authEnv, options.Headers)

	credentials, err := h.sandboxCredentials(ctx, sandboxCredentialsOptions{
		start:    options,
		env:      claudeEnv,
		mode:     mode,
		previous: data.SandboxCredentialEnvironment,
	})
	if err != nil {
		return nil, err
	}

	current := &session{
		harness:               h,
		sessionID:             options.SessionID,
		env:                   credentials.sandboxEnv,
		credentialEnvironment: credentials.credentialEnv,
		permissionMode:        options.PermissionMode,
		builtinToolFiltering:  options.BuiltinToolFiltering,
		observability:         options.Observability,
		claudeSessionID:       data.ClaudeSessionID,
	}

	runtime, err := bridgeadapter.Launch(ctx, bridgeadapter.LaunchOptions{
		HarnessID:    HarnessID,
		DisplayName:  displayName,
		BootstrapDir: bootstrapDir,
		Settings:     h.settings.ConnectionSettings,
		Start:        options,
		Coordinates:  data.Bridge,
		WaitForHello: true,
		Hooks: bridgeadapter.Hooks{
			OnFinish: current.recordClaudeSession,
		},
	})
	if err != nil {
		return nil, err
	}

	strategy := runtime.Strategy()

	current.runtime = runtime
	current.isResume = strategy != bridgeadapter.StrategyFresh
	current.continueOnNextPrompt = strategy.IsRespawn()
	current.rerunContinue = strategy == bridgeadapter.StrategyRerun

	return current, nil
}

// claudeEnvironment returns the environment of the Claude Code process: the
// credentials, the AI Gateway attribution, the configured environment and
// the request headers.
func (h *Harness) claudeEnvironment(authEnv map[string]string, headers http.Header) map[string]string {
	env := maps.Clone(authEnv)

	// The Claude Agent SDK does not expose arbitrary request headers. It
	// sends this variable as x-client-app and appends it to the User-Agent,
	// which attributes AI Gateway requests.
	if authEnv["AI_GATEWAY_BASE_URL"] != "" {
		env["CLAUDE_AGENT_SDK_CLIENT_APP"] = clientApp
	}

	maps.Copy(env, h.settings.Env)

	if len(headers) > 0 {
		env["ANTHROPIC_CUSTOM_HEADERS"] = customHeaders(headers)
	}

	return env
}

// customHeaders formats headers as Claude Code's ANTHROPIC_CUSTOM_HEADERS,
// one "Name: value" line per value.
func customHeaders(headers http.Header) string {
	var lines []string

	for _, name := range slices.Sorted(maps.Keys(headers)) {
		for _, value := range headers[name] {
			lines = append(lines, name+": "+value)
		}
	}

	return strings.Join(lines, "\n")
}

// sandboxCredentialsOptions configure sandboxCredentials.
type sandboxCredentialsOptions struct {
	start    harness.StartOptions
	env      map[string]string
	mode     resolvedMode
	previous map[string]string
}

// sandboxCredentials is the environment the bridge receives.
type sandboxCredentials struct {
	sandboxEnv map[string]string

	// credentialEnv holds the placeholders of brokered credentials, and is
	// nil when the sandbox cannot broker them.
	credentialEnv map[string]string
}

// sandboxCredentials brokers the credentials when the sandbox can add
// request transformations, and forwards them otherwise.
func (h *Harness) sandboxCredentials(ctx context.Context, options sandboxCredentialsOptions) (sandboxCredentials, error) {
	brokering := bridgeadapter.BrokeringOptions{
		Sandbox:                        options.start.SandboxSession,
		Environment:                    options.env,
		CredentialEnvironmentVariables: credentialEnvironmentVariables,
		Forwarding:                     h.settings.CredentialForwarding,
		Previous:                       options.previous,
		Transformations: func(sandboxEnv map[string]string) ([]harness.RequestTransformation, error) {
			return requestTransformations(harness.RequestTransformationSources{
				Env:        options.env,
				SandboxEnv: sandboxEnv,
				Auth:       h.settings.Auth,
			}, options.mode)
		},
	}

	brokered, ok, err := bridgeadapter.BrokerCredentials(ctx, brokering)
	if err != nil {
		return sandboxCredentials{}, err
	}

	if ok {
		return sandboxCredentials{
			sandboxEnv:    brokered.SandboxEnvironment,
			credentialEnv: brokered.CredentialEnvironment,
		}, nil
	}

	forwarded, err := bridgeadapter.ForwardCredentials(ctx, brokering)
	if err != nil {
		return sandboxCredentials{}, err
	}

	return sandboxCredentials{
		sandboxEnv: forwarded,
	}, nil
}
