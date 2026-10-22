package opencode

import (
	"context"
	"maps"
	"net/http"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// Start starts an OpenCode session: it resolves the credentials, brokers
// them when the sandbox can inject them outside its boundary, and connects
// to the session's bridge, reattaching to a live bridge or respawning one
// when the session resumes.
func (h *Harness) Start(ctx context.Context, options harness.StartOptions) (harness.Session, error) {
	if err := validateBuiltinToolFiltering(options.BuiltinToolFiltering); err != nil {
		return nil, err
	}

	data, err := bridgeadapter.DecodeStateData[stateData](options, "opencode")
	if err != nil {
		return nil, err
	}

	processEnv := bridgeadapter.HostEnvironment()

	authentication, err := resolveAuthentication(ctx, authenticationOptions{
		auth:       h.settings.Auth,
		provider:   h.settings.Provider,
		processEnv: processEnv,
		client:     h.settings.HTTPClient,
	})
	if err != nil {
		return nil, err
	}

	credentials, err := h.sandboxCredentials(ctx, sandboxCredentialsOptions{
		start:          options,
		authentication: authentication,
		processEnv:     processEnv,
		previous:       data.SandboxCredentialEnvironment,
	})
	if err != nil {
		return nil, err
	}

	current := &session{
		harness:                  h,
		sessionID:                options.SessionID,
		credentialEnvironment:    credentials.credentialEnv,
		openCodeConfig:           credentials.openCodeConfig,
		transformModel:           credentials.transformModel,
		permissionMode:           options.PermissionMode,
		observability:            options.Observability,
		headers:                  bridgeadapter.HeaderMap(options.Headers),
		inactiveBuiltinToolNames: inactiveBuiltinToolNames(options.BuiltinToolFiltering),
		latestSessionID:          data.OpenCodeSessionID,
	}

	runtime, err := bridgeadapter.Launch(ctx, bridgeadapter.LaunchOptions{
		HarnessID:      HarnessID,
		DisplayName:    displayName,
		BootstrapDir:   bootstrapDir,
		Settings:       h.settings.ConnectionSettings,
		Start:          options,
		Coordinates:    data.Bridge,
		WaitForHello:   true,
		SpawnEnv:       credentials.spawnEnv,
		ExtraArguments: bridgeArguments,
		Hooks: bridgeadapter.Hooks{
			OnThread: current.recordSession,
		},
	})
	if err != nil {
		return nil, err
	}

	strategy := runtime.Strategy()

	current.runtime = runtime
	current.isResume = strategy != bridgeadapter.StrategyFresh
	current.rerunContinue = strategy == bridgeadapter.StrategyRerun

	// A respawned bridge resumes the persisted OpenCode session on its first
	// prompt; an attached bridge still holds the session itself.
	if strategy.IsRespawn() {
		current.pendingResumeSessionID = data.OpenCodeSessionID
	}

	return current, nil
}

// sandboxCredentialsOptions configure sandboxCredentials.
type sandboxCredentialsOptions struct {
	start          harness.StartOptions
	authentication resolvedAuthentication
	processEnv     map[string]string
	previous       map[string]string
}

// sandboxCredentials are the credentials the bridge receives and the
// configuration they require.
type sandboxCredentials struct {
	// credentialEnv holds the placeholders of brokered credentials, and is
	// nil when the sandbox cannot broker them.
	credentialEnv map[string]string

	// openCodeConfig is the OpenCode configuration of the session, which a
	// brokered GitLab login extends with its gateway providers.
	openCodeConfig map[string]any

	// transformModel maps turn models to the providers of a brokered GitLab
	// login. It is nil otherwise.
	transformModel func(modelName string) (string, error)

	// spawnEnv returns the environment of a spawned bridge.
	spawnEnv func(ctx context.Context) (map[string]string, error)
}

// sandboxCredentials brokers the credentials when the sandbox can add
// request transformations. Otherwise the credentials are forwarded, once a
// bridge is spawned.
func (h *Harness) sandboxCredentials(ctx context.Context, options sandboxCredentialsOptions) (sandboxCredentials, error) {
	authentication := options.authentication

	plan := &brokeringPlan{
		authentication: authentication,
		processEnv:     options.processEnv,
		client:         h.settings.HTTPClient,
	}

	brokering := bridgeadapter.BrokeringOptions{
		Sandbox:                        options.start.SandboxSession,
		Environment:                    authentication.environment,
		CredentialEnvironmentVariables: credentialEnvironmentVariables,
		Forwarding:                     h.settings.CredentialForwarding,
		Previous:                       options.previous,
		Transformations: func(sandboxEnv map[string]string) ([]harness.RequestTransformation, error) {
			return plan.transformations(ctx, sandboxEnv)
		},
	}

	brokered, ok, err := bridgeadapter.BrokerCredentials(ctx, brokering)
	if err != nil {
		return sandboxCredentials{}, err
	}

	if !ok {
		return sandboxCredentials{
			openCodeConfig: h.settings.OpenCodeConfig,
			spawnEnv: func(ctx context.Context) (map[string]string, error) {
				forwarded, err := bridgeadapter.ForwardCredentials(ctx, brokering)
				if err != nil {
					return nil, err
				}

				return bridgeEnvironment(forwarded, authentication.subscription, false)
			},
		}, nil
	}

	credentials := sandboxCredentials{
		credentialEnv:  brokered.CredentialEnvironment,
		openCodeConfig: h.settings.OpenCodeConfig,
		spawnEnv: func(context.Context) (map[string]string, error) {
			return bridgeEnvironment(brokered.SandboxEnvironment, authentication.subscription, plan.gitLab != nil)
		},
	}

	if plan.gitLab != nil {
		credentials.openCodeConfig = gitLabSubscriptionConfig(gitLabConfigOptions{
			openCodeConfig: h.settings.OpenCodeConfig,
			placeholder:    plan.gitLab.placeholder,
			aiGatewayURL:   plan.gitLab.access.aiGatewayURL,
			headers:        bridgeadapter.HeaderMap(options.start.Headers),
		})

		credentials.transformModel = func(modelName string) (string, error) {
			return gitLabSubscriptionModel(modelName, h.settings.Provider)
		}
	}

	return credentials, nil
}

// brokeringPlan builds the request transformations of a brokered session.
// A GitLab login is brokered through a direct access credential of the
// GitLab AI gateway, which the plan records for the session's
// configuration.
type brokeringPlan struct {
	authentication resolvedAuthentication
	processEnv     map[string]string
	client         *http.Client

	// gitLab is set when the plan brokered a GitLab login.
	gitLab *brokeredGitLab
}

// brokeredGitLab is a GitLab login brokered through the AI gateway.
type brokeredGitLab struct {
	access      gitLabDirectAccess
	placeholder string
}

// transformations returns the transformations that replace the sandbox's
// placeholders: those of a native login when one is used, and those of the
// configured credentials otherwise.
func (p *brokeringPlan) transformations(ctx context.Context, sandboxEnv map[string]string) ([]harness.RequestTransformation, error) {
	login := p.authentication.subscription
	placeholder, hasPlaceholder := sandboxEnv[subscriptionAccessTokenVariable]

	if login == nil || !hasPlaceholder {
		return requestTransformations(transformationSources{
			env:        p.authentication.environment,
			sandboxEnv: sandboxEnv,
			mode:       p.authentication.mode,
		})
	}

	if login.provider != resolvedGitLab {
		return subscriptionTransformations(login, placeholder)
	}

	access, err := requestGitLabDirectAccess(ctx, directAccessRequest{
		accessToken:  login.accessToken,
		instanceURL:  orDefault(login.enterpriseURL, defaultGitLabInstanceURL),
		aiGatewayURL: firstPresentOr(p.processEnv, "GITLAB_AI_GATEWAY_URL", defaultGitLabAIGatewayURL),
		client:       p.client,
	})
	if err != nil {
		return nil, err
	}

	p.gitLab = &brokeredGitLab{
		access:      access,
		placeholder: placeholder,
	}

	return gitLabTransformations(access, placeholder)
}

// bridgeEnvironment returns the environment of a spawned bridge: the
// sandbox's credentials, a native login converted into OpenCode's auth
// content, and the client application attribution. The login's access token
// variable itself is not forwarded. A login brokered through the GitLab AI
// gateway authenticates through the configured providers instead.
func bridgeEnvironment(sandboxEnv map[string]string, login *subscription, gitLabBrokered bool) (map[string]string, error) {
	env := maps.Clone(sandboxEnv)
	if env == nil {
		env = make(map[string]string)
	}

	accessToken, hasAccessToken := env[subscriptionAccessTokenVariable]
	delete(env, subscriptionAccessTokenVariable)

	if login != nil && hasAccessToken && !gitLabBrokered {
		content, err := subscriptionAuthContent(login, accessToken)
		if err != nil {
			return nil, err
		}

		env["OPENCODE_AUTH_CONTENT"] = content
	}

	env["AI_SDK_HARNESS_CLIENT_APP"] = clientApp

	return env, nil
}

// orDefault returns value, or fallback when value is empty.
func orDefault(value string, fallback string) string {
	if value == "" {
		return fallback
	}

	return value
}
