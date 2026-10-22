package fx

import (
	"context"
	"errors"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/acp"
)

// errMissingGatewayCredential reports that no AI Gateway credential is
// available for fx.
var errMissingGatewayCredential = errors.New("fx requires AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN in the host environment or an explicit authentication environment; subscription credentials are not supported by this adapter")

// gatewayAuth routes every authentication mode through the AI Gateway.
// Upstream's direct mode prefers fx, Codex and Grok subscriptions and falls
// back to the AI Gateway; without subscriptions only that fallback remains.
func gatewayAuth(auth harness.Authentication) harness.Authentication {
	if auth.Environment != nil || auth.Mode != acp.AuthenticationDirect {
		return auth
	}

	return harness.Authentication{
		Mode: harness.AuthenticationAIGateway,
	}
}

// requireGatewayCredential fails before the bridge starts when neither the
// explicit authentication environment nor the host environment holds an AI
// Gateway credential. It is the adapter's
// acp.Settings.ResolveAuthenticationEnvironment.
func requireGatewayCredential(_ context.Context, event acp.AuthenticationEnvironmentEvent) (map[string]string, error) {
	environment := event.Env
	if event.Auth.Environment != nil {
		environment = event.Auth.Environment
	}

	if harness.AIGatewayAuthFromEnv(environment).APIKey == "" {
		return nil, errMissingGatewayCredential
	}

	return environment, nil
}
