package harness

import (
	"context"
)

// defaultAIGatewayBaseURL is the AI Gateway endpoint used when
// AI_GATEWAY_BASE_URL is unset.
const defaultAIGatewayBaseURL = "https://ai-gateway.vercel.sh"

// AuthenticationMode selects how an adapter authenticates its runtime.
// Adapters define further modes, such as a direct provider key or a native
// CLI subscription.
type AuthenticationMode string

// Authentication modes shared by adapters. The empty mode is
// AuthenticationAuto.
const (
	// AuthenticationAuto lets the adapter discover credentials.
	AuthenticationAuto AuthenticationMode = "auto"

	// AuthenticationAIGateway routes the runtime through the AI Gateway.
	AuthenticationAIGateway AuthenticationMode = "ai-gateway"
)

// Authentication is the authentication option of an adapter: a mode, or an
// explicit environment. When Environment is set it replaces the host process
// environment for credential discovery and Mode is ignored.
type Authentication struct {
	Mode        AuthenticationMode
	Environment map[string]string
}

// CredentialForwardingFunc customizes a credential immediately before an
// adapter forwards it into a sandbox process. It only controls the value
// sandbox processes see; it does not restrict which credentials the adapter
// reads on the host.
type CredentialForwardingFunc func(ctx context.Context, event CredentialForwardingEvent) (string, error)

// CredentialForwardingEvent describes a credential about to be forwarded.
type CredentialForwardingEvent struct {
	// Credential is the value the adapter would forward: a generated
	// placeholder when credential brokering is available (see
	// IsSandboxCredentialPlaceholder), and the real credential otherwise.
	Credential string

	// EnvironmentVariableName is the variable exposing the value in the
	// sandbox.
	EnvironmentVariableName string
}

// MintBridgeTokenFunc creates the token that secures a bridge-backed
// adapter's channel for one session. Adapters default to bridge.NewToken.
type MintBridgeTokenFunc func(sandboxID string) string

// RequestTransformationSources are the inputs an adapter uses to build the
// request transformations of a session.
type RequestTransformationSources struct {
	// Env is the host environment the adapter resolved credentials from.
	Env map[string]string

	// SandboxEnv is the environment forwarded into the sandbox.
	SandboxEnv map[string]string

	Auth Authentication
}

// AIGatewayAuth is the AI Gateway configuration found in an environment.
type AIGatewayAuth struct {
	// APIKey is AI_GATEWAY_API_KEY, falling back to VERCEL_OIDC_TOKEN. It is
	// empty when neither is set.
	APIKey string

	// BaseURL is AI_GATEWAY_BASE_URL, defaulting to the public gateway.
	BaseURL string
}

// AIGatewayAuthFromEnv reads the AI Gateway configuration from env.
func AIGatewayAuthFromEnv(env map[string]string) AIGatewayAuth {
	apiKey := env["AI_GATEWAY_API_KEY"]
	if apiKey == "" {
		apiKey = env["VERCEL_OIDC_TOKEN"]
	}

	baseURL, ok := env["AI_GATEWAY_BASE_URL"]
	if !ok {
		baseURL = defaultAIGatewayBaseURL
	}

	return AIGatewayAuth{
		APIKey:  apiKey,
		BaseURL: baseURL,
	}
}

// NativeSubscriptionCheck describes an adapter's authentication state for
// ShouldResolveNativeSubscription.
type NativeSubscriptionCheck struct {
	// Mode is the configured authentication mode.
	Mode AuthenticationMode

	// DirectMode is the adapter's mode for direct runtime credentials, such
	// as "direct".
	DirectMode AuthenticationMode

	// Env is the environment credentials are discovered from.
	Env map[string]string

	// HasDirectCredential reports a configured direct credential, such as a
	// provider API key.
	HasDirectCredential bool
}

// ShouldResolveNativeSubscription reports whether an adapter should look up
// its runtime's native subscription credentials: when the mode is not the
// AI Gateway, no direct credential is configured, and either the mode is
// the direct mode or no AI Gateway key is available.
func ShouldResolveNativeSubscription(check NativeSubscriptionCheck) bool {
	if check.Mode == AuthenticationAIGateway || check.HasDirectCredential {
		return false
	}

	return check.Mode == check.DirectMode || AIGatewayAuthFromEnv(check.Env).APIKey == ""
}
