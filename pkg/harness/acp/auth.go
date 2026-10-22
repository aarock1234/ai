package acp

import (
	"encoding/json"
	"errors"

	"github.com/aarock1234/ai/pkg/harness"
)

// Environment variables of the provider authentication.
const (
	gatewayAPIKeyVariable    = "AI_SDK_ACP_GATEWAY_API_KEY"
	gatewayBaseURLVariable   = "AI_SDK_ACP_GATEWAY_BASE_URL"
	clientAppNameVariable    = "AI_SDK_ACP_CLIENT_APP_NAME"
	clientAppVersionVariable = "AI_SDK_ACP_CLIENT_APP_VERSION"
)

// defaultGatewayBaseURL is the AI Gateway endpoint used when
// AI_GATEWAY_BASE_URL is unset.
const defaultGatewayBaseURL = "https://ai-gateway.vercel.sh"

// Provider authentication kinds.
const (
	providerDirect    = "direct"
	providerAIGateway = "ai-gateway"
)

// Gateway credential sources.
const (
	sourceAIGatewayAPIKey = "AI_GATEWAY_API_KEY"
	sourceVercelOIDCToken = "VERCEL_OIDC_TOKEN"
)

// providerAuthenticationCompatibility is the non-secret provider routing
// a session was started with. It takes part in the implementation identity,
// so that lifecycle state cannot resume with a different routing.
type providerAuthenticationCompatibility struct {
	kind string
	mode harness.AuthenticationMode

	// The gateway fields are set for the AI Gateway kind. An empty
	// credentialSource means no credential was found.
	env              map[string]any
	credentialSource string
	baseURL          string
}

// MarshalJSON encodes the compatibility in the upstream shape.
func (c providerAuthenticationCompatibility) MarshalJSON() ([]byte, error) {
	if c.kind == providerDirect {
		return json.Marshal(struct {
			Type string                     `json:"type"`
			Mode harness.AuthenticationMode `json:"mode"`
		}{
			Type: c.kind,
			Mode: c.mode,
		})
	}

	var credentialSource *string
	if c.credentialSource != "" {
		credentialSource = &c.credentialSource
	}

	return json.Marshal(struct {
		Type             string                     `json:"type"`
		Mode             harness.AuthenticationMode `json:"mode"`
		Env              map[string]any             `json:"env"`
		CredentialSource *string                    `json:"credentialSource"`
		BaseURL          string                     `json:"baseUrl"`
	}{
		Type:             c.kind,
		Mode:             c.mode,
		Env:              c.env,
		CredentialSource: credentialSource,
		BaseURL:          c.baseURL,
	})
}

// authenticationProfile identifies the authentication a session was
// started with. Lifecycle state stores it; its digest must match on resume.
type authenticationProfile struct {
	Digest                  string                     `json:"digest"`
	ACPMethodID             string                     `json:"acpMethodId,omitempty"`
	ProviderKind            string                     `json:"providerKind"`
	ProviderMode            harness.AuthenticationMode `json:"providerMode,omitempty"`
	GatewayCredentialSource string                     `json:"gatewayCredentialSource,omitempty"`
}

// newAuthenticationProfile digests the ACP authentication and the provider
// routing.
func newAuthenticationProfile(authentication *Authentication, compatibility *providerAuthenticationCompatibility) (authenticationProfile, error) {
	digest, err := stableFingerprint(map[string]any{
		"authentication":         nullablePointer(authentication),
		"providerAuthentication": nullablePointer(compatibility),
	})
	if err != nil {
		return authenticationProfile{}, err
	}

	profile := authenticationProfile{
		Digest:       digest,
		ProviderKind: providerDirect,
	}

	if authentication != nil {
		profile.ACPMethodID = authentication.MethodID
	}

	if compatibility == nil {
		return profile, nil
	}

	profile.ProviderKind = compatibility.kind
	profile.ProviderMode = compatibility.mode

	if compatibility.kind == providerAIGateway {
		profile.GatewayCredentialSource = compatibility.credentialSource
	}

	return profile, nil
}

// bridgeProviderAuthentication is the provider routing the bridge applies.
type bridgeProviderAuthentication struct {
	Type string         `json:"type"`
	Env  map[string]any `json:"env,omitzero"`
}

// resolvedProviderAuthentication is the provider routing of a session and
// the environment that carries it into the bridge.
type resolvedProviderAuthentication struct {
	// provider is nil when no provider authentication is configured.
	provider *bridgeProviderAuthentication

	env map[string]string
}

// isGateway reports routing through the AI Gateway.
func (r resolvedProviderAuthentication) isGateway() bool {
	return r.provider != nil && r.provider.Type == providerAIGateway
}

// providerAuthenticationInput is the input of provider authentication
// resolution.
type providerAuthenticationInput struct {
	auth                   harness.Authentication
	providerAuthentication *ProviderAuthentication
	clientApp              ClientApp
	env                    map[string]string
	compatibility          *providerAuthenticationCompatibility
}

// resolveProviderAuthentication selects direct or AI Gateway routing and
// the environment the bridge needs for it.
func resolveProviderAuthentication(input providerAuthenticationInput) (resolvedProviderAuthentication, error) {
	clientAppEnv := map[string]string{
		clientAppNameVariable:    input.clientApp.Name,
		clientAppVersionVariable: input.clientApp.Version,
	}

	direct := resolvedProviderAuthentication{
		provider: &bridgeProviderAuthentication{
			Type: providerDirect,
		},
		env: clientAppEnv,
	}

	if input.providerAuthentication == nil {
		return resolvedProviderAuthentication{
			env: clientAppEnv,
		}, nil
	}

	if input.compatibility != nil && input.compatibility.kind == providerDirect {
		return direct, nil
	}

	mode := providerAuthenticationMode(input.auth)
	if mode == AuthenticationDirect {
		return direct, nil
	}

	resolvedEnv := authenticationEnvironment(input.auth, input.env)
	gateway := harness.AIGatewayAuthFromEnv(resolvedEnv)
	gatewayEnv := input.providerAuthentication.Gateway.Env

	if input.compatibility != nil && input.compatibility.kind == providerAIGateway {
		gateway = harness.AIGatewayAuth{
			APIKey:  gatewayCredential(resolvedEnv, input.compatibility.credentialSource),
			BaseURL: input.compatibility.baseURL,
		}
		gatewayEnv = input.compatibility.env
	}

	if input.compatibility == nil && mode == harness.AuthenticationAuto && gateway.APIKey == "" {
		return direct, nil
	}

	if gateway.APIKey == "" {
		return resolvedProviderAuthentication{}, errors.New("ai gateway authentication was selected, but neither AI_GATEWAY_API_KEY nor VERCEL_OIDC_TOKEN is set")
	}

	gatewayCredentials := map[string]string{
		gatewayAPIKeyVariable:  gateway.APIKey,
		gatewayBaseURLVariable: gateway.BaseURL,
	}

	return resolvedProviderAuthentication{
		provider: &bridgeProviderAuthentication{
			Type: providerAIGateway,
			Env:  gatewayEnv,
		},
		env: mergeEnvironments(gatewayCredentials, clientAppEnv),
	}, nil
}

// resolveProviderAuthenticationCompatibility returns the provider routing a
// session is started with, or nil without provider authentication.
func resolveProviderAuthenticationCompatibility(auth harness.Authentication, providerAuthentication *ProviderAuthentication, env map[string]string) *providerAuthenticationCompatibility {
	if providerAuthentication == nil {
		return nil
	}

	mode := providerAuthenticationMode(auth)
	if mode == AuthenticationDirect {
		return &providerAuthenticationCompatibility{
			kind: providerDirect,
			mode: mode,
		}
	}

	resolvedEnv := authenticationEnvironment(auth, env)

	credentialSource := gatewayCredentialSource(resolvedEnv)
	if mode == harness.AuthenticationAuto && credentialSource == "" {
		return &providerAuthenticationCompatibility{
			kind: providerDirect,
			mode: mode,
		}
	}

	baseURL, ok := resolvedEnv["AI_GATEWAY_BASE_URL"]
	if !ok {
		baseURL = defaultGatewayBaseURL
	}

	return &providerAuthenticationCompatibility{
		kind:             providerAIGateway,
		mode:             mode,
		env:              providerAuthentication.Gateway.Env,
		credentialSource: credentialSource,
		baseURL:          baseURL,
	}
}

// providerAuthenticationMode returns the configured mode, AuthenticationAuto
// for an explicit environment or no mode.
func providerAuthenticationMode(auth harness.Authentication) harness.AuthenticationMode {
	if auth.Environment != nil || auth.Mode == "" {
		return harness.AuthenticationAuto
	}

	return auth.Mode
}

// authenticationEnvironment returns the explicit authentication
// environment, or env when none is configured.
func authenticationEnvironment(auth harness.Authentication, env map[string]string) map[string]string {
	if auth.Environment != nil {
		return auth.Environment
	}

	return env
}

// gatewayCredentialSource names the variable holding the gateway
// credential, or returns empty when neither is set.
func gatewayCredentialSource(env map[string]string) string {
	switch {
	case env[sourceAIGatewayAPIKey] != "":
		return sourceAIGatewayAPIKey
	case env[sourceVercelOIDCToken] != "":
		return sourceVercelOIDCToken
	default:
		return ""
	}
}

// gatewayCredential reads the gateway credential from the variable the
// session was started with.
func gatewayCredential(env map[string]string, credentialSource string) string {
	switch credentialSource {
	case sourceAIGatewayAPIKey:
		if _, ok := env[sourceAIGatewayAPIKey]; !ok {
			return ""
		}

		return harness.AIGatewayAuthFromEnv(env).APIKey
	case sourceVercelOIDCToken:
		return env[sourceVercelOIDCToken]
	default:
		return ""
	}
}
