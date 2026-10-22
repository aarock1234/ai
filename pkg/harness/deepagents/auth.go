package deepagents

import (
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
)

// AuthenticationAnthropic authenticates Deep Agents directly with
// Anthropic, never routing it through the AI Gateway.
const AuthenticationAnthropic harness.AuthenticationMode = "anthropic"

// defaultAnthropicBaseURL is Anthropic's API without a base URL override.
const defaultAnthropicBaseURL = "https://api.anthropic.com"

// Environment variables of the Anthropic client.
const (
	anthropicAPIKeyVariable    = "ANTHROPIC_API_KEY"
	anthropicAuthTokenVariable = "ANTHROPIC_AUTH_TOKEN"
	anthropicBaseURLVariable   = "ANTHROPIC_BASE_URL"
	gatewayAPIKeyVariable      = "AI_GATEWAY_API_KEY"
)

// credentialEnvironmentVariables hold the credentials the adapter brokers
// or forwards.
var credentialEnvironmentVariables = []string{
	gatewayAPIKeyVariable,
	anthropicAPIKeyVariable,
	anthropicAuthTokenVariable,
}

// resolvedMode is the provider routing an authentication resolves to.
type resolvedMode string

// Resolved modes.
const (
	resolvedAnthropic resolvedMode = "anthropic"
	resolvedGateway   resolvedMode = "ai-gateway"
)

// resolveMode resolves the provider routing: an explicit mode, or the AI
// Gateway when a gateway key is available.
func resolveMode(auth harness.Authentication, processEnv map[string]string) resolvedMode {
	if auth.Environment != nil {
		return modeOfEnvironment(auth.Environment)
	}

	switch auth.Mode {
	case AuthenticationAnthropic:
		return resolvedAnthropic
	case harness.AuthenticationAIGateway:
		return resolvedGateway
	default:
		return modeOfEnvironment(processEnv)
	}
}

// modeOfEnvironment routes through the AI Gateway when env holds a gateway
// key.
func modeOfEnvironment(env map[string]string) resolvedMode {
	if harness.AIGatewayAuthFromEnv(env).APIKey != "" {
		return resolvedGateway
	}

	return resolvedAnthropic
}

// resolveEnvironment returns the Anthropic client environment of the
// authentication. Non-Anthropic models reach the client through the AI
// Gateway's Anthropic-compatible endpoint.
func resolveEnvironment(auth harness.Authentication, processEnv map[string]string) map[string]string {
	authEnv := processEnv
	if auth.Environment != nil {
		authEnv = auth.Environment
	}

	if auth.Environment == nil && auth.Mode == AuthenticationAnthropic {
		return anthropicEnvironment(authEnv)
	}

	gateway := harness.AIGatewayAuthFromEnv(authEnv)

	isGateway := auth.Environment == nil && auth.Mode == harness.AuthenticationAIGateway
	if isGateway || gateway.APIKey != "" {
		return gatewayEnvironment(gateway)
	}

	return anthropicEnvironment(authEnv)
}

// anthropicEnvironment picks the non-empty Anthropic client variables.
func anthropicEnvironment(env map[string]string) map[string]string {
	picked := make(map[string]string)

	for _, name := range []string{
		anthropicAPIKeyVariable,
		anthropicAuthTokenVariable,
		anthropicBaseURLVariable,
	} {
		if value := env[name]; value != "" {
			picked[name] = value
		}
	}

	return picked
}

// gatewayEnvironment points the Anthropic client at the AI Gateway. The
// Anthropic client appends /v1/messages, so the base URL stays at the
// gateway's root.
func gatewayEnvironment(gateway harness.AIGatewayAuth) map[string]string {
	env := map[string]string{
		anthropicBaseURLVariable: strings.TrimRight(gateway.BaseURL, "/"),
	}

	if gateway.APIKey != "" {
		env[gatewayAPIKeyVariable] = gateway.APIKey
		env[anthropicAPIKeyVariable] = gateway.APIKey
	}

	return env
}

// transformationSources are the inputs of requestTransformations.
type transformationSources struct {
	env        map[string]string
	sandboxEnv map[string]string
	mode       resolvedMode
}

// credentialReplacement is a credential header the transformations
// replace.
type credentialReplacement struct {
	variable string
	header   string
	scheme   string
}

// requestTransformations replace the placeholder credentials of requests
// to the Anthropic endpoint with the real credentials.
func requestTransformations(sources transformationSources) ([]harness.RequestTransformation, error) {
	matchURL, ok := sources.env[anthropicBaseURLVariable]
	if !ok && sources.mode != resolvedGateway {
		matchURL = defaultAnthropicBaseURL
	}

	var transformations []harness.RequestTransformation

	replacements := []credentialReplacement{
		{
			variable: anthropicAPIKeyVariable,
			header:   "x-api-key",
		},
		{
			variable: anthropicAuthTokenVariable,
			header:   "Authorization",
			scheme:   "Bearer ",
		},
	}

	for _, replacement := range replacements {
		credential, hasCredential := sources.env[replacement.variable]
		placeholder, hasPlaceholder := sources.sandboxEnv[replacement.variable]

		if !hasCredential || !hasPlaceholder {
			continue
		}

		transformation, err := harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
			MatchURL: matchURL,
			MatchHeaders: map[string]string{
				replacement.header: replacement.scheme + placeholder,
			},
			TransformHeaders: map[string]string{
				replacement.header: replacement.scheme + credential,
			},
		})
		if err != nil {
			return nil, err
		}

		transformations = append(transformations, transformation)
	}

	return transformations, nil
}
