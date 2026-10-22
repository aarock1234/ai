package opencode

import (
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// subscriptionAccessTokenVariable carries the access token of a native
// login through credential brokering. The bridge never sees it: it is
// converted into OpenCode's auth content before the bridge is spawned.
const subscriptionAccessTokenVariable = "AI_SDK_OPENCODE_NATIVE_ACCESS_TOKEN"

// credentialEnvironmentVariables name the variables that hold credentials.
var credentialEnvironmentVariables = []string{
	"AI_GATEWAY_API_KEY",
	"OPENAI_API_KEY",
	"ANTHROPIC_API_KEY",
	"ANTHROPIC_AUTH_TOKEN",
	"GOOGLE_GENERATIVE_AI_API_KEY",
	"XAI_API_KEY",
	"GITHUB_TOKEN",
	"GITHUB_COPILOT_TOKEN",
	"POE_API_KEY",
	"OPENCODE_API_KEY",
	"GITLAB_TOKEN",
	subscriptionAccessTokenVariable,
}

// nonGoogleDirectCredentialVariables name the direct credentials that keep
// an unconfigured session from defaulting to Google.
var nonGoogleDirectCredentialVariables = []string{
	"OPENAI_API_KEY",
	"ANTHROPIC_API_KEY",
	"ANTHROPIC_AUTH_TOKEN",
	"XAI_API_KEY",
	"GITHUB_TOKEN",
	"GITHUB_COPILOT_TOKEN",
	"POE_API_KEY",
	"OPENCODE_API_KEY",
	"GITLAB_TOKEN",
}

// resolvedMode is the route OpenCode's requests take: directly to a
// provider, or through the AI Gateway.
type resolvedMode string

// Resolved authentication modes. Every mode except resolvedAIGateway is a
// direct provider.
const (
	resolvedAnthropic     resolvedMode = "anthropic"
	resolvedOpenAI        resolvedMode = "openai"
	resolvedGoogle        resolvedMode = "google"
	resolvedXAI           resolvedMode = "xai"
	resolvedGitHubCopilot resolvedMode = "github-copilot"
	resolvedPoe           resolvedMode = "poe"
	resolvedOpenCodeGo    resolvedMode = "opencode-go"
	resolvedGitLab        resolvedMode = "gitlab"
	resolvedAIGateway     resolvedMode = "ai-gateway"
)

// directProviders are the providers the adapter authenticates directly.
var directProviders = []resolvedMode{
	resolvedAnthropic,
	resolvedOpenAI,
	resolvedGoogle,
	resolvedXAI,
	resolvedGitHubCopilot,
	resolvedPoe,
	resolvedOpenCodeGo,
	resolvedGitLab,
}

// isDirectProvider reports whether provider names a direct provider.
func isDirectProvider(provider string) bool {
	return slices.Contains(directProviders, resolvedMode(provider))
}

// explicitMode returns the configured mode, or empty when an explicit
// environment replaces it.
func explicitMode(auth harness.Authentication) harness.AuthenticationMode {
	if auth.Environment != nil {
		return ""
	}

	return auth.Mode
}

// pinsDirectProvider reports whether the authentication pins the selected
// provider: AuthenticationAnthropic for Anthropic and AuthenticationOpenAI
// for OpenAI.
func pinsDirectProvider(auth harness.Authentication, selected resolvedMode) bool {
	mode := explicitMode(auth)

	return (selected == resolvedOpenAI && mode == AuthenticationOpenAI) ||
		(selected == resolvedAnthropic && mode == AuthenticationAnthropic)
}

// resolveEnv returns the environment OpenCode authenticates with: the
// credentials of a pinned direct provider, the AI Gateway when it is
// selected or its key is available, and the selected direct provider
// otherwise.
func resolveEnv(auth harness.Authentication, provider string, processEnv map[string]string) map[string]string {
	environment := processEnv
	if auth.Environment != nil {
		environment = auth.Environment
	}

	selected := resolveAuthenticationProvider(auth, provider, environment)
	if pinsDirectProvider(auth, selected) {
		return pickDirectProvider(selected, environment)
	}

	gateway := harness.AIGatewayAuthFromEnv(environment)
	if explicitMode(auth) == harness.AuthenticationAIGateway || gateway.APIKey != "" {
		return pickGateway(gateway)
	}

	return pickDirectProvider(selected, environment)
}

// resolveMode returns the route of the configured authentication.
func resolveMode(auth harness.Authentication, provider string, processEnv map[string]string) resolvedMode {
	if auth.Environment != nil {
		if harness.AIGatewayAuthFromEnv(auth.Environment).APIKey != "" {
			return resolvedAIGateway
		}

		return resolveAuthenticationProvider(auth, provider, auth.Environment)
	}

	selected := resolveAuthenticationProvider(auth, provider, processEnv)

	switch {
	case pinsDirectProvider(auth, selected):
		return selected
	case auth.Mode == harness.AuthenticationAIGateway:
		return resolvedAIGateway
	case harness.AIGatewayAuthFromEnv(processEnv).APIKey != "":
		return resolvedAIGateway
	default:
		return selected
	}
}

// resolveAuthenticationProvider selects the direct provider: the configured
// provider, the pinned provider of an unconfigured session, Google when an
// unconfigured session has only a Google key, and Anthropic otherwise.
func resolveAuthenticationProvider(auth harness.Authentication, provider string, environment map[string]string) resolvedMode {
	if isDirectProvider(provider) {
		return resolvedMode(provider)
	}

	if provider != "" {
		return resolvedAnthropic
	}

	switch mode := explicitMode(auth); mode {
	case AuthenticationAnthropic, AuthenticationOpenAI:
		return resolvedMode(mode)
	}

	if hasOnlyGoogleCredential(environment) {
		return resolvedGoogle
	}

	return resolvedAnthropic
}

// hasOnlyGoogleCredential reports a Google key without any other direct
// credential.
func hasOnlyGoogleCredential(environment map[string]string) bool {
	if environment["GOOGLE_GENERATIVE_AI_API_KEY"] == "" {
		return false
	}

	for _, name := range nonGoogleDirectCredentialVariables {
		if environment[name] != "" {
			return false
		}
	}

	return true
}

// pickDirectProvider returns the credentials of a direct provider.
func pickDirectProvider(provider resolvedMode, environment map[string]string) map[string]string {
	switch provider {
	case resolvedOpenAI:
		return pickNonEmpty(environment, "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORGANIZATION", "OPENAI_PROJECT")
	case resolvedAnthropic:
		return pickNonEmpty(environment, "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
	case resolvedGoogle:
		return pickPresent(environment, "GOOGLE_GENERATIVE_AI_API_KEY")
	case resolvedXAI:
		return pickPresent(environment, "XAI_API_KEY", "XAI_BASE_URL")
	case resolvedGitHubCopilot:
		return pickPresent(environment, "GITHUB_COPILOT_TOKEN", "GITHUB_TOKEN")
	case resolvedPoe:
		return pickPresent(environment, "POE_API_KEY")
	case resolvedOpenCodeGo:
		return pickPresent(environment, "OPENCODE_API_KEY")
	default:
		return pickPresent(environment, "GITLAB_TOKEN", "GITLAB_INSTANCE_URL")
	}
}

// pickNonEmpty returns the named variables that hold a value.
func pickNonEmpty(environment map[string]string, names ...string) map[string]string {
	picked := make(map[string]string)

	for _, name := range names {
		if value := environment[name]; value != "" {
			picked[name] = value
		}
	}

	return picked
}

// pickPresent returns the named variables that are defined, even when
// empty.
func pickPresent(environment map[string]string, names ...string) map[string]string {
	picked := make(map[string]string)

	for _, name := range names {
		if value, ok := environment[name]; ok {
			picked[name] = value
		}
	}

	return picked
}

// pickGateway returns the AI Gateway credentials and its OpenAI-compatible
// endpoint.
func pickGateway(gateway harness.AIGatewayAuth) map[string]string {
	env := map[string]string{
		"AI_GATEWAY_BASE_URL": gatewayBaseURL(gateway.BaseURL),
	}

	if gateway.APIKey != "" {
		env["AI_GATEWAY_API_KEY"] = gateway.APIKey
	}

	return env
}

// gatewayBaseURL returns the AI Gateway endpoint with its /v1 path.
func gatewayBaseURL(baseURL string) string {
	trimmed := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(trimmed, "/v1") {
		return trimmed
	}

	return trimmed + "/v1"
}

// hasCredential reports whether environment holds a credential of the mode.
func hasCredential(environment map[string]string, mode resolvedMode) bool {
	if mode == resolvedAIGateway {
		_, ok := environment["AI_GATEWAY_API_KEY"]

		return ok
	}

	for name := range environment {
		if strings.HasSuffix(name, "_API_KEY") || strings.HasSuffix(name, "_TOKEN") {
			return true
		}
	}

	return false
}

// transformationSources are the inputs of requestTransformations.
type transformationSources struct {
	env        map[string]string
	sandboxEnv map[string]string
	mode       resolvedMode
}

// requestTransformations returns the transformations that inject the real
// credentials of the mode in place of their sandbox placeholders.
func requestTransformations(sources transformationSources) ([]harness.RequestTransformation, error) {
	env := sources.env
	sandboxEnv := sources.sandboxEnv

	switch sources.mode {
	case resolvedAIGateway:
		return gatewayTransformations(sources)
	case resolvedOpenAI:
		if env["OPENAI_API_KEY"] == "" || sandboxEnv["OPENAI_API_KEY"] == "" {
			return nil, nil
		}

		return transformations(bearerReplacement(
			firstPresentOr(env, "OPENAI_BASE_URL", "https://api.openai.com/v1"),
			sandboxEnv["OPENAI_API_KEY"],
			env["OPENAI_API_KEY"],
		))
	case resolvedAnthropic:
		return anthropicTransformations(sources)
	case resolvedGoogle:
		if env["GOOGLE_GENERATIVE_AI_API_KEY"] == "" || sandboxEnv["GOOGLE_GENERATIVE_AI_API_KEY"] == "" {
			return nil, nil
		}

		return transformations(headerReplacement{
			matchURL:   "https://generativelanguage.googleapis.com",
			header:     "x-goog-api-key",
			matchValue: sandboxEnv["GOOGLE_GENERATIVE_AI_API_KEY"],
			value:      env["GOOGLE_GENERATIVE_AI_API_KEY"],
		})
	case resolvedXAI:
		return presentBearerTransformations(sources, "XAI_API_KEY", firstPresentOr(env, "XAI_BASE_URL", "https://api.x.ai/v1"))
	case resolvedGitHubCopilot:
		name := "GITHUB_TOKEN"
		if _, ok := env["GITHUB_COPILOT_TOKEN"]; ok {
			name = "GITHUB_COPILOT_TOKEN"
		}

		return presentBearerTransformations(sources, name, "https://api.githubcopilot.com")
	case resolvedPoe:
		return presentBearerTransformations(sources, "POE_API_KEY", "https://api.poe.com")
	case resolvedOpenCodeGo:
		return presentBearerTransformations(sources, "OPENCODE_API_KEY", openCodeGoURL)
	default:
		return presentBearerTransformations(sources, "GITLAB_TOKEN", firstPresentOr(env, "GITLAB_INSTANCE_URL", defaultGitLabInstanceURL))
	}
}

// gatewayTransformations replace the AI Gateway key, which OpenCode sends as
// x-api-key or as a bearer token, with the real key as a bearer token.
func gatewayTransformations(sources transformationSources) ([]harness.RequestTransformation, error) {
	realKey := sources.env["AI_GATEWAY_API_KEY"]
	placeholder := sources.sandboxEnv["AI_GATEWAY_API_KEY"]

	if realKey == "" || placeholder == "" {
		return nil, nil
	}

	matchURL := sources.env["AI_GATEWAY_BASE_URL"]

	return transformations(
		headerReplacement{
			matchURL:        matchURL,
			header:          "x-api-key",
			matchValue:      placeholder,
			transformHeader: "Authorization",
			value:           "Bearer " + realKey,
		},
		bearerReplacement(matchURL, placeholder, realKey),
	)
}

// anthropicTransformations replace the Anthropic API key and auth token.
func anthropicTransformations(sources transformationSources) ([]harness.RequestTransformation, error) {
	env := sources.env
	sandboxEnv := sources.sandboxEnv
	matchURL := firstPresentOr(env, "ANTHROPIC_BASE_URL", "https://api.anthropic.com")

	var replacements []headerReplacement

	apiKey, hasAPIKey := env["ANTHROPIC_API_KEY"]
	apiKeyPlaceholder, hasAPIKeyPlaceholder := sandboxEnv["ANTHROPIC_API_KEY"]

	if hasAPIKey && hasAPIKeyPlaceholder {
		replacements = append(replacements, headerReplacement{
			matchURL:   matchURL,
			header:     "x-api-key",
			matchValue: apiKeyPlaceholder,
			value:      apiKey,
		})
	}

	authToken, hasAuthToken := env["ANTHROPIC_AUTH_TOKEN"]
	authTokenPlaceholder, hasAuthTokenPlaceholder := sandboxEnv["ANTHROPIC_AUTH_TOKEN"]

	if hasAuthToken && hasAuthTokenPlaceholder {
		replacements = append(replacements, bearerReplacement(matchURL, authTokenPlaceholder, authToken))
	}

	return transformations(replacements...)
}

// presentBearerTransformations replace a bearer credential that both
// environments define.
func presentBearerTransformations(sources transformationSources, name string, matchURL string) ([]harness.RequestTransformation, error) {
	realValue, hasRealValue := sources.env[name]
	placeholder, hasPlaceholder := sources.sandboxEnv[name]

	if !hasRealValue || !hasPlaceholder {
		return nil, nil
	}

	return transformations(bearerReplacement(matchURL, placeholder, realValue))
}

// headerReplacement replaces the placeholder value of a request header,
// sent to matchURL, with the real value.
type headerReplacement struct {
	matchURL   string
	header     string
	matchValue string

	// transformHeader is the header that receives value. Empty replaces
	// header itself.
	transformHeader string
	value           string
}

// bearerReplacement replaces a placeholder bearer token.
func bearerReplacement(matchURL string, placeholder string, realValue string) headerReplacement {
	return headerReplacement{
		matchURL:   matchURL,
		header:     "Authorization",
		matchValue: "Bearer " + placeholder,
		value:      "Bearer " + realValue,
	}
}

// transformations builds the request transformations of replacements.
func transformations(replacements ...headerReplacement) ([]harness.RequestTransformation, error) {
	built := make([]harness.RequestTransformation, 0, len(replacements))

	for _, replacement := range replacements {
		transformHeader := replacement.transformHeader
		if transformHeader == "" {
			transformHeader = replacement.header
		}

		transformation, err := harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
			MatchURL: replacement.matchURL,
			MatchHeaders: map[string]string{
				replacement.header: replacement.matchValue,
			},
			TransformHeaders: map[string]string{
				transformHeader: replacement.value,
			},
		})
		if err != nil {
			return nil, err
		}

		built = append(built, transformation)
	}

	return built, nil
}

// firstPresentOr returns the variable when it is defined, and fallback
// otherwise.
func firstPresentOr(env map[string]string, name string, fallback string) string {
	if value, ok := bridgeadapter.FirstPresent(env, name); ok {
		return value
	}

	return fallback
}
