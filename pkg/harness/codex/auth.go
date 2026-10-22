package codex

import (
	"maps"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// Endpoints of the OpenAI routes.
const (
	defaultAIGatewayBaseURL = "https://ai-gateway.vercel.sh/v1"
	defaultOpenAIBaseURL    = "https://api.openai.com/v1"
)

// credentialEnvironmentVariables name the variables that hold credentials.
var credentialEnvironmentVariables = []string{
	"AI_GATEWAY_API_KEY",
	"CODEX_API_KEY",
}

// resolvedMode is the route requests take: directly to OpenAI or through
// the AI Gateway.
type resolvedMode string

// Resolved authentication modes.
const (
	resolvedDirect    resolvedMode = "direct"
	resolvedAIGateway resolvedMode = "ai-gateway"
)

// resolveEnv returns the environment the Codex bridge needs. An explicit
// mode pins the route; otherwise the AI Gateway is preferred when its key is
// available, then CODEX_API_KEY or OPENAI_API_KEY.
func resolveEnv(auth harness.Authentication, processEnv map[string]string) map[string]string {
	suppliedEnvironment := auth.Environment != nil

	environment := processEnv
	if suppliedEnvironment {
		environment = auth.Environment
	}

	if !suppliedEnvironment && auth.Mode == AuthenticationDirect {
		return pickOpenAI(environment)
	}

	gateway := harness.AIGatewayAuthFromEnv(environment)
	usesGateway := !suppliedEnvironment && auth.Mode == harness.AuthenticationAIGateway
	if usesGateway || gateway.APIKey != "" {
		return pickGateway(gateway)
	}

	return pickOpenAI(environment)
}

// resolveMode returns the route of the configured authentication.
func resolveMode(auth harness.Authentication, processEnv map[string]string) resolvedMode {
	environment := processEnv

	switch {
	case auth.Environment != nil:
		environment = auth.Environment
	case auth.Mode == AuthenticationDirect:
		return resolvedDirect
	case auth.Mode == harness.AuthenticationAIGateway:
		return resolvedAIGateway
	}

	if harness.AIGatewayAuthFromEnv(environment).APIKey != "" {
		return resolvedAIGateway
	}

	return resolvedDirect
}

// pickOpenAI returns the direct OpenAI credentials of environment.
func pickOpenAI(environment map[string]string) map[string]string {
	env := make(map[string]string)

	apiKey, _ := bridgeadapter.FirstPresent(environment, "OPENAI_API_KEY", "CODEX_API_KEY")

	setIfNotEmpty(env, "CODEX_API_KEY", apiKey)
	setIfNotEmpty(env, "OPENAI_BASE_URL", environment["OPENAI_BASE_URL"])
	setIfNotEmpty(env, "OPENAI_ORGANIZATION", environment["OPENAI_ORGANIZATION"])
	setIfNotEmpty(env, "OPENAI_PROJECT", environment["OPENAI_PROJECT"])

	return env
}

// pickGateway returns the AI Gateway credentials and its OpenAI-compatible
// endpoint.
func pickGateway(gateway harness.AIGatewayAuth) map[string]string {
	baseURL := codexGatewayBaseURL(gateway.BaseURL)

	env := map[string]string{
		"AI_GATEWAY_BASE_URL": baseURL,
		"OPENAI_BASE_URL":     baseURL,
	}

	if gateway.APIKey != "" {
		env["AI_GATEWAY_API_KEY"] = gateway.APIKey
		env["CODEX_API_KEY"] = gateway.APIKey
	}

	return env
}

// codexGatewayBaseURL returns the AI Gateway endpoint with its /v1 path.
func codexGatewayBaseURL(baseURL string) string {
	trimmed := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(trimmed, "/v1") {
		return trimmed
	}

	return trimmed + "/v1"
}

// transformationSources are the inputs of requestTransformations.
type transformationSources struct {
	env        map[string]string
	sandboxEnv map[string]string
	mode       resolvedMode

	// requestHeaders are added to every transformation, such as the ChatGPT
	// account of a native subscription.
	requestHeaders map[string]string
}

// requestTransformations returns the transformation that injects the real
// CODEX_API_KEY in place of its sandbox placeholder.
func requestTransformations(sources transformationSources) ([]harness.RequestTransformation, error) {
	if sources.env["CODEX_API_KEY"] == "" || sources.sandboxEnv["CODEX_API_KEY"] == "" {
		return nil, nil
	}

	transformHeaders := map[string]string{
		"Authorization": "Bearer " + sources.env["CODEX_API_KEY"],
	}
	maps.Copy(transformHeaders, sources.requestHeaders)

	transformation, err := harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
		MatchURL: openAIMatchURL(sources.env, sources.mode),
		MatchHeaders: map[string]string{
			"Authorization": "Bearer " + sources.sandboxEnv["CODEX_API_KEY"],
		},
		TransformHeaders: transformHeaders,
	})
	if err != nil {
		return nil, err
	}

	return []harness.RequestTransformation{
		transformation,
	}, nil
}

// openAIMatchURL returns the endpoint the credential is sent to.
func openAIMatchURL(env map[string]string, mode resolvedMode) string {
	if baseURL, ok := env["OPENAI_BASE_URL"]; ok {
		return baseURL
	}

	if mode == resolvedAIGateway {
		return defaultAIGatewayBaseURL
	}

	return defaultOpenAIBaseURL
}

// setIfNotEmpty sets a variable to a non-empty value.
func setIfNotEmpty(env map[string]string, name string, value string) {
	if value != "" {
		env[name] = value
	}
}
