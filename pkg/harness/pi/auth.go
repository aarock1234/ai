package pi

import (
	"maps"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
)

// Default endpoints of the providers the adapter registers.
const (
	defaultGatewayBaseURL   = "https://ai-gateway.vercel.sh"
	defaultOpenAIBaseURL    = "https://api.openai.com/v1"
	defaultAnthropicBaseURL = "https://api.anthropic.com"
)

// gatewayProvider is Pi's provider ID of the AI Gateway.
const gatewayProvider = "vercel-ai-gateway"

// namedConfig is a named Pi configuration: a provider configuration or an
// MCP server configuration.
type namedConfig struct {
	Name   string         `json:"name"`
	Config map[string]any `json:"config"`
}

// credentialHeader is how a provider sends a credential.
type credentialHeader string

// Credential headers.
const (
	headerBearer credentialHeader = "bearer"
	headerAPIKey credentialHeader = "x-api-key"
)

// credentialUse is one credential variable a registered provider sends, for
// brokering.
type credentialUse struct {
	variable string
	matchURL string
	headers  []credentialHeader

	// realHeader is the header that carries the real credential when the
	// placeholder arrives in another header. Empty keeps the header.
	realHeader credentialHeader
}

// providerSetup is the outcome of resolving the authentication: the Pi
// providers to register, the environment Pi needs and the credentials it
// sends.
type providerSetup struct {
	registrations []namedConfig

	// environment holds the credentials and endpoints of the registered
	// providers.
	environment map[string]string

	credentials []credentialUse

	// usesGateway reports AI Gateway credentials, which select the default
	// gateway model.
	usesGateway bool
}

// authenticationEnvironment returns the environment credentials are read
// from: the explicit environment, or the host process environment.
func authenticationEnvironment(auth harness.Authentication, processEnv map[string]string) map[string]string {
	if auth.Environment != nil {
		return auth.Environment
	}

	return processEnv
}

// authenticationMode returns the configured mode. An explicit environment
// uses the automatic discovery over that environment, as upstream.
func authenticationMode(auth harness.Authentication) harness.AuthenticationMode {
	if auth.Environment != nil || auth.Mode == "" {
		return harness.AuthenticationAuto
	}

	return auth.Mode
}

// resolveProviders ports upstream resolvePiEnv and registerPiProviders: it
// selects the credentials of the mode and the Pi providers that use them.
// The auto mode prefers the AI Gateway and otherwise registers every
// provider with credentials.
func resolveProviders(auth harness.Authentication, processEnv map[string]string, headers map[string]string) providerSetup {
	environment := authenticationEnvironment(auth, processEnv)
	gateway := harness.AIGatewayAuthFromEnv(environment)

	builder := providerBuilder{
		headers:     headers,
		environment: make(map[string]string),
	}

	switch authenticationMode(auth) {
	case AuthenticationOpenAI:
		builder.registerCustomProviders(pickOpenAI(environment))
	case AuthenticationAnthropic:
		builder.registerCustomProviders(pickAnthropic(environment))
	case AuthenticationCustom:
		builder.registerCustomProviders(pickProviderCredentials(environment))
	case harness.AuthenticationAIGateway:
		if gateway.APIKey != "" {
			builder.registerGateway(gateway.APIKey, gateway.BaseURL)
		}
	default:
		if gateway.APIKey != "" {
			builder.registerGateway(gateway.APIKey, gateway.BaseURL)

			break
		}

		builder.registerCustomProviders(pickProviderCredentials(environment))
	}

	return builder.setup()
}

// providerBuilder collects provider registrations.
type providerBuilder struct {
	headers       map[string]string
	registrations []namedConfig
	environment   map[string]string
	credentials   []credentialUse
	usesGateway   bool
}

// setup returns the collected registrations.
func (b *providerBuilder) setup() providerSetup {
	return providerSetup{
		registrations: b.registrations,
		environment:   b.environment,
		credentials:   b.credentials,
		usesGateway:   b.usesGateway,
	}
}

// registerGateway registers the AI Gateway with the client application
// attribution headers.
func (b *providerBuilder) registerGateway(apiKey string, baseURL string) {
	b.usesGateway = true
	b.environment["AI_GATEWAY_API_KEY"] = apiKey

	headers := configLiteralHeaders(b.headers)
	headers["User-Agent"] = configLiteral(clientApp)
	headers["x-client-app"] = configLiteral(clientApp)

	b.registrations = append(b.registrations, namedConfig{
		Name: gatewayProvider,
		Config: map[string]any{
			"apiKey":     environmentReference("AI_GATEWAY_API_KEY"),
			"baseUrl":    configLiteral(baseURL),
			"authHeader": true,
			"headers":    headers,
		},
	})

	b.credentials = append(b.credentials, credentialUse{
		variable: "AI_GATEWAY_API_KEY",
		matchURL: baseURL,
		headers: []credentialHeader{
			headerBearer,
			headerAPIKey,
		},
		realHeader: headerBearer,
	})
}

// registerCustomProviders ports upstream registerCustomProviders: the AI
// Gateway, OpenAI and Anthropic with their default endpoints, and every
// other <PREFIX>_API_KEY that has a <PREFIX>_BASE_URL.
func (b *providerBuilder) registerCustomProviders(customEnv map[string]string) {
	if apiKey := customEnv["AI_GATEWAY_API_KEY"]; apiKey != "" {
		b.registerGateway(apiKey, valueOr(customEnv["AI_GATEWAY_BASE_URL"], defaultGatewayBaseURL))
	}

	if customEnv["OPENAI_API_KEY"] != "" {
		b.registerOpenAI(customEnv)
	}

	if customEnv["ANTHROPIC_API_KEY"] != "" {
		b.registerAnthropic(customEnv)
	}

	for _, name := range slices.Sorted(maps.Keys(customEnv)) {
		prefix, isAPIKey := strings.CutSuffix(name, "_API_KEY")
		if !isAPIKey || customEnv[name] == "" || isBuiltinPrefix(prefix) {
			continue
		}

		baseURL := customEnv[prefix+"_BASE_URL"]
		if baseURL == "" {
			continue
		}

		b.registerBearerProvider(bearerProvider{
			name:        strings.ReplaceAll(strings.ToLower(prefix), "_", "-"),
			apiKeyName:  name,
			apiKey:      customEnv[name],
			baseURLName: prefix + "_BASE_URL",
			baseURL:     baseURL,
		})
	}
}

// registerOpenAI registers OpenAI.
func (b *providerBuilder) registerOpenAI(customEnv map[string]string) {
	b.registerBearerProvider(bearerProvider{
		name:        "openai",
		apiKeyName:  "OPENAI_API_KEY",
		apiKey:      customEnv["OPENAI_API_KEY"],
		baseURLName: "OPENAI_BASE_URL",
		baseURL:     valueOr(customEnv["OPENAI_BASE_URL"], defaultOpenAIBaseURL),
	})
}

// bearerProvider is a provider authenticated with a bearer API key.
type bearerProvider struct {
	name        string
	apiKeyName  string
	apiKey      string
	baseURLName string
	baseURL     string
}

// registerBearerProvider registers a provider that sends its key as a
// bearer token.
func (b *providerBuilder) registerBearerProvider(provider bearerProvider) {
	b.environment[provider.apiKeyName] = provider.apiKey
	b.environment[provider.baseURLName] = provider.baseURL

	config := map[string]any{
		"apiKey":     environmentReference(provider.apiKeyName),
		"baseUrl":    configLiteral(provider.baseURL),
		"authHeader": true,
	}

	if len(b.headers) > 0 {
		config["headers"] = configLiteralHeaders(b.headers)
	}

	b.registrations = append(b.registrations, namedConfig{
		Name:   provider.name,
		Config: config,
	})

	b.credentials = append(b.credentials, credentialUse{
		variable: provider.apiKeyName,
		matchURL: provider.baseURL,
		headers: []credentialHeader{
			headerBearer,
		},
	})
}

// registerAnthropic registers Anthropic, which sends its key as x-api-key
// and an optional auth token as a bearer token.
func (b *providerBuilder) registerAnthropic(customEnv map[string]string) {
	baseURL := valueOr(customEnv["ANTHROPIC_BASE_URL"], defaultAnthropicBaseURL)

	b.environment["ANTHROPIC_API_KEY"] = customEnv["ANTHROPIC_API_KEY"]
	b.environment["ANTHROPIC_BASE_URL"] = baseURL

	config := map[string]any{
		"apiKey":  environmentReference("ANTHROPIC_API_KEY"),
		"baseUrl": configLiteral(baseURL),
	}

	headers := configLiteralHeaders(b.headers)

	if token := customEnv["ANTHROPIC_AUTH_TOKEN"]; token != "" {
		b.environment["ANTHROPIC_AUTH_TOKEN"] = token
		headers["authorization"] = "Bearer " + environmentReference("ANTHROPIC_AUTH_TOKEN")

		b.credentials = append(b.credentials, credentialUse{
			variable: "ANTHROPIC_AUTH_TOKEN",
			matchURL: baseURL,
			headers: []credentialHeader{
				headerBearer,
			},
		})
	}

	if len(headers) > 0 {
		config["headers"] = headers
	}

	b.registrations = append(b.registrations, namedConfig{
		Name:   "anthropic",
		Config: config,
	})

	b.credentials = append(b.credentials, credentialUse{
		variable: "ANTHROPIC_API_KEY",
		matchURL: baseURL,
		headers: []credentialHeader{
			headerAPIKey,
		},
	})
}

// isBuiltinPrefix reports the credential prefixes registered separately.
func isBuiltinPrefix(prefix string) bool {
	return prefix == "AI_GATEWAY" || prefix == "OPENAI" || prefix == "ANTHROPIC"
}

// pickOpenAI returns the OpenAI credentials of environment.
func pickOpenAI(environment map[string]string) map[string]string {
	return pickNonEmpty(environment, "OPENAI_API_KEY", "OPENAI_BASE_URL")
}

// pickAnthropic returns the Anthropic credentials of environment.
func pickAnthropic(environment map[string]string) map[string]string {
	return pickNonEmpty(environment, "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")
}

// pickProviderCredentials returns every *_API_KEY, *_BASE_URL and
// ANTHROPIC_AUTH_TOKEN variable of environment, as upstream pickProviderEnv.
func pickProviderCredentials(environment map[string]string) map[string]string {
	picked := make(map[string]string)

	for name, value := range environment {
		if value == "" {
			continue
		}

		if strings.HasSuffix(name, "_API_KEY") || strings.HasSuffix(name, "_BASE_URL") || name == "ANTHROPIC_AUTH_TOKEN" {
			picked[name] = value
		}
	}

	return picked
}

// pickNonEmpty returns the named variables that are set and not empty.
func pickNonEmpty(environment map[string]string, names ...string) map[string]string {
	picked := make(map[string]string)

	for _, name := range names {
		if value := environment[name]; value != "" {
			picked[name] = value
		}
	}

	return picked
}

// credentialVariables returns the credential variables of the providers, for
// brokering.
func (s providerSetup) credentialVariables() []string {
	variables := make([]string, 0, len(s.credentials))
	for _, use := range s.credentials {
		variables = append(variables, use.variable)
	}

	return variables
}

// requestTransformations returns the transformations that inject the real
// credentials in place of their sandbox placeholders.
func (s providerSetup) requestTransformations(sandboxEnv map[string]string) ([]harness.RequestTransformation, error) {
	var transformations []harness.RequestTransformation

	for _, use := range s.credentials {
		realValue := s.environment[use.variable]
		placeholder := sandboxEnv[use.variable]

		if realValue == "" || placeholder == "" {
			continue
		}

		for _, header := range use.headers {
			transformation, err := credentialTransformation(credentialReplacement{
				use:         use,
				header:      header,
				placeholder: placeholder,
				realValue:   realValue,
			})
			if err != nil {
				return nil, err
			}

			transformations = append(transformations, transformation)
		}
	}

	return transformations, nil
}

// credentialReplacement is one header whose placeholder is replaced.
type credentialReplacement struct {
	use         credentialUse
	header      credentialHeader
	placeholder string
	realValue   string
}

// credentialTransformation builds the transformation of one replacement.
func credentialTransformation(replacement credentialReplacement) (harness.RequestTransformation, error) {
	realHeader := replacement.use.realHeader
	if realHeader == "" {
		realHeader = replacement.header
	}

	headerName, matchValue := headerValue(replacement.header, replacement.placeholder)
	realHeaderName, realValue := headerValue(realHeader, replacement.realValue)

	return harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
		MatchURL: replacement.use.matchURL,
		MatchHeaders: map[string]string{
			headerName: matchValue,
		},
		TransformHeaders: map[string]string{
			realHeaderName: realValue,
		},
	})
}

// headerValue returns the header name and value that carry a credential.
func headerValue(header credentialHeader, credential string) (string, string) {
	if header == headerAPIKey {
		return "x-api-key", credential
	}

	return "Authorization", "Bearer " + credential
}

// environmentReference returns Pi's configuration reference to an
// environment variable, so that credentials stay out of configuration files.
func environmentReference(name string) string {
	return "${" + name + "}"
}

// configLiteral escapes a literal Pi configuration value: Pi interpolates
// "$" references and runs values that start with "!" as commands.
func configLiteral(value string) string {
	escaped := strings.ReplaceAll(value, "$", "$$")
	if strings.HasPrefix(escaped, "!") {
		return "$" + escaped
	}

	return escaped
}

// configLiteralHeaders escapes literal header values.
func configLiteralHeaders(headers map[string]string) map[string]any {
	escaped := make(map[string]any, len(headers))
	for name, value := range headers {
		escaped[name] = configLiteral(value)
	}

	return escaped
}

// valueOr returns value, or fallback when it is empty.
func valueOr(value string, fallback string) string {
	if value != "" {
		return value
	}

	return fallback
}
