// Package grokbuild ports the experimental upstream
// @ai-sdk/harness-grok-build package: a harness adapter for xAI's Grok
// Build CLI, driven through the Agent Client Protocol by the acp package.
//
// The adapter installs the pinned Grok Build CLI inside the sandbox, so the
// sandbox needs network access and at least one exposed TCP port. Set
// XAI_API_KEY for direct authentication. Without one, the adapter reads the
// host's Grok CLI login from auth.json, refreshing it when it expires soon.
// Set AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN to route Grok Build through
// the AI Gateway instead, or pass an isolated environment in
// Settings.Auth.Environment. Grok Build's native question requests surface
// as the common askUserQuestions tool.
//
// Upstream's default grokBuild instance corresponds to New(Settings{}).
package grokbuild

import (
	"maps"
	"net/http"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/acp"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// Version is the upstream @ai-sdk/harness-grok-build version this package
// ports. It is reported as the client app version.
const Version = "1.0.95"

// clientAppName is the client app name reported to Grok Build.
const clientAppName = "ai-sdk-harness-grok-build"

// harnessID is the adapter's harness ID.
const harnessID = "grok-build"

// Environment variables of Grok Build's xAI connection.
const (
	apiKeyVariable           = "XAI_API_KEY"
	apiBaseURLVariable       = "GROK_XAI_API_BASE_URL"
	modelsBaseURLVariable    = "GROK_MODELS_BASE_URL"
	chatProxyBaseURLVariable = "GROK_CLI_CHAT_PROXY_BASE_URL"
)

// defaultAPIBaseURL is xAI's API without a base URL override.
const defaultAPIBaseURL = "https://api.x.ai/v1"

// ReasoningEffort is the reasoning effort of reasoning-capable models.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningNone    ReasoningEffort = "none"
	ReasoningMinimal ReasoningEffort = "minimal"
	ReasoningLow     ReasoningEffort = "low"
	ReasoningMedium  ReasoningEffort = "medium"
	ReasoningHigh    ReasoningEffort = "high"
	ReasoningXHigh   ReasoningEffort = "xhigh"
	ReasoningMax     ReasoningEffort = "max"
)

// Settings configure New. The zero value is a valid configuration.
type Settings struct {
	// Auth selects direct xAI or AI Gateway authentication:
	// harness.AuthenticationAuto, harness.AuthenticationAIGateway,
	// acp.AuthenticationDirect, or an isolated environment.
	Auth harness.Authentication

	// CredentialForwarding customizes each credential value before it is
	// forwarded into a sandbox process. It does not restrict which
	// credentials the adapter can read on the host.
	CredentialForwarding harness.CredentialForwardingFunc

	// ReasoningEffort is passed to the CLI. Empty keeps Grok Build's
	// default.
	ReasoningEffort ReasoningEffort

	// Port overrides the sandbox port of the bridge.
	Port int

	// PortEndpoint overrides the host endpoint of the bridge. It is required,
	// together with Port, with basic sandbox sessions.
	PortEndpoint *harness.PortEndpoint

	// StartupTimeout bounds the bridge startup. Zero selects two minutes.
	StartupTimeout time.Duration

	// Reconnect bounds the reconnects after an established bridge connection
	// drops.
	Reconnect bridge.ReconnectOptions

	// MCPServers are MCP server definitions keyed by server name, in Grok
	// Build's native MCP server configuration format.
	MCPServers map[string]any

	// MintBridgeToken creates the bridge token. Nil uses a random 32-byte
	// hexadecimal token.
	MintBridgeToken harness.MintBridgeTokenFunc

	// HTTPClient sends the OAuth discovery and token refresh requests of the
	// host's Grok CLI login. Nil uses http.DefaultClient.
	HTTPClient *http.Client
}

// New returns the Grok Build harness adapter. The common askUserQuestions
// tool is added to the built-in tools.
func New(settings Settings) (*acp.Harness, error) {
	builtinTools, err := builtinTools()
	if err != nil {
		return nil, err
	}

	return acp.New(acp.Settings{
		Version:   "v1",
		HarnessID: harnessID,
		NonTextContentTypes: []acp.NonTextContentType{
			acp.NonTextImage,
		},
		Auth: settings.Auth,
		ResolveAuthenticationEnvironment: subscriptionResolver{
			client: settings.HTTPClient,
		}.resolveEnvironment,
		Authentication: &acp.Authentication{
			MethodID: "xai.api_key",
		},
		CredentialForwarding: settings.CredentialForwarding,
		Port:                 settings.Port,
		PortEndpoint:         settings.PortEndpoint,
		StartupTimeout:       settings.StartupTimeout,
		Reconnect:            settings.Reconnect,
		MCPServers:           settings.MCPServers,
		ModelMapping: acp.ModelMapping{
			Type: acp.ModelSessionModel,
			Path: "modelId",
		},
		IsMCPToolCall:    isMCPToolCall,
		MintBridgeToken:  settings.MintBridgeToken,
		BuiltinTools:     builtinTools,
		AskUserQuestions: askUserQuestions(),
		ClientApp: acp.ClientApp{
			Name:    clientAppName,
			Version: Version,
		},
		Source:     implementationSource(),
		Executable: "grok",
		Args:       launchArguments(settings.ReasoningEffort),
		ForwardEnv: []string{
			apiBaseURLVariable,
			modelsBaseURLVariable,
			chatProxyBaseURLVariable,
		},
		CredentialEnv: []string{
			apiKeyVariable,
		},
		CredentialBrokering: credentialTransformations,
		InstructionMapping: &acp.InstructionMapping{
			Type: acp.InstructionsFilesystem,
			File: ".grok/AGENTS.md",
		},
		OutputSchemaMapping: &acp.OutputSchemaMapping{
			Type: acp.OutputSchemaPromptMeta,
			Path: []string{
				"outputSchema",
			},
		},
		ProviderAuthentication: &acp.ProviderAuthentication{
			Gateway: gatewayAuthentication(),
		},
	})
}

// launchArguments starts Grok Build's ACP agent over stdio, with the
// configured reasoning effort.
func launchArguments(effort ReasoningEffort) []string {
	arguments := []string{
		"agent",
	}

	if effort != "" {
		arguments = append(arguments, "--reasoning-effort", string(effort))
	}

	return append(arguments, "stdio")
}

// gatewayAuthentication routes Grok Build's xAI API and model catalog
// through the AI Gateway.
func gatewayAuthentication() acp.GatewayAuthentication {
	return acp.GatewayAuthentication{
		Env: map[string]any{
			"GROK_CLIENT_NAME": acp.ValueSource{
				Source: acp.SourceClientAppName,
			},
			"GROK_CLIENT_VERSION": acp.ValueSource{
				Source: acp.SourceClientAppVersion,
			},
			apiKeyVariable: acp.ValueSource{
				Source: acp.SourceGatewayAPIKey,
			},
			apiBaseURLVariable: acp.ValueSource{
				Source:       acp.SourceGatewayBaseURL,
				EnsureSuffix: "/v1",
			},
			modelsBaseURLVariable: acp.ValueSource{
				Source:       acp.SourceGatewayBaseURL,
				EnsureSuffix: "/v1",
			},
		},
	}
}

// credentialTransformations inject the real xAI API key, and the agent's
// request headers, into requests to the xAI API.
func credentialTransformations(event acp.CredentialBrokeringEvent) ([]harness.RequestTransformation, error) {
	apiKey := event.Env[apiKeyVariable]
	placeholder := event.SandboxEnv[apiKeyVariable]

	if apiKey == "" || placeholder == "" {
		return nil, nil
	}

	baseURL, ok := event.Env[apiBaseURLVariable]
	if !ok {
		baseURL = defaultAPIBaseURL
	}

	headers := maps.Clone(event.Headers)
	if headers == nil {
		headers = make(map[string]string, 2)
	}

	headers["Authorization"] = "Bearer " + apiKey

	if _, viaChatProxy := event.Env[chatProxyBaseURLVariable]; viaChatProxy {
		headers["X-XAI-Token-Auth"] = "xai-grok-cli"
	}

	transformation, err := harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
		MatchURL: baseURL,
		MatchHeaders: map[string]string{
			"Authorization": "Bearer " + placeholder,
		},
		TransformHeaders: headers,
	})
	if err != nil {
		return nil, err
	}

	return []harness.RequestTransformation{
		transformation,
	}, nil
}

// isMCPToolCall reports Grok Build's MCP tool calls, whose xAI tool
// metadata names the mcp namespace.
func isMCPToolCall(toolCall acp.ToolCall) bool {
	metadata, ok := toolCall.Meta["x.ai/tool"].(map[string]any)

	return ok && metadata["namespace"] == "mcp"
}
