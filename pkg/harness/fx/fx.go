// Package fx ports the experimental upstream @ai-sdk/harness-fx package: a
// harness adapter for fx, the native coding agent of fx.sh, driven through
// the Agent Client Protocol by the acp package.
//
// Upstream runs libfx, fx's agent engine, in the host Node.js process and
// executes its own sandbox tools remotely. Go cannot host libfx, so the
// adapter installs the pinned fx release inside the sandbox and drives its
// ACP server (`fx acp`), which runs fx's native tools there. The sandbox
// needs network access and at least one exposed TCP port.
//
// fx authenticates with the Vercel AI Gateway: set AI_GATEWAY_API_KEY or
// VERCEL_OIDC_TOKEN, or pass them in Settings.Auth.Environment. Upstream's
// subscription fallback (fx, Codex and Grok logins) is not ported, because
// fx's ACP server reads such logins only from its own credential files and
// would rotate their refresh tokens inside the sandbox.
//
// Upstream's default fx instance corresponds to New(Settings{}).
package fx

import (
	"maps"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/acp"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// Version is the upstream @ai-sdk/harness-fx version this package ports.
// It is reported as the client app version.
const Version = "1.0.72"

// clientAppName is the client app name reported to fx and the AI Gateway.
const clientAppName = "ai-sdk-harness-fx"

// harnessID is the adapter's harness ID.
const harnessID = "fx"

// gatewayURL is the only AI Gateway endpoint fx sends model requests to;
// fx ignores base URL overrides that are not loopback addresses.
const gatewayURL = "https://ai-gateway.vercel.sh"

// apiKeyVariable holds the AI Gateway credential fx reads.
const apiKeyVariable = "AI_GATEWAY_API_KEY"

// ServiceTier is the inference tier of fx's model requests.
type ServiceTier string

// Service tiers.
const (
	ServiceTierDefault   ServiceTier = "default"
	ServiceTierFast      ServiceTier = "fast"
	ServiceTierUltrafast ServiceTier = "ultrafast"
)

// Settings configure New. The zero value is a valid configuration.
type Settings struct {
	// Auth supplies the AI Gateway credential: harness.AuthenticationAuto,
	// harness.AuthenticationAIGateway, acp.AuthenticationDirect, or an
	// isolated environment. Every mode routes fx through the AI Gateway,
	// which upstream's direct mode falls back to without a subscription.
	Auth harness.Authentication

	// CredentialForwarding customizes each credential value before it is
	// forwarded into a sandbox process. It does not restrict which
	// credentials the adapter can read on the host.
	CredentialForwarding harness.CredentialForwardingFunc

	// MCPServers are MCP server definitions keyed by server name, in the
	// ACP session/new mcpServers format fx accepts: stdio servers with an
	// absolute command, and http or sse servers with a URL and headers.
	MCPServers map[string]any

	// Headers are added to fx's AI Gateway requests. They require a
	// sandbox that supports request transformations.
	Headers map[string]string

	// Effort is the reasoning effort of the selected model, such as "high".
	// Empty keeps fx's automatic effort.
	Effort string

	// ServiceTier requests fx's fast or ultrafast inference. Empty and
	// ServiceTierDefault keep fx's default.
	ServiceTier ServiceTier

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

	// MintBridgeToken creates the bridge token. Nil uses a random 32-byte
	// hexadecimal token.
	MintBridgeToken harness.MintBridgeTokenFunc
}

// New returns the fx harness adapter.
func New(settings Settings) (*acp.Harness, error) {
	builtinTools, err := builtinTools()
	if err != nil {
		return nil, err
	}

	if err := validateServiceTier(settings.ServiceTier); err != nil {
		return nil, err
	}

	return acp.New(acp.Settings{
		Version:                          "v1",
		HarnessID:                        harnessID,
		BuiltinTools:                     builtinTools,
		MCPServers:                       settings.MCPServers,
		IsMCPToolCall:                    isMCPToolCall,
		Port:                             settings.Port,
		PortEndpoint:                     settings.PortEndpoint,
		StartupTimeout:                   settings.StartupTimeout,
		Reconnect:                        settings.Reconnect,
		MintBridgeToken:                  settings.MintBridgeToken,
		Auth:                             gatewayAuth(settings.Auth),
		ResolveAuthenticationEnvironment: requireGatewayCredential,
		CredentialForwarding:             settings.CredentialForwarding,
		ClientApp: acp.ClientApp{
			Name:    clientAppName,
			Version: Version,
		},
		Source:     implementationSource(),
		Executable: "fx",
		Args:       launchArguments(settings.ServiceTier),
		Env:        runtimeEnvironment(),

		// The gateway credential reaches fx through ProviderAuthentication,
		// so no further variable holds a credential.
		CredentialEnv: []string{},
		CredentialBrokering: func(event acp.CredentialBrokeringEvent) ([]harness.RequestTransformation, error) {
			return credentialTransformations(settings.Headers, event)
		},
		AuthenticationFiles: func(acp.AuthenticationFilesEvent) ([]acp.AuthenticationFile, error) {
			return profileFiles(settings)
		},
		ProviderAuthentication: &acp.ProviderAuthentication{
			Gateway: acp.GatewayAuthentication{
				Env: map[string]any{
					apiKeyVariable: acp.ValueSource{
						Source: acp.SourceGatewayAPIKey,
					},
				},
			},
		},
		ModelMapping: acp.ModelMapping{
			Type: acp.ModelSessionConfigOption,
			Path: "model",
		},
		SkillsDirectory: ".fx/skills",
		InstructionMapping: &acp.InstructionMapping{
			Type: acp.InstructionsFilesystem,
			File: ".fx/AGENTS.md",
		},
		NonTextContentTypes: []acp.NonTextContentType{
			acp.NonTextImage,
		},
	})
}

// launchArguments starts fx's ACP server over stdio, requesting or
// declining the ultrafast tier as the service tier selects.
func launchArguments(tier ServiceTier) []string {
	arguments := []string{
		"acp",
	}

	switch tier {
	case ServiceTierUltrafast:
		return append(arguments, "--ultrafast")
	case ServiceTierFast:
		return append(arguments, "--no-ultrafast")
	default:
		return arguments
	}
}

// runtimeEnvironment configures fx for unattended use: the AI Gateway
// provider, no background upgrades or browser sign-ins, and the ask
// permission mode, so that every sensitive action reaches the bridge as an
// ACP permission request that it decides by the session's permission mode.
func runtimeEnvironment() map[string]string {
	return map[string]string{
		"FX_PROVIDER":        "gateway",
		"FX_PERMISSION_MODE": "ask",
		"FX_AUTO_UPGRADE":    "0",
		"FX_NO_OPEN_BROWSER": "1",
		"FX_SKIP_ONBOARDING": "1",
	}
}

// credentialTransformations inject the real AI Gateway credential, the
// configured and per-request headers, and the client app into fx's AI
// Gateway requests.
func credentialTransformations(headers map[string]string, event acp.CredentialBrokeringEvent) ([]harness.RequestTransformation, error) {
	apiKey := event.Env[apiKeyVariable]
	placeholder := event.SandboxEnv[apiKeyVariable]

	if apiKey == "" || placeholder == "" {
		return nil, nil
	}

	clientApp := clientAppName + "/" + Version

	transformHeaders := make(map[string]string, len(headers)+len(event.Headers)+3)
	maps.Copy(transformHeaders, headers)
	maps.Copy(transformHeaders, event.Headers)
	transformHeaders["Authorization"] = "Bearer " + apiKey
	transformHeaders["User-Agent"] = clientApp
	transformHeaders["x-client-app"] = clientApp

	transformation, err := harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
		MatchURL: gatewayURL,
		MatchHeaders: map[string]string{
			"Authorization": "Bearer " + placeholder,
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

// isMCPToolCall reports fx's MCP tool calls, which name their server and
// tool in the fx tool call metadata.
func isMCPToolCall(toolCall acp.ToolCall) bool {
	fxMeta, ok := toolCall.Meta["fx"].(map[string]any)
	if !ok {
		return false
	}

	toolCallMeta, ok := fxMeta["toolCall"].(map[string]any)
	if !ok {
		return false
	}

	_, isMCP := toolCallMeta["mcp"].(map[string]any)

	return isMCP
}
