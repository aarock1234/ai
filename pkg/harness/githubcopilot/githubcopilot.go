// Package githubcopilot ports the experimental upstream
// @ai-sdk/harness-github-copilot package: a harness adapter for GitHub
// Copilot CLI, driven through the Agent Client Protocol by the acp package.
//
// The adapter installs the pinned GitHub Copilot CLI inside the sandbox, so
// the sandbox needs network access and at least one exposed TCP port. Set
// COPILOT_GITHUB_TOKEN, GH_TOKEN or GITHUB_TOKEN for direct authentication.
// Without one, the adapter reads the host's GitHub Copilot CLI login from
// the native credential store or its plaintext fallback, and then
// `gh auth token` when gh is on PATH. Set AI_GATEWAY_API_KEY or
// VERCEL_OIDC_TOKEN to route Copilot through the AI Gateway instead, or pass
// an isolated environment in Settings.Auth.Environment.
//
// Upstream's default githubCopilot instance corresponds to New(Settings{}).
package githubcopilot

import (
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/acp"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// Version is the upstream @ai-sdk/harness-github-copilot version this
// package ports. It is reported as the client app version.
const Version = "1.0.53"

// clientAppName is the client app name reported to GitHub Copilot.
const clientAppName = "ai-sdk-harness-github-copilot"

// harnessID is the adapter's harness ID.
const harnessID = "github-copilot"

// builtinMCPToolTitlePrefix starts the titles of the tools of Copilot's
// built-in GitHub MCP server.
const builtinMCPToolTitlePrefix = "github-mcp-server-"

// ReasoningEffort is the reasoning effort Copilot requests from its model.
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
	// Auth selects how Copilot authenticates: harness.AuthenticationAuto,
	// harness.AuthenticationAIGateway, acp.AuthenticationDirect, or an
	// isolated environment.
	Auth harness.Authentication

	// CredentialForwarding customizes each credential value before it is
	// forwarded into a sandbox process. It does not restrict which
	// credentials the adapter can read on the host.
	CredentialForwarding harness.CredentialForwardingFunc

	// ReasoningEffort is passed to the CLI. Empty keeps Copilot's default.
	ReasoningEffort ReasoningEffort

	// MCPServers are MCP server definitions keyed by server name, in
	// Copilot's native MCP server configuration format.
	MCPServers map[string]any

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

// New returns the GitHub Copilot harness adapter.
func New(settings Settings) (*acp.Harness, error) {
	builtinTools, err := builtinTools()
	if err != nil {
		return nil, err
	}

	return acp.New(acp.Settings{
		Version:                          "v1",
		HarnessID:                        harnessID,
		ClientApp:                        clientApp(),
		Source:                           implementationSource(),
		Executable:                       "copilot",
		Args:                             launchArguments(settings.ReasoningEffort),
		Auth:                             settings.Auth,
		ResolveAuthenticationEnvironment: resolveSubscriptionEnvironment,

		// Copilot CLI drops every MCP server a client passes in session/new
		// unless it uses the http or sse transport, so application tools are
		// served over the bridge's HTTP MCP endpoint.
		HostToolMCPTransport: acp.HostToolsHTTP,

		ForwardEnv: []string{
			hostVariable,
			ghHostVariable,
		},
		CredentialEnv:        tokenVariables,
		CredentialBrokering:  credentialTransformations,
		CredentialForwarding: newCredentialForwarding(settings.CredentialForwarding),
		ProviderAuthentication: &acp.ProviderAuthentication{
			Gateway: gatewayAuthentication(),
		},
		ModelMapping: acp.ModelMapping{
			Type: acp.ModelSessionConfigOption,
			Path: "model",
		},
		SkillsDirectory: ".copilot/skills",
		InstructionMapping: &acp.InstructionMapping{
			Type: acp.InstructionsFilesystem,
			File: ".copilot/copilot-instructions.md",
		},
		BuiltinTools:    builtinTools,
		MCPServers:      settings.MCPServers,
		IsMCPToolCall:   mcpToolCallDetector(settings.MCPServers),
		Port:            settings.Port,
		PortEndpoint:    settings.PortEndpoint,
		StartupTimeout:  settings.StartupTimeout,
		Reconnect:       settings.Reconnect,
		MintBridgeToken: settings.MintBridgeToken,
	})
}

// clientApp identifies the adapter to GitHub Copilot.
func clientApp() acp.ClientApp {
	return acp.ClientApp{
		Name:    clientAppName,
		Version: Version,
	}
}

// launchArguments starts Copilot's ACP server over stdio, with the
// configured reasoning effort.
func launchArguments(effort ReasoningEffort) []string {
	arguments := []string{
		"--acp",
		"--stdio",
		"--no-auto-update",
	}

	if effort == "" {
		return arguments
	}

	return append(arguments, "--reasoning-effort="+string(effort))
}

// gatewayAuthentication routes Copilot's OpenAI-compatible provider through
// the AI Gateway's Responses API.
func gatewayAuthentication() acp.GatewayAuthentication {
	return acp.GatewayAuthentication{
		Env: map[string]any{
			"COPILOT_PROVIDER_BASE_URL": acp.ValueSource{
				Source:       acp.SourceGatewayBaseURL,
				EnsureSuffix: "/v1",
			},
			"COPILOT_PROVIDER_TYPE": "openai",
			"COPILOT_PROVIDER_API_KEY": acp.ValueSource{
				Source: acp.SourceGatewayAPIKey,
			},
			"COPILOT_PROVIDER_WIRE_API": "responses",
			"COPILOT_MODEL":             "openai/gpt-5.5",
			"COPILOT_PROVIDER_HEADERS": acp.ValueSource{
				Source: acp.SourceClientApp,
				Prefix: "x-client-app: ",
			},
		},
	}
}

// mcpToolCallDetector reports Copilot's MCP tool calls, whose titles start
// with the name of their MCP server: the built-in GitHub MCP server or one
// of the configured servers.
func mcpToolCallDetector(mcpServers map[string]any) func(toolCall acp.ToolCall) bool {
	prefixes := []string{
		builtinMCPToolTitlePrefix,
	}

	for name := range mcpServers {
		prefixes = append(prefixes, name+"-")
	}

	return func(toolCall acp.ToolCall) bool {
		for _, prefix := range prefixes {
			if strings.HasPrefix(toolCall.Title, prefix) {
				return true
			}
		}

		return false
	}
}
