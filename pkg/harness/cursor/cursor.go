// Package cursor ports the experimental upstream @ai-sdk/harness-cursor
// package: a harness adapter for Cursor CLI, driven through the Agent
// Client Protocol by the acp package.
//
// The adapter installs Cursor CLI inside the sandbox with Cursor's official
// installer, so the sandbox needs network access and at least one exposed
// TCP port. Set CURSOR_API_KEY, or pass it in Settings.Auth.Environment;
// without one, the adapter reads the host's Cursor CLI login. Cursor
// controls model-provider authentication in its own settings.
//
// Upstream's default cursor instance corresponds to New(Settings{}).
package cursor

import (
	"context"
	"encoding/json"
	"log/slog"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/acp"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/harness/internal/jsonvalue"
)

// Version is the upstream @ai-sdk/harness-cursor version this package
// ports. It is reported as the client app version.
const Version = "1.0.71"

// clientAppName is the client app name reported to Cursor.
const clientAppName = "ai-sdk-harness-cursor"

// harnessID is the adapter's harness ID.
const harnessID = "cursor"

// apiKeyVariable holds the Cursor user API key.
const apiKeyVariable = "CURSOR_API_KEY"

// Settings configure New. The zero value is a valid configuration.
type Settings struct {
	// Auth declares the provider authentication configured in Cursor, or
	// supplies an isolated environment for Cursor CLI authentication. The
	// adapter cannot change provider routing and warns for the
	// acp.AuthenticationDirect and harness.AuthenticationAIGateway modes.
	Auth harness.Authentication

	// CredentialForwarding customizes each credential value before it is
	// forwarded into a sandbox process. It does not restrict which
	// credentials the adapter can read on the host.
	CredentialForwarding harness.CredentialForwardingFunc

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

	// MCPServers are MCP server definitions keyed by server name, in
	// Cursor's native MCP server configuration format.
	MCPServers map[string]any

	// MintBridgeToken creates the bridge token. Nil uses a random 32-byte
	// hexadecimal token.
	MintBridgeToken harness.MintBridgeTokenFunc
}

// New returns the Cursor harness adapter.
func New(settings Settings) (*acp.Harness, error) {
	builtinTools, err := builtinTools()
	if err != nil {
		return nil, err
	}

	if settings.Auth.Environment == nil {
		warnAuthenticationMode(settings.Auth.Mode)
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
		Auth:                             settings.Auth,
		ResolveAuthenticationEnvironment: resolveSubscriptionEnvironment,
		CredentialForwarding:             settings.CredentialForwarding,
		MintBridgeToken:                  settings.MintBridgeToken,
		NonTextContentTypes: []acp.NonTextContentType{
			acp.NonTextImage,
		},
		ClientApp: acp.ClientApp{
			Name:    clientAppName,
			Version: Version,
		},
		Source: acp.InstallCommandSource{
			Command: "curl https://cursor.com/install -fsS | bash",
		},
		Executable: "agent",
		Args: []string{
			"--disable-auto-update",
			"acp",
		},
		ModelMapping: acp.ModelMapping{
			Type: acp.ModelSessionConfigOption,
			Path: "model",
		},
		ClientCapabilities: map[string]any{
			"_meta": map[string]any{
				"parameterizedModelPicker": true,
			},
		},
		CredentialEnv: []string{
			apiKeyVariable,
		},
		CredentialBrokering: func(event acp.CredentialBrokeringEvent) ([]harness.RequestTransformation, error) {
			return credentialTransformations(settings.Auth.Mode, event), nil
		},
	})
}

// credentialTransformations inject the real Cursor API key into the key
// exchange, and the agent's request headers into model requests.
func credentialTransformations(mode harness.AuthenticationMode, event acp.CredentialBrokeringEvent) []harness.RequestTransformation {
	var transformations []harness.RequestTransformation

	apiKey := event.Env[apiKeyVariable]
	placeholder := event.SandboxEnv[apiKeyVariable]

	if apiKey != "" && placeholder != "" {
		transformations = append(transformations, harness.RequestTransformation{
			Match: harness.RequestMatch{
				Host: "api2.cursor.sh",
				Path: &harness.Matcher{
					Exact: "/auth/exchange_user_api_key",
				},
				Method: []string{
					"POST",
				},
				Headers: []harness.KeyValueMatcher{
					{
						Key: &harness.Matcher{
							Exact: "Authorization",
						},
						Value: &harness.Matcher{
							Exact: "Bearer " + placeholder,
						},
					},
				},
			},
			Transform: harness.RequestTransform{
				Headers: map[string]string{
					"Authorization": "Bearer " + apiKey,
				},
			},
		})
	}

	if event.Headers != nil {
		transformations = append(transformations, harness.RequestTransformation{
			Match: modelRequestMatch(mode),
			Transform: harness.RequestTransform{
				Headers: event.Headers,
			},
		})
	}

	return transformations
}

// modelRequestMatch selects Cursor's model requests: those to the AI
// Gateway's Cursor endpoint in AI Gateway mode, and those to Cursor's API
// otherwise.
func modelRequestMatch(mode harness.AuthenticationMode) harness.RequestMatch {
	if mode == harness.AuthenticationAIGateway {
		return harness.RequestMatch{
			Host: "ai-gateway.vercel.sh",
			Path: &harness.Matcher{
				StartsWith: "/cursor/v1",
			},
		}
	}

	return harness.RequestMatch{
		Host: "api2.cursor.sh",
	}
}

// isMCPToolCall reports Cursor's MCP tool calls, whose raw input names the
// MCP provider and tool.
func isMCPToolCall(toolCall acp.ToolCall) bool {
	var rawInput struct {
		ProviderIdentifier *string         `json:"providerIdentifier"`
		ToolName           *string         `json:"toolName"`
		Args               json.RawMessage `json:"args"`
	}

	if err := json.Unmarshal(toolCall.RawInput, &rawInput); err != nil {
		return false
	}

	return rawInput.ProviderIdentifier != nil && rawInput.ToolName != nil && jsonvalue.IsObject(rawInput.Args)
}

// warnAuthenticationMode reminds that explicit provider routing modes must
// be configured in Cursor itself.
func warnAuthenticationMode(mode harness.AuthenticationMode) {
	var detail string

	switch mode {
	case harness.AuthenticationAIGateway:
		detail = "configure an AI Gateway API key as Cursor's OpenAI API key and set Override OpenAI Base URL to https://ai-gateway.vercel.sh/cursor/v1"
	case acp.AuthenticationDirect:
		detail = "configure Cursor to use its direct model-provider routing"
	default:
		return
	}

	slog.Default().WarnContext(
		context.Background(),
		"auth mode cannot configure Cursor provider authentication; CURSOR_API_KEY is still required for Cursor CLI authentication",
		slog.String("harness", harnessID),
		slog.String("auth", string(mode)),
		slog.String("detail", detail),
	)
}
