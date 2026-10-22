package acp

import (
	"context"
	"encoding/json"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/model"
)

// Settings configure New. HarnessID, Source, Executable and ModelMapping
// are required.
type Settings struct {
	// Version is the ACP protocol version. Empty selects "v1", the only
	// supported version.
	Version string

	// HarnessID is the stable kebab-case identifier of the adapter, such as
	// "acp-codex".
	HarnessID string

	// BuiltinTools are the runtime's native tools, in declaration order. The
	// bridge maps ACP tool calls to them by name, title or input schema.
	BuiltinTools []harness.BuiltinTool

	// MCPServers are MCP server definitions keyed by server name, in the
	// runtime's native MCP server configuration format. The name
	// "ai-sdk-harness-tools" is reserved for application tools.
	MCPServers map[string]any

	// IsMCPToolCall reports runtime tool calls that belong to an MCP server,
	// which are emitted as dynamic tool calls.
	IsMCPToolCall func(toolCall ToolCall) bool

	// Port is the sandbox port of the bridge. Zero uses the first port the
	// network sandbox exposes. It is required with basic sandbox sessions.
	Port int

	// PortEndpoint overrides the host endpoint of the bridge. It is required,
	// together with Port, with basic sandbox sessions.
	PortEndpoint *harness.PortEndpoint

	// StartupTimeout bounds the bridge startup. Zero selects two minutes.
	StartupTimeout time.Duration

	// Reconnect bounds the reconnects after an established bridge connection
	// drops.
	Reconnect bridge.ReconnectOptions

	// ClientApp identifies the client to the ACP implementation. The zero
	// value selects "ai-sdk-harness-acp" with this package's Version.
	ClientApp ClientApp

	// Auth selects how provider credentials are discovered.
	Auth harness.Authentication

	// ResolveAuthenticationEnvironment resolves adapter-owned
	// authentication on the host before any environment values are
	// forwarded to the ACP implementation, for example from a native CLI
	// subscription. Nil uses Auth.Environment or the host environment.
	ResolveAuthenticationEnvironment func(ctx context.Context, event AuthenticationEnvironmentEvent) (map[string]string, error)

	// Source installs the ACP implementation in the sandbox: an
	// NPMSimpleSource, NPMLockedSource or InstallCommandSource.
	Source Source

	// Executable is the bare command name of the implementation, such as
	// "codex-acp".
	Executable string

	Args []string

	// ForwardEnv names host environment variables forwarded to the
	// implementation.
	ForwardEnv []string

	// CredentialEnv names host environment variables holding credentials.
	// It must be set together with CredentialBrokering.
	CredentialEnv []string

	// CredentialBrokering returns the request transformations that inject
	// the credentials outside the sandbox, which then only sees
	// placeholders. Sandboxes without request transformations receive the
	// credentials instead.
	CredentialBrokering func(event CredentialBrokeringEvent) ([]harness.RequestTransformation, error)

	// AuthenticationFiles materializes private authentication files below
	// the implementation's home directory before it starts.
	AuthenticationFiles func(event AuthenticationFilesEvent) ([]AuthenticationFile, error)

	// CredentialForwarding customizes each credential value before it is
	// forwarded into a sandbox process. It does not restrict which
	// credentials the adapter can read on the host.
	CredentialForwarding harness.CredentialForwardingFunc

	// Env holds runtime environment values that are safe to persist in the
	// bootstrap recipe and in the lifecycle compatibility identity.
	Env map[string]string

	// Authentication selects the ACP authentication method. Nil skips ACP
	// authentication.
	Authentication *Authentication

	// ClientCapabilities are merged into the client capabilities of the ACP
	// initialize request.
	ClientCapabilities map[string]any

	// ProviderAuthentication describes how the implementation is routed
	// through the AI Gateway.
	ProviderAuthentication *ProviderAuthentication

	// ModelMapping maps the agent's model identifier to an ACP session
	// operation. No operation is sent when the agent has no model.
	ModelMapping ModelMapping

	// SkillsDirectory is the native skills directory relative to the
	// implementation's home directory. Empty selects ".agents/skills".
	SkillsDirectory string

	// InstructionMapping routes the agent's instructions to a native system
	// or developer prompt. Nil prepends them to the first user prompt.
	InstructionMapping *InstructionMapping

	// OutputSchemaMapping maps a structured output JSON Schema below the ACP
	// prompt request's _meta field. Nil rejects structured output.
	OutputSchemaMapping *OutputSchemaMapping

	// HostToolMCPTransport is the transport of the harness-owned MCP server
	// exposing application tools. Empty selects HostToolsStdio.
	HostToolMCPTransport HostToolMCPTransport

	// NonTextContentTypes are the non-text tool output types the
	// implementation can deliver to its model. Undeclared image content is
	// omitted with a warning.
	NonTextContentTypes []NonTextContentType

	// AskUserQuestions maps the implementation's native question requests
	// to the common askUserQuestions tool, which it adds to the built-in
	// tools.
	AskUserQuestions *AskUserQuestionsSettings

	// PermissionModeMapping maps permission modes to ACP session operations.
	PermissionModeMapping *PermissionModeMapping

	// Session configures the ACP session requests.
	Session SessionSettings

	// MintBridgeToken creates the bridge token. Nil uses bridge.NewToken.
	// It requires a network sandbox session, whose ID it receives.
	MintBridgeToken harness.MintBridgeTokenFunc
}

// ClientApp identifies the ACP client.
type ClientApp struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// SessionSettings configure the ACP session requests.
type SessionSettings struct {
	// Meta is sent as the _meta field of the session requests.
	Meta map[string]any
}

// AuthenticationDirect selects direct provider credentials, never routing
// the implementation through the AI Gateway.
const AuthenticationDirect harness.AuthenticationMode = "direct"

// AuthenticationEnvironmentEvent is the input of
// Settings.ResolveAuthenticationEnvironment.
type AuthenticationEnvironmentEvent struct {
	Auth harness.Authentication

	// Env is the host environment.
	Env map[string]string
}

// CredentialBrokeringEvent is the input of Settings.CredentialBrokering.
type CredentialBrokeringEvent struct {
	// Env holds the real host values resolved for the implementation.
	Env map[string]string

	// SandboxEnv holds the values the sandbox sees, with placeholders in
	// place of credentials.
	SandboxEnv map[string]string

	// Headers are the agent's additional model request headers. It is nil
	// when there are none.
	Headers map[string]string
}

// AuthenticationFilesEvent is the input of Settings.AuthenticationFiles.
type AuthenticationFilesEvent struct {
	// Env holds the host values resolved for the implementation.
	Env map[string]string

	// SandboxEnv holds the values exposed to the sandbox process after
	// credential brokering.
	SandboxEnv map[string]string

	CredentialBrokeringAvailable bool
}

// AuthenticationFile is a private file written below the implementation's
// home directory.
type AuthenticationFile struct {
	// Path is relative to the implementation's home directory.
	Path    string
	Content string
}

// Source installs the ACP implementation: NPMSimpleSource, NPMLockedSource
// or InstallCommandSource.
type Source interface {
	isSource()
}

// NPMSimpleSource installs one npm package.
type NPMSimpleSource struct {
	PackageName string

	// PackageVersion is the exact version to install. Empty installs the
	// latest dist-tag, and no version takes part in the implementation
	// identity, so a new release does not invalidate lifecycle state.
	PackageVersion string
}

// NPMLockedSource installs a locked pnpm project.
type NPMLockedSource struct {
	PackageJSON  string
	PNPMLockYAML string

	// PNPMWorkspaceYAML is the optional workspace configuration the locked
	// install needs.
	PNPMWorkspaceYAML string
}

// InstallCommandSource installs the implementation with a shell command
// that places the executable in $HOME/.local/bin of a private home.
type InstallCommandSource struct {
	Command string
}

func (NPMSimpleSource) isSource()      {}
func (NPMLockedSource) isSource()      {}
func (InstallCommandSource) isSource() {}

// Authentication selects the ACP authenticate request.
type Authentication struct {
	MethodID string `json:"methodId"`

	// Meta is the _meta field of the authenticate request.
	Meta map[string]any `json:"meta,omitzero"`

	// ClientCapabilities are merged into the initialize request's client
	// capabilities.
	ClientCapabilities map[string]any `json:"clientCapabilities,omitzero"`
}

// ProviderAuthentication describes the AI Gateway routing of the
// implementation.
type ProviderAuthentication struct {
	Gateway GatewayAuthentication `json:"gateway"`
}

// GatewayAuthentication is the launch environment that routes the
// implementation through the AI Gateway. Each value is a profile value:
// JSON-compatible data (strings, numbers, booleans, nil, []any and
// map[string]any) in which ValueSource values are replaced by gateway
// values. Values that are not strings are JSON-encoded.
type GatewayAuthentication struct {
	Env map[string]any `json:"env"`
}

// ValueSourceName names a value a ValueSource resolves to.
type ValueSourceName string

// Value sources.
const (
	SourceGatewayAPIKey        ValueSourceName = "gateway-api-key"
	SourceGatewayBaseURL       ValueSourceName = "gateway-base-url"
	SourceGatewayAuthorization ValueSourceName = "gateway-authorization"
	SourceClientApp            ValueSourceName = "client-app"
	SourceClientAppName        ValueSourceName = "client-app-name"
	SourceClientAppVersion     ValueSourceName = "client-app-version"
)

// ValueSource is a profile value resolved from the gateway configuration or
// the client app, with an optional prefix and suffix. EnsureSuffix appends
// its suffix after trailing slashes are removed, unless the value already
// ends with it; it replaces Suffix.
type ValueSource struct {
	Source       ValueSourceName `json:"$source"`
	Prefix       string          `json:"prefix,omitempty"`
	Suffix       string          `json:"suffix,omitempty"`
	EnsureSuffix string          `json:"ensureSuffix,omitempty"`
}

// ModelMappingType selects how the model is applied.
type ModelMappingType string

// Model mapping types.
const (
	// ModelSessionConfigOption sets the session configuration option Path.
	ModelSessionConfigOption ModelMappingType = "session-config-option"

	// ModelSessionModel calls the legacy session/set_model method with the
	// model in the request property Path.
	ModelSessionModel ModelMappingType = "session-model"
)

// ModelMapping maps the agent's model to an ACP session operation.
type ModelMapping struct {
	Type ModelMappingType `json:"type"`
	Path string           `json:"path"`
}

// PermissionModeTargetType selects how a permission mode is applied.
type PermissionModeTargetType string

// Permission mode target types.
const (
	TargetSessionMode         PermissionModeTargetType = "session-mode"
	TargetSessionConfigOption PermissionModeTargetType = "session-config-option"
)

// PermissionModeTarget is the ACP session operation of a permission mode:
// a session mode with ModeID, or a session configuration option with
// ConfigID and Value.
type PermissionModeTarget struct {
	Type     PermissionModeTargetType `json:"type"`
	ModeID   string                   `json:"modeId,omitempty"`
	ConfigID string                   `json:"configId,omitempty"`

	// Value is a string or a bool.
	Value any `json:"value,omitempty"`
}

// PermissionModeMapping maps each permission mode to an ACP session
// operation. A nil target applies no operation for that mode.
type PermissionModeMapping struct {
	AllowReads *PermissionModeTarget `json:"allow-reads"`
	AllowEdits *PermissionModeTarget `json:"allow-edits"`
	AllowAll   *PermissionModeTarget `json:"allow-all"`
}

// InstructionMappingType selects where instructions are applied.
type InstructionMappingType string

// Instruction mapping types.
const (
	// InstructionsSessionMeta writes the instructions below Path of the ACP
	// session request's _meta field.
	InstructionsSessionMeta InstructionMappingType = "session-meta"

	// InstructionsLaunchEnvJSON merges the instructions at Path into the
	// JSON object of the environment variable Variable before the
	// implementation starts.
	InstructionsLaunchEnvJSON InstructionMappingType = "launch-env-json"

	// InstructionsFilesystem writes the instructions to the markdown File,
	// relative to the implementation's home directory.
	InstructionsFilesystem InstructionMappingType = "filesystem"
)

// InstructionMapping describes where the implementation accepts native
// session instructions.
type InstructionMapping struct {
	Type InstructionMappingType

	// Path is the JSON path of the session-meta and launch-env-json types.
	Path []string

	// Variable is the environment variable of the launch-env-json type.
	Variable string

	// File is the relative markdown file of the filesystem type.
	File string
}

// MarshalJSON encodes the mapping in the upstream shape, where the
// filesystem type carries its file as path.
func (m InstructionMapping) MarshalJSON() ([]byte, error) {
	switch m.Type {
	case InstructionsFilesystem:
		return json.Marshal(struct {
			Type InstructionMappingType `json:"type"`
			Path string                 `json:"path"`
		}{
			Type: m.Type,
			Path: m.File,
		})
	case InstructionsLaunchEnvJSON:
		return json.Marshal(struct {
			Type     InstructionMappingType `json:"type"`
			Variable string                 `json:"variable"`
			Path     []string               `json:"path"`
		}{
			Type:     m.Type,
			Variable: m.Variable,
			Path:     m.Path,
		})
	default:
		return json.Marshal(struct {
			Type InstructionMappingType `json:"type"`
			Path []string               `json:"path"`
		}{
			Type: m.Type,
			Path: m.Path,
		})
	}
}

// OutputSchemaMappingType selects where the output schema is applied.
type OutputSchemaMappingType string

// OutputSchemaPromptMeta writes the schema below Path of the ACP prompt
// request's _meta field.
const OutputSchemaPromptMeta OutputSchemaMappingType = "session-prompt-meta"

// OutputSchemaMapping maps a structured output JSON Schema into the ACP
// prompt request.
type OutputSchemaMapping struct {
	Type OutputSchemaMappingType `json:"type"`
	Path []string                `json:"path"`
}

// HostToolMCPTransport is the transport of the harness-owned MCP server
// that exposes application tools to the implementation.
type HostToolMCPTransport string

// Host tool MCP transports.
const (
	HostToolsStdio HostToolMCPTransport = "stdio"

	// HostToolsHTTP serves the tools over HTTP, for implementations that
	// only accept HTTP or SSE MCP servers. The implementation must advertise
	// agentCapabilities.mcpCapabilities.http.
	HostToolsHTTP HostToolMCPTransport = "http"
)

// NonTextContentType is a non-text tool output type.
type NonTextContentType string

// NonTextImage is image tool output.
const NonTextImage NonTextContentType = "image"

// AskUserQuestionsSettings map an implementation's native question
// requests, received as the JSON-RPC method RequestMethod, to the common
// askUserQuestions tool.
type AskUserQuestionsSettings struct {
	RequestMethod string

	// IsNativeToolCall reports the runtime tool call that carries a question
	// request, so that it is not emitted as a built-in tool call.
	IsNativeToolCall func(nativeToolCall ToolCall) bool

	// FromNativeRequest converts a native request into a client-executed
	// askUserQuestions tool call. A nil call leaves the request unhandled.
	FromNativeRequest func(request NativeQuestionRequest) (*harness.ToolCallPart, error)

	// ToNativeResponse converts the tool result into the native response.
	ToNativeResponse func(response NativeQuestionResponse) (any, error)

	// MatchesNativeRequest reports whether a request re-sent after a
	// recovery is the request a buffered answer belongs to.
	MatchesNativeRequest func(match NativeQuestionMatch) bool
}

// NativeQuestionRequest is the input of
// AskUserQuestionsSettings.FromNativeRequest.
type NativeQuestionRequest struct {
	NativeRequest json.RawMessage

	// NativeToolCall is the runtime tool call announcing the request, when
	// the runtime sent one.
	NativeToolCall *ToolCall
}

// NativeQuestionResponse is the input of
// AskUserQuestionsSettings.ToNativeResponse.
type NativeQuestionResponse struct {
	NativeRequest json.RawMessage
	ToolResult    model.ToolResultPart
}

// NativeQuestionMatch is the input of
// AskUserQuestionsSettings.MatchesNativeRequest.
type NativeQuestionMatch struct {
	PreviousNativeRequest json.RawMessage
	NativeRequest         json.RawMessage
}
