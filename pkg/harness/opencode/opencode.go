// Package opencode is the OpenCode harness adapter, the Go port of upstream
// @ai-sdk/harness-opencode. It runs an OpenCode server inside the sandbox
// behind a small Node.js bridge and drives it over the bridge protocol of
// harness/bridge.
//
// The bridge is JavaScript that runs in the sandbox, so this package ships
// it as files of its bootstrap recipe (see Harness.Bootstrap): upstream's
// bundled bridge verbatim, so the sandbox needs Node.js 22 or later and
// pnpm.
//
// Go adaptations: context.Context replaces abort signals, and also cancels a
// manual compaction; optional strings use their zero values for unset; and
// the start command is sent as soon as the turn is wired, because Go has no
// event loop to defer to.
package opencode

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// HarnessID identifies the adapter.
const HarnessID = "opencode"

// Version is the upstream adapter version this package ports. It is sent in
// the client application attribution of model requests.
const Version = "1.0.159"

// displayName names the adapter in messages.
const displayName = "OpenCode"

// clientApp attributes model requests to the adapter.
const clientApp = "ai-sdk-harness-opencode/" + Version

// reservedMCPServerName is the MCP server the bridge uses for the
// application's tools.
const reservedMCPServerName = "harness-tools"

// Authentication modes that pin a direct provider.
const (
	// AuthenticationAnthropic authenticates OpenCode directly with
	// Anthropic.
	AuthenticationAnthropic harness.AuthenticationMode = "anthropic"

	// AuthenticationOpenAI authenticates OpenCode directly with OpenAI.
	AuthenticationOpenAI harness.AuthenticationMode = "openai"
)

// ErrReservedMCPServerName reports an MCP server named "harness-tools", which
// the adapter reserves for the application's tools.
var ErrReservedMCPServerName = errors.New(`opencode mcp server name "harness-tools" is reserved for harness agent tools`)

// Settings configure New.
type Settings struct {
	// Auth selects the authentication: harness.AuthenticationAuto (the
	// default), harness.AuthenticationAIGateway, AuthenticationAnthropic,
	// AuthenticationOpenAI, or an explicit environment. Without credentials
	// the adapter falls back to OpenCode's native login of the selected
	// provider.
	Auth harness.Authentication

	// CredentialForwarding customizes each credential before it is forwarded
	// into a sandbox process. It does not restrict which credentials the
	// adapter reads on the host.
	CredentialForwarding harness.CredentialForwardingFunc

	// OpenCodeConfig is passed to OpenCode as-is. OpenCode configuration
	// keys use their native names. Values the adapter manages take
	// precedence over conflicting entries.
	OpenCodeConfig map[string]any

	// MCPServers are MCP server definitions keyed by server name, in
	// OpenCode's native configuration format.
	MCPServers map[string]any

	// Provider is the OpenCode provider of models named without a provider
	// prefix, such as "anthropic" or "github-copilot".
	Provider string

	// ReasoningVariant is the OpenCode reasoning variant of
	// reasoning-capable models, such as "low", "medium" or "high". Empty
	// uses the runtime default.
	ReasoningVariant string

	// ConnectionSettings configure the bridge connection. Its HTTPClient
	// also sends the requests of a native login: OAuth token refreshes and
	// the GitLab direct access exchange.
	bridge.ConnectionSettings
}

// Harness is the OpenCode harness adapter. It implements harness.Harness,
// harness.Bootstrapper, harness.LifecycleStateSchemaProvider,
// harness.BuiltinToolApprovalSupporter and
// harness.BuiltinToolFilteringSupporter.
type Harness struct {
	settings Settings
}

// New returns an OpenCode harness. It fails when Settings.MCPServers uses the
// reserved name "harness-tools".
func New(settings Settings) (*Harness, error) {
	if _, reserved := settings.MCPServers[reservedMCPServerName]; reserved {
		return nil, ErrReservedMCPServerName
	}

	return &Harness{
		settings: settings,
	}, nil
}

// HarnessID returns "opencode".
func (h *Harness) HarnessID() string {
	return HarnessID
}

// BuiltinTools returns the native tools OpenCode can invoke.
func (h *Harness) BuiltinTools() []harness.BuiltinTool {
	return builtinTools()
}

// SupportsBuiltinToolApprovals reports true: the bridge asks for approval of
// built-in tools when the permission mode requires it.
func (h *Harness) SupportsBuiltinToolApprovals() bool {
	return true
}

// SupportsBuiltinToolFiltering reports true: the bridge disables inactive
// built-in tools in OpenCode.
func (h *Harness) SupportsBuiltinToolFiltering() bool {
	return true
}

// LifecycleStateSchema returns the JSON Schema of the adapter's lifecycle
// state data.
func (h *Harness) LifecycleStateSchema() json.RawMessage {
	return json.RawMessage(lifecycleStateSchema)
}

// Bootstrap returns the recipe that installs the bridge and the OpenCode CLI
// into the sandbox.
func (h *Harness) Bootstrap(_ context.Context) (*harness.Bootstrap, error) {
	return bootstrapRecipe(), nil
}

// unsupported returns a capability error of the adapter.
func unsupported(message string) error {
	return &harness.CapabilityUnsupportedError{
		Message:   message,
		HarnessID: HarnessID,
	}
}

// The adapter implements the harness specification and its optional
// capabilities.
var (
	_ harness.Harness                       = (*Harness)(nil)
	_ harness.Bootstrapper                  = (*Harness)(nil)
	_ harness.LifecycleStateSchemaProvider  = (*Harness)(nil)
	_ harness.BuiltinToolApprovalSupporter  = (*Harness)(nil)
	_ harness.BuiltinToolFilteringSupporter = (*Harness)(nil)
)
