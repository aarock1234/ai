// Package claudecode is the Claude Code harness adapter, the Go port of
// upstream @ai-sdk/harness-claude-code. It runs the Claude Agent SDK inside
// the sandbox behind a small Node.js bridge and drives it over the bridge
// protocol of harness/bridge.
//
// The bridge is JavaScript that runs in the sandbox, so this package ships
// it as files of its bootstrap recipe (see Harness.Bootstrap): the bundled
// bridge of the published @ai-sdk/harness-claude-code package, embedded
// verbatim, so the sandbox needs Node.js and pnpm.
//
// Go adaptations: context.Context replaces abort signals; optional numbers
// and strings use their zero values for unset; and the deferred start of a
// turn, which upstream needs for its event loop, is not needed.
package claudecode

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// HarnessID identifies the adapter.
const HarnessID = "claude-code"

// Version is the upstream adapter version this package ports. It is sent in
// the client application attribution of AI Gateway requests.
const Version = "1.0.161"

// displayName names the adapter in messages.
const displayName = "Claude Code"

// clientApp attributes AI Gateway requests to the adapter.
const clientApp = "ai-sdk-harness-claude-code/" + Version

// reservedMCPServerName is the MCP server the bridge uses for the
// application's tools.
const reservedMCPServerName = "harness-tools"

// AuthenticationDirect authenticates Claude Code directly with Anthropic, or
// with a native Claude subscription.
const AuthenticationDirect harness.AuthenticationMode = "direct"

// ErrReservedMCPServerName reports an MCP server named "harness-tools", which
// the adapter reserves for the application's tools.
var ErrReservedMCPServerName = errors.New(`claude code mcp server name "harness-tools" is reserved for harness agent tools`)

// Settings configure New.
type Settings struct {
	// Auth selects the authentication: harness.AuthenticationAuto (the
	// default), harness.AuthenticationAIGateway, AuthenticationDirect, or an
	// explicit environment.
	Auth harness.Authentication

	// CredentialForwarding customizes each credential before it is forwarded
	// into a sandbox process. It does not restrict which credentials the
	// adapter reads on the host.
	CredentialForwarding harness.CredentialForwardingFunc

	// MCPServers are MCP server definitions keyed by server name, in Claude
	// Code's native configuration format.
	MCPServers map[string]any

	// MaxTurns caps the internal turns of the runtime before it yields. Zero
	// uses the runtime default.
	MaxTurns int

	// AgentProgressSummaries enables periodic progress summaries of running
	// subagents, forwarded in raw task_progress parts. Nil uses the runtime
	// default.
	AgentProgressSummaries *bool

	// ForwardSubagentText forwards subagent text and thinking messages as raw
	// parts in addition to their tool activity. Nil uses the runtime default.
	ForwardSubagentText *bool

	// Env is merged over the bridge process environment of the Claude Code
	// process.
	Env map[string]string

	// Thinking controls extended thinking. Nil selects adaptive thinking with
	// summarized reasoning.
	Thinking *Thinking

	// Effort controls how much effort Claude applies with adaptive thinking.
	// Empty uses the runtime default.
	Effort Effort

	// ConnectionSettings configure the bridge connection. Its HTTPClient
	// also sends the OAuth token refreshes of a native subscription.
	bridge.ConnectionSettings
}

// ThinkingType selects the extended-thinking behavior.
type ThinkingType string

// Thinking types.
const (
	ThinkingAdaptive ThinkingType = "adaptive"
	ThinkingEnabled  ThinkingType = "enabled"
	ThinkingDisabled ThinkingType = "disabled"
)

// ThinkingDisplay selects whether reasoning is summarized or omitted.
type ThinkingDisplay string

// Thinking displays.
const (
	ThinkingSummarized ThinkingDisplay = "summarized"
	ThinkingOmitted    ThinkingDisplay = "omitted"
)

// Thinking configures extended thinking. Display does not apply to
// ThinkingDisabled.
type Thinking struct {
	Type    ThinkingType    `json:"type"`
	Display ThinkingDisplay `json:"display,omitempty"`
}

// Effort is how much effort Claude applies with adaptive thinking.
type Effort string

// Efforts.
const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
	EffortXHigh  Effort = "xhigh"
	EffortMax    Effort = "max"
)

// Harness is the Claude Code harness adapter. It implements harness.Harness,
// harness.Bootstrapper, harness.LifecycleStateSchemaProvider,
// harness.BuiltinToolApprovalSupporter and
// harness.BuiltinToolFilteringSupporter.
type Harness struct {
	settings Settings
	thinking Thinking
}

// New returns a Claude Code harness. It fails when Settings.MCPServers uses
// the reserved name "harness-tools".
func New(settings Settings) (*Harness, error) {
	if _, reserved := settings.MCPServers[reservedMCPServerName]; reserved {
		return nil, ErrReservedMCPServerName
	}

	thinking := Thinking{
		Type:    ThinkingAdaptive,
		Display: ThinkingSummarized,
	}
	if settings.Thinking != nil {
		thinking = *settings.Thinking
	}

	return &Harness{
		settings: settings,
		thinking: thinking,
	}, nil
}

// HarnessID returns "claude-code".
func (h *Harness) HarnessID() string {
	return HarnessID
}

// BuiltinTools returns the native tools Claude Code can invoke.
func (h *Harness) BuiltinTools() []harness.BuiltinTool {
	return builtinTools()
}

// SupportsBuiltinToolApprovals reports true: the bridge asks for approval of
// built-in tools when the permission mode requires it.
func (h *Harness) SupportsBuiltinToolApprovals() bool {
	return true
}

// SupportsBuiltinToolFiltering reports true: the bridge hides inactive
// built-in tools from Claude Code.
func (h *Harness) SupportsBuiltinToolFiltering() bool {
	return true
}

// LifecycleStateSchema returns the JSON Schema of the adapter's lifecycle
// state data.
func (h *Harness) LifecycleStateSchema() json.RawMessage {
	return json.RawMessage(lifecycleStateSchema)
}

// Bootstrap returns the recipe that installs the bridge and the Claude Code
// CLI into the sandbox.
func (h *Harness) Bootstrap(_ context.Context) (*harness.Bootstrap, error) {
	return bootstrapRecipe(), nil
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
