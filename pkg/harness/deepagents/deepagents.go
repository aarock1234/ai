// Package deepagents is the Deep Agents harness adapter, the Go port of
// upstream @ai-sdk/harness-deepagents. It runs a LangChain Deep Agents agent
// inside the sandbox behind a small Node.js bridge and drives it over the
// bridge protocol of harness/bridge.
//
// Deep Agents always drives the Anthropic client. Set ANTHROPIC_API_KEY or
// ANTHROPIC_AUTH_TOKEN for direct authentication, or AI_GATEWAY_API_KEY or
// VERCEL_OIDC_TOKEN to reach any model through the AI Gateway's
// Anthropic-compatible endpoint.
//
// The bridge is JavaScript that runs in the sandbox, so this package ships
// it as files of its bootstrap recipe (see Harness.Bootstrap): the bundled
// bridge of the published @ai-sdk/harness-deepagents package, embedded
// verbatim, so the sandbox needs Node.js and pnpm.
//
// Go adaptations: context.Context replaces abort signals; optional numbers
// and strings use their zero values for unset; and a respawned bridge
// replays a suspended turn from its event log, like the other bridge
// adapters of this module. Upstream's default deepAgents instance
// corresponds to New(Settings{}).
package deepagents

import (
	"context"
	"encoding/json"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// HarnessID identifies the adapter.
const HarnessID = "deepagents"

// Version is the upstream adapter version this package ports. It is sent in
// the client application attribution of model requests.
const Version = "1.0.157"

// clientApp attributes model requests to the adapter in the User-Agent and
// x-client-app headers.
const clientApp = "ai-sdk-harness-deepagents/" + Version

// Settings configure New. The zero value is a valid configuration.
type Settings struct {
	// Auth selects the authentication: harness.AuthenticationAuto (the
	// default), harness.AuthenticationAIGateway, AuthenticationAnthropic, or
	// an explicit environment.
	Auth harness.Authentication

	// CredentialForwarding customizes each credential before it is forwarded
	// into a sandbox process. It does not restrict which credentials the
	// adapter reads on the host.
	CredentialForwarding harness.CredentialForwardingFunc

	// Thinking controls Anthropic extended thinking. Nil keeps the Deep
	// Agents default.
	Thinking *Thinking

	// Effort controls how much effort Claude applies with adaptive thinking.
	// Empty uses the LangChain Anthropic client default.
	Effort Effort

	// RecursionLimit caps the LangGraph super-steps of a turn before it
	// fails. Zero keeps the Deep Agents default.
	RecursionLimit int

	// MCPServers are MCP server definitions keyed by server name, in the
	// LangChain MCP adapters' configuration format.
	MCPServers map[string]any

	// ConnectionSettings configure the bridge connection.
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

// Thinking configures extended thinking. BudgetTokens applies to
// ThinkingEnabled, which requires it, and Display does not apply to
// ThinkingDisabled.
type Thinking struct {
	Type         ThinkingType    `json:"type"`
	BudgetTokens int             `json:"budget_tokens,omitempty"`
	Display      ThinkingDisplay `json:"display,omitempty"`
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

// Harness is the Deep Agents harness adapter. It implements
// harness.Harness, harness.Bootstrapper, harness.LifecycleStateSchemaProvider
// and harness.BuiltinToolApprovalSupporter.
type Harness struct {
	settings Settings
}

// New returns a Deep Agents harness.
func New(settings Settings) *Harness {
	return &Harness{
		settings: settings,
	}
}

// HarnessID returns "deepagents".
func (h *Harness) HarnessID() string {
	return HarnessID
}

// BuiltinTools returns every model-callable Deep Agents tool.
func (h *Harness) BuiltinTools() []harness.BuiltinTool {
	return builtinTools()
}

// SupportsBuiltinToolApprovals reports true: the bridge gates built-in
// tools through the Deep Agents human-in-the-loop middleware.
func (h *Harness) SupportsBuiltinToolApprovals() bool {
	return true
}

// LifecycleStateSchema returns the JSON Schema of the adapter's lifecycle
// state data.
func (h *Harness) LifecycleStateSchema() json.RawMessage {
	return json.RawMessage(lifecycleStateSchema)
}

// Bootstrap returns the recipe that installs the bridge and its Deep Agents
// dependencies into the sandbox.
func (h *Harness) Bootstrap(_ context.Context) (*harness.Bootstrap, error) {
	return bootstrapRecipe(), nil
}

// The adapter implements the harness specification and its optional
// capabilities.
var (
	_ harness.Harness                      = (*Harness)(nil)
	_ harness.Bootstrapper                 = (*Harness)(nil)
	_ harness.LifecycleStateSchemaProvider = (*Harness)(nil)
	_ harness.BuiltinToolApprovalSupporter = (*Harness)(nil)
)
