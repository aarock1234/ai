// Package codex is the Codex harness adapter, the Go port of upstream
// @ai-sdk/harness-codex. It runs the Codex app-server inside the sandbox
// behind a small Node.js bridge and drives it over the bridge protocol of
// harness/bridge.
//
// The bridge is JavaScript that runs in the sandbox, so this package ships
// it as files of its bootstrap recipe (see Harness.Bootstrap): the bundled
// bridge of the published @ai-sdk/harness-codex package, embedded verbatim,
// so the sandbox needs Node.js and pnpm.
//
// Go adaptations: context.Context replaces abort signals; optional strings
// use their zero values for unset; and the start command is sent as soon as
// the turn is wired, because Go has no event loop to defer to.
package codex

import (
	"context"
	"encoding/json"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// HarnessID identifies the adapter.
const HarnessID = "codex"

// Version is the upstream adapter version this package ports. It is sent in
// the client application attribution of model requests.
const Version = "1.0.159"

// clientApp attributes model requests to the adapter.
const clientApp = "ai-sdk-harness-codex/" + Version

// DefaultModel is the model of turns that select none. It is intentionally
// not the latest Codex model: newer models use Responses Lite, which does not
// expose their code-mode tools through the custom model provider the bridge
// uses (https://github.com/openai/codex/issues/31894).
const DefaultModel = "gpt-5.5"

// AuthenticationDirect authenticates Codex directly with OpenAI, or with a
// native ChatGPT subscription.
const AuthenticationDirect harness.AuthenticationMode = "direct"

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

	// CodexConfig is passed to Codex as-is. Codex configuration keys
	// typically use snake_case. Values the adapter manages take precedence
	// over conflicting entries.
	CodexConfig map[string]any

	// MCPServers are MCP server definitions keyed by server name, in Codex's
	// native configuration format.
	MCPServers map[string]any

	// ReasoningEffort is the reasoning effort of reasoning-capable models.
	// Empty uses the runtime default.
	ReasoningEffort ReasoningEffort

	// WebSearch allows the runtime to use live web search. Nil uses the
	// runtime default.
	WebSearch *bool

	// ConnectionSettings configure the bridge connection. Its HTTPClient
	// also sends the OAuth token refreshes of a native subscription.
	bridge.ConnectionSettings
}

// ReasoningEffort is the reasoning effort of reasoning-capable models.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningLow    ReasoningEffort = "low"
	ReasoningMedium ReasoningEffort = "medium"
	ReasoningHigh   ReasoningEffort = "high"
	ReasoningXHigh  ReasoningEffort = "xhigh"
	ReasoningMax    ReasoningEffort = "max"
)

// Harness is the Codex harness adapter. It implements harness.Harness,
// harness.Bootstrapper, harness.LifecycleStateSchemaProvider,
// harness.BuiltinToolApprovalSupporter and
// harness.BuiltinToolFilteringSupporter.
type Harness struct {
	settings Settings
}

// New returns a Codex harness.
func New(settings Settings) *Harness {
	return &Harness{
		settings: settings,
	}
}

// HarnessID returns "codex".
func (h *Harness) HarnessID() string {
	return HarnessID
}

// BuiltinTools returns the native tools Codex can invoke as model-callable
// tools.
func (h *Harness) BuiltinTools() []harness.BuiltinTool {
	return builtinTools()
}

// SupportsBuiltinToolApprovals reports false: Codex runs with every
// built-in tool allowed.
func (h *Harness) SupportsBuiltinToolApprovals() bool {
	return false
}

// SupportsBuiltinToolFiltering reports true: the bridge hides inactive
// built-in tools from Codex.
func (h *Harness) SupportsBuiltinToolFiltering() bool {
	return true
}

// LifecycleStateSchema returns the JSON Schema of the adapter's lifecycle
// state data.
func (h *Harness) LifecycleStateSchema() json.RawMessage {
	return json.RawMessage(lifecycleStateSchema)
}

// Bootstrap returns the recipe that installs the bridge and the Codex CLI
// into the sandbox.
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
