// Package pi is the Pi harness adapter, the Go port of upstream
// @ai-sdk/harness-pi, backed by the Pi coding agent
// (@earendil-works/pi-coding-agent).
//
// Upstream runs Pi in the host Node.js process and proxies its tools to the
// sandbox. Go cannot host Pi in-process, so this adapter runs Pi's documented
// RPC mode (pi --mode rpc) as a long-lived process inside the sandbox and
// exchanges strict JSONL records with it: commands, correlated responses,
// session events and extension UI records. Its built-in tools therefore run
// natively in the sandbox, and model requests leave from the sandbox, where
// the adapter brokers credentials like the other in-sandbox adapters.
//
// Sandbox processes expose no standard input, so a small relay (a Node.js
// script installed by the bootstrap recipe) owns Pi's standard input and
// output: the host writes each record as a numbered file the relay feeds to
// Pi, and the relay appends Pi's output to a file the host follows from a
// byte offset. A process that detaches or suspends leaves Pi running, and a
// later Start reattaches at the recorded offset without losing records.
//
// A Pi extension installed with the relay replaces Pi's built-in tools with
// the upstream tool declarations, registers the application tools of each
// turn and answers their calls with the results the host submits through the
// RPC extension UI protocol, asks the host for built-in tool approvals the
// same way, and applies each turn's instructions.
//
// Go adaptations: context.Context replaces abort signals; settings are one
// Settings struct; upstream's inline extension factories become extension
// source files loaded inside the sandbox; and upstream's in-process session
// parking becomes a live Pi process in the sandbox that any host process can
// reattach to.
package pi

import (
	"context"
	"encoding/json"
	"slices"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
)

// HarnessID identifies the adapter.
const HarnessID = "pi"

// Version is the upstream adapter version this package ports. It is sent in
// the client application attribution of model requests.
const Version = "1.0.159"

// PiVersion is the pinned version of the Pi coding agent the bootstrap
// recipe installs, the version upstream depends on.
const PiVersion = "1.0.3"

// clientApp attributes model requests to the adapter.
const clientApp = "ai-sdk-harness-pi/" + Version

// defaultStartupTimeout bounds the wait for Pi to answer its first command.
const defaultStartupTimeout = 120 * time.Second

// Authentication modes beyond the shared harness.AuthenticationAuto and
// harness.AuthenticationAIGateway.
const (
	// AuthenticationOpenAI uses OPENAI_API_KEY and OPENAI_BASE_URL.
	AuthenticationOpenAI harness.AuthenticationMode = "openai"

	// AuthenticationAnthropic uses ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL and
	// ANTHROPIC_AUTH_TOKEN.
	AuthenticationAnthropic harness.AuthenticationMode = "anthropic"

	// AuthenticationCustom registers every provider whose <PREFIX>_API_KEY
	// and <PREFIX>_BASE_URL are set.
	AuthenticationCustom harness.AuthenticationMode = "custom"
)

// ThinkingLevel is Pi's extended-thinking budget level.
type ThinkingLevel string

// Thinking levels.
const (
	ThinkingOff     ThinkingLevel = "off"
	ThinkingMinimal ThinkingLevel = "minimal"
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXHigh   ThinkingLevel = "xhigh"
	ThinkingMax     ThinkingLevel = "max"
)

// CacheRetention is the prompt cache retention of model requests.
type CacheRetention string

// Cache retentions.
const (
	CacheRetentionNone  CacheRetention = "none"
	CacheRetentionShort CacheRetention = "short"
	CacheRetentionLong  CacheRetention = "long"
)

// Settings configure New.
type Settings struct {
	// Auth selects where Pi's model credentials come from:
	// harness.AuthenticationAuto (the default) prefers the AI Gateway and
	// falls back to provider credentials, harness.AuthenticationAIGateway,
	// AuthenticationOpenAI, AuthenticationAnthropic and AuthenticationCustom
	// pin a route, and an explicit environment replaces the host environment.
	Auth harness.Authentication

	// CredentialForwarding customizes each credential before it is forwarded
	// into the sandbox. It does not restrict which credentials the adapter
	// reads on the host.
	CredentialForwarding harness.CredentialForwardingFunc

	// Credentials is application-owned storage for Pi's credential document
	// (the content of Pi's auth.json). When set, it replaces Pi's native
	// auth.json for every authentication mode.
	Credentials CredentialStore

	// ReattachLiveRuntime reports whether Detach and SuspendTurn leave Pi
	// running in the sandbox, so that a later Start, in this or another
	// process, reattaches to it. Disable it in stateless or multi-replica
	// applications so that every resume restores Pi from its persisted
	// session file with the current settings. Nil enables it.
	ReattachLiveRuntime *bool

	// Providers are explicit Pi provider configurations keyed by provider ID,
	// in Pi's native ProviderConfig shape, such as custom models and their
	// API protocol. They are registered after the providers of the
	// authentication.
	Providers map[string]map[string]any

	// ThinkingLevel is Pi's extended-thinking level. Empty uses Pi's default.
	ThinkingLevel ThinkingLevel

	// CacheRetention is the default prompt cache retention. For Anthropic
	// models, CacheRetentionLong uses the 1-hour cache. Empty uses Pi's
	// default (CacheRetentionShort).
	CacheRetention CacheRetention

	// AgentDir is the host directory holding Pi's global agent configuration
	// (auth.json, models.json and settings.json). Its models.json and
	// settings.json are copied into the session's private agent directory.
	// When empty, native subscription credentials are read from
	// PI_CODING_AGENT_DIR or ~/.pi/agent, and model and general settings stay
	// isolated per session.
	AgentDir string

	// MCPServers are MCP servers keyed by server name, in Pi's native MCP
	// server configuration (stdio or streamable HTTP). Tools are named
	// mcp__<server>__<tool>. Exposure defaults to "direct"; "deferred" tools
	// stay undeclared until Pi's tool_search tool finds them. The codemode
	// exposures are not supported.
	MCPServers map[string]map[string]any

	// Extensions are trusted Pi extension sources loaded in every session, in
	// order, after the adapter's own extension. They run inside the sandbox;
	// see Extension. Filesystem-discovered user and project extensions stay
	// disabled.
	Extensions []Extension

	// FileToolPathPolicy widens or narrows what the native file tools can
	// reach. Nil confines them to the session working directory and the
	// harness skills directory.
	FileToolPathPolicy *FileToolPathPolicy

	// StartupTimeout bounds the wait for Pi to become ready. Zero selects two
	// minutes.
	StartupTimeout time.Duration
}

// Extension is a trusted Pi extension source file, the Go form of upstream's
// inline extension factories. The adapter writes it into the session's
// private directory inside the sandbox and loads it with Pi's --extension
// option. The extension reads the current turn's instructions and
// application tools from the JSON file named by the
// AI_SDK_HARNESS_PI_TURN_FILE environment variable; its working directory
// is the session working directory.
type Extension struct {
	// Name is the file name, such as "review.ts". It must be a plain file
	// name of letters, digits, dots, underscores and hyphens ending in .ts or
	// .js.
	Name string

	// Content is the TypeScript or JavaScript source.
	Content string
}

// FileToolPathPolicy holds absolute sandbox paths that widen or narrow what
// Pi's native file tools (read, write, edit, grep, glob and ls) can reach.
// The bash tool is not restricted.
type FileToolPathPolicy struct {
	// ReadableRoots lets the read-only tools reach outside the workspace.
	ReadableRoots []string `json:"readableRoots,omitempty"`

	// WritableRoots also permits writes and edits outside the workspace, and
	// grants read access.
	WritableRoots []string `json:"writableRoots,omitempty"`

	// DeniedRoots are refused by every native file tool and take precedence,
	// including through symbolic links.
	DeniedRoots []string `json:"deniedRoots,omitempty"`
}

// Harness is the Pi harness adapter. It implements harness.Harness,
// harness.Bootstrapper, harness.LifecycleStateSchemaProvider,
// harness.BuiltinToolApprovalSupporter and
// harness.BuiltinToolFilteringSupporter.
type Harness struct {
	settings Settings
}

// New validates the settings and returns a Pi harness.
func New(settings Settings) (*Harness, error) {
	if err := validateSettings(settings); err != nil {
		return nil, err
	}

	return &Harness{
		settings: settings,
	}, nil
}

// HarnessID returns "pi".
func (h *Harness) HarnessID() string {
	return HarnessID
}

// BuiltinTools returns Pi's built-in file and shell tools.
func (h *Harness) BuiltinTools() []harness.BuiltinTool {
	return slices.Clone(builtinTools)
}

// SupportsBuiltinToolApprovals reports true: the adapter's Pi extension asks
// the host before running a built-in tool the permission mode gates.
func (h *Harness) SupportsBuiltinToolApprovals() bool {
	return true
}

// SupportsBuiltinToolFiltering reports true: inactive built-in tools are
// never registered with Pi.
func (h *Harness) SupportsBuiltinToolFiltering() bool {
	return true
}

// LifecycleStateSchema returns the JSON Schema of the adapter's lifecycle
// state data.
func (h *Harness) LifecycleStateSchema() json.RawMessage {
	return json.RawMessage(lifecycleStateSchema)
}

// Bootstrap returns the recipe that installs the pinned Pi coding agent, the
// RPC relay and the adapter's Pi extension into the sandbox.
func (h *Harness) Bootstrap(_ context.Context) (*harness.Bootstrap, error) {
	return bootstrapRecipe(), nil
}

// reattachLiveRuntime reports whether lifecycle methods keep Pi running.
func (h *Harness) reattachLiveRuntime() bool {
	return h.settings.ReattachLiveRuntime == nil || *h.settings.ReattachLiveRuntime
}

// startupTimeout returns the configured startup timeout or the default.
func (h *Harness) startupTimeout() time.Duration {
	if h.settings.StartupTimeout > 0 {
		return h.settings.StartupTimeout
	}

	return defaultStartupTimeout
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
