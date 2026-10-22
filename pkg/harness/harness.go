// Package harness ports the experimental upstream @ai-sdk/harness package:
// the harness specification (version 1), the Agent that drives a
// third-party coding-agent runtime such as Claude Code or Codex through a
// harness adapter, and the helpers adapters share.
//
// The specification is the Harness interface and the types it uses: an
// adapter starts Sessions, each Session runs prompt turns that emit
// StreamPart events, and a PromptControl feeds tool results, approvals and
// user messages back into a running turn. Optional capabilities are
// optional interfaces, following the pkg/model convention: a Harness may
// implement Bootstrapper, LifecycleStateSchemaProvider,
// BuiltinToolApprovalSupporter and BuiltinToolFilteringSupporter; a Session
// may implement HistoryReader; a PromptControl may implement
// ToolApprovalSubmitter and UserMessageSubmitter.
//
// Harnesses run inside a sandbox. NetworkSession extends sandbox.Session
// with ports, lifecycle and network policy; sandbox/vercel and sandbox/local
// implement it.
//
// Go adaptations, applied throughout: context.Context replaces abortSignal
// options; upstream tool context and runtime context map to context values;
// tools are []ai.Tool in declaration order instead of keyed tool sets;
// optional strings and slices use their zero values for unset; and
// asynchronous promises become blocking calls or Done channels.
package harness

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// SpecificationVersion is the harness specification version this package
// implements. Lifecycle states carry it.
const SpecificationVersion = "harness-v1"

// Harness is a harness adapter: the integration point for one third-party
// coding-agent runtime. It is the Go form of upstream HarnessV1.
//
// A Harness is a stable, stateless description constructed once; Start
// creates a session. Adapters that cannot satisfy a request return
// *CapabilityUnsupportedError from the method that needs the capability.
type Harness interface {
	// HarnessID is the stable identifier of the adapter, used as the key of
	// Metadata values. Conventionally a kebab-case slug such as
	// "claude-code" or "codex".
	HarnessID() string

	// BuiltinTools are the tools the runtime exposes natively, named as the
	// adapter emits them on tool call events (the common name when set, the
	// native name otherwise). The agent merges them with the application's
	// tools when validating tool calls.
	BuiltinTools() []BuiltinTool

	// Start starts a fresh session, resumes a parked session from
	// StartOptions.ResumeFrom, or reconnects to a suspended turn from
	// StartOptions.ContinueFrom. Ctx bounds the startup only: cancelling it
	// must stop spawned processes and network calls of the startup, but the
	// returned session outlives it.
	Start(ctx context.Context, options StartOptions) (Session, error)
}

// Bootstrapper is implemented by harnesses that install files and commands
// into the sandbox before their runtime starts. The agent hashes the recipe
// into a stable identity for snapshot-based reuse and applies it
// idempotently.
type Bootstrapper interface {
	// Bootstrap returns the recipe. A nil recipe means no bootstrap.
	Bootstrap(ctx context.Context) (*Bootstrap, error)
}

// LifecycleStateSchemaProvider is implemented by harnesses that validate the
// adapter-defined Data of their lifecycle states. A state whose data
// validates against the schema can be passed back to Start.
type LifecycleStateSchemaProvider interface {
	// LifecycleStateSchema returns the JSON Schema of lifecycle state data.
	LifecycleStateSchema() json.RawMessage
}

// BuiltinToolApprovalSupporter is implemented by harnesses that can emit
// approval requests for built-in tools when the permission mode is not
// PermissionAllowAll. Approvals of application tools are handled by the
// agent and need no adapter support.
type BuiltinToolApprovalSupporter interface {
	SupportsBuiltinToolApprovals() bool
}

// BuiltinToolFilteringSupporter is implemented by harnesses that can hide
// inactive built-in tools from their runtime. Harnesses without native
// filtering still support active and inactive built-in tools when they
// support built-in tool approvals: the agent then denies calls of inactive
// tools through the approval path.
type BuiltinToolFilteringSupporter interface {
	SupportsBuiltinToolFiltering() bool
}

// StartOptions configure Harness.Start. SandboxSession and SessionWorkDir
// are always set: the agent prepares the sandbox and the session's working
// directory before it starts the adapter.
type StartOptions struct {
	// SessionID identifies the harness session. Adapters use it to name the
	// underlying resources, such as a native session ID.
	SessionID string

	// Headers are additional HTTP headers to send with model requests.
	Headers http.Header

	// ResumeFrom resumes a session parked by Session.Detach or Session.Stop.
	ResumeFrom *ResumeSessionState

	// ContinueFrom reconnects to a turn suspended by Session.SuspendTurn, or
	// the unfinished turn nested in ResumeFrom, so that the session is ready
	// for Session.ContinueTurn instead of a fresh prompt.
	ContinueFrom *ContinueTurnState

	// PermissionMode is the approval policy of built-in tools. Application
	// tools are approved by the agent before their results are submitted.
	PermissionMode PermissionMode

	// BuiltinToolFiltering selects the built-in tools of the session. Nil
	// keeps every built-in tool. Application tools are filtered by the agent
	// before they reach the adapter.
	BuiltinToolFiltering *BuiltinToolFiltering

	// Observability wires diagnostics. It is nil when the application has not
	// enabled them.
	Observability *Observability

	// SandboxSession is the sandbox the adapter operates in. It implements
	// NetworkSession when the sandbox supports ports and lifecycle control;
	// a caller-provided basic sandbox.Session exposes only files and
	// processes. Adapters must not stop or destroy the sandbox.
	SandboxSession sandbox.Session

	// SessionWorkDir is the absolute directory the adapter runs the agent in.
	// It lies under the sandbox's default working directory and exists before
	// Start is called.
	SessionWorkDir string
}

// Bootstrap is an adapter-owned bootstrap recipe: the files and commands its
// runtime needs inside any sandbox.
type Bootstrap struct {
	// HarnessID identifies the adapter owning the recipe. It contributes to
	// the recipe hash.
	HarnessID string

	// BootstrapDir is where the recipe writes its state. Relative paths are
	// resolved against the harness state directory
	// ($HOME/.ai-sdk-harness). The marker file lives directly under it, and
	// Files should use it as their prefix so that an adapter upgrade can
	// clear stale state.
	BootstrapDir string

	// Files are written before any command runs.
	Files []BootstrapFile

	// Commands run sequentially from BootstrapDir after the files are
	// written. A non-zero exit aborts the bootstrap.
	Commands []BootstrapCommand
}

// BootstrapFile is one file of a bootstrap recipe. Relative paths are
// resolved against the harness state directory.
type BootstrapFile struct {
	Path    string
	Content string
}

// BootstrapCommand is one command of a bootstrap recipe.
type BootstrapCommand struct {
	Command string `json:"command"`
}

// PermissionMode is the baseline approval policy of built-in tools.
// Application tools use AgentSettings.ToolApproval instead.
type PermissionMode string

// Permission modes. The empty mode is PermissionAllowAll.
const (
	PermissionAllowReads PermissionMode = "allow-reads"
	PermissionAllowEdits PermissionMode = "allow-edits"
	PermissionAllowAll   PermissionMode = "allow-all"
)
