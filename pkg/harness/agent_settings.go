package harness

import (
	"context"
	"net/http"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// AgentSettings configure an Agent. They are the Go form of upstream
// HarnessAgentSettings.
//
// Go adaptations: upstream toolsContext, runtimeContext and the tool
// contextSchema map to context.Context values of the calls; the deprecated
// sandbox provider, onSandboxSession and callOptionsSchema settings are not
// ported (pass a sandbox session to CreateSession, use
// SandboxConfig.OnSession, and type-assert CallOptions.Options in
// PrepareCall).
type AgentSettings struct {
	// Harness is the adapter driving the runtime. It is required.
	Harness Harness

	// ID identifies the agent. It may be empty.
	ID string

	// Model selects the runtime's model. Supported values are defined by the
	// harness. PrepareCall can replace it between completed turns.
	Model string

	// Tools are application tools offered to the runtime in addition to the
	// harness's built-in tools, in declaration order. When the runtime calls
	// one, the agent runs its Execute function on the host and submits the
	// result. An application tool overrides a built-in tool of the same name.
	Tools []ai.Tool

	// Skills are made available to the runtime. PrepareCall can replace them.
	Skills []Skill

	// Instructions are appended to the runtime's system or developer prompt
	// when supported, and prepended to the user message otherwise.
	// PrepareCall can replace them.
	Instructions string

	// Headers are sent with every model request. The harness manages
	// Authorization, X-Api-Key, User-Agent and X-Client-App, which are not
	// allowed.
	Headers http.Header

	// PrepareCall prepares the prompt and the settings that may vary between
	// completed turns. The prepared values are frozen for the lifetime of the
	// turn, including continuations of a suspended turn.
	PrepareCall PrepareCallFunc

	// Output requires typed output on every turn. Nil generates text.
	Output ai.Output

	// StopWhen stops the current result after a completed step that could
	// continue into another model step. The runtime's turn stays unfinished
	// and can be continued. A terminal text-only step finishes naturally. Nil
	// runs until the turn finishes or pauses.
	StopWhen []ai.StopCondition

	// ActiveTools limits the tools the harness may call, application and
	// built-in, without changing the declared tools. Set at most one of
	// ActiveTools and InactiveTools.
	ActiveTools []string

	// InactiveTools excludes tools from the set the harness may call.
	InactiveTools []string

	// PermissionMode is the approval policy of built-in tools. Empty selects
	// PermissionAllowAll. Other modes require a harness that supports
	// built-in tool approvals.
	PermissionMode PermissionMode

	// ToolApproval maps application tool names to approval decisions:
	// not-applicable and approved run the tool, user-approval pauses the turn
	// for a decision, and denied submits an execution denial. Unlisted tools
	// run without approval.
	ToolApproval map[string]ai.ToolApprovalStatus

	// SandboxConfig configures working directories and sandbox hooks.
	SandboxConfig SandboxConfig

	// Telemetry receives the lifecycle events of each turn with the span
	// shape of StreamText.
	Telemetry ai.TelemetryOptions

	// Debug enables diagnostics: bridge log forwarding and the stderr
	// default. Nil falls back to the HARNESS_DEBUG environment variables.
	Debug *DebugConfig

	// OnLog receives every forwarded diagnostic.
	OnLog func(diagnostic Diagnostic)

	// The callbacks have the meaning of the ai.StreamTextOptions fields of
	// the same names.
	OnStart                  func(ctx context.Context, event ai.GenerateTextStartEvent)
	OnStepStart              func(ctx context.Context, event ai.GenerateTextStepStartEvent)
	OnLanguageModelCallStart func(ctx context.Context, event ai.LanguageModelCallStartEvent)
	OnLanguageModelCallEnd   func(ctx context.Context, event ai.LanguageModelCallEndEvent)
	OnToolExecutionStart     func(ctx context.Context, event ai.ToolExecutionStartEvent)
	OnToolExecutionEnd       func(ctx context.Context, event ai.ToolExecutionEndEvent)
	OnStepEnd                func(ctx context.Context, step ai.StepResult)
	OnEnd                    func(ctx context.Context, event ai.GenerateTextEndEvent)
}

// SandboxConfig configures the sandbox of an agent's sessions.
type SandboxConfig struct {
	// WorkDir is a fixed working directory for every session, relative to the
	// sandbox's default working directory; "." uses the default directory
	// itself. Empty gives each session its own "<harnessId>-<sessionId>"
	// directory.
	WorkDir string

	// BootstrapHash identifies the side effects of OnBootstrap. Change it
	// whenever they should invalidate reusable sandbox snapshots. It must be
	// set together with OnBootstrap.
	BootstrapHash string

	// OnBootstrap runs during sandbox template preparation, after the
	// harness's own bootstrap and before snapshot-capable providers publish a
	// snapshot.
	OnBootstrap func(ctx context.Context, event SandboxBootstrapEvent) error

	// OnSession runs after each sandbox session is acquired and its working
	// directory exists, before the harness starts, for fresh and resumed
	// sessions. Keep it idempotent when sessions may resume.
	OnSession func(ctx context.Context, event SandboxSessionEvent) error
}

// SandboxBootstrapEvent describes the sandbox being prepared by
// SandboxConfig.OnBootstrap.
type SandboxBootstrapEvent struct {
	Session sandbox.Session

	// WorkDir is the absolute working directory of the bootstrap.
	WorkDir string
}

// SandboxSessionEvent describes the sandbox of a session for
// SandboxConfig.OnSession.
type SandboxSessionEvent struct {
	Session        sandbox.Session
	SessionWorkDir string
}

// PrepareCallFunc prepares a turn. It receives the agent's settings and the
// call's prompt and returns the values to use; fields left empty in the
// result stay empty, so copy the values to keep.
type PrepareCallFunc func(ctx context.Context, call PreparedCall) (PreparedCall, error)

// PreparedCall is the prompt and the turn-scoped settings of a call.
type PreparedCall struct {
	// Options are the application-defined CallOptions.Options.
	Options any

	// Prompt and Messages are the call's input. The harness owns the
	// conversation history, so only the last user message of Messages is
	// sent; a prepared call must not be a tool continuation.
	Prompt   string
	Messages []model.Message

	Model        string
	Skills       []Skill
	Instructions string

	// Tools are the application tools of the turn.
	Tools []ai.Tool
}

// CallOptions configure Agent.Generate and Agent.Stream. Set either Prompt
// or Messages. A trailing tool message with approval responses or tool
// results continues the session's unfinished turn instead of starting one.
// The callbacks run after the agent's callbacks of the same kind.
type CallOptions struct {
	// Session is the session the turn runs in. It is required.
	Session *AgentSession

	Prompt   string
	Messages []model.Message

	// Options are application-defined call options for PrepareCall.
	Options any

	OnStart              func(ctx context.Context, event ai.GenerateTextStartEvent)
	OnStepStart          func(ctx context.Context, event ai.GenerateTextStepStartEvent)
	OnToolExecutionStart func(ctx context.Context, event ai.ToolExecutionStartEvent)
	OnToolExecutionEnd   func(ctx context.Context, event ai.ToolExecutionEndEvent)
	OnStepEnd            func(ctx context.Context, step ai.StepResult)
	OnEnd                func(ctx context.Context, event ai.GenerateTextEndEvent)
}

// ContinueOptions configure Agent.ContinueGenerate and
// Agent.ContinueStream.
type ContinueOptions struct {
	Session *AgentSession

	// ToolApprovals answer pending approval requests of the unfinished turn.
	ToolApprovals []model.ToolApprovalResponsePart

	// ToolResults provide results for pending application tool calls.
	ToolResults []model.ToolResultPart
}

// CreateSessionOptions configure Agent.CreateSession.
type CreateSessionOptions struct {
	// SandboxSession is the sandbox to run the harness in. It is required;
	// pass a NetworkSession for harnesses that need ports. The caller keeps
	// ownership: the agent never stops or destroys it.
	SandboxSession sandbox.Session

	// SessionID identifies the session. Empty generates one. Pass the
	// original ID together with ResumeFrom or ContinueFrom.
	SessionID string

	// ResumeFrom resumes a session from the state returned by
	// AgentSession.Detach or AgentSession.Stop.
	ResumeFrom *ResumeSessionState

	// ContinueFrom resumes from the state returned by
	// AgentSession.SuspendTurn, before ContinueStream or ContinueGenerate.
	ContinueFrom *ContinueTurnState
}
