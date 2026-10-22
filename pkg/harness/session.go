package harness

import (
	"context"

	"github.com/aarock1234/ai/pkg/model"
)

// Session is an active harness session returned by Harness.Start: one
// sandbox, one conversation history and one running agent runtime. The
// agent drives prompt turns against it and ends the local instance with
// Detach, Stop or Destroy, after which no other method may be called.
type Session interface {
	// SessionID returns StartOptions.SessionID.
	SessionID() string

	// IsResume reports whether the session was started from ResumeFrom or
	// ContinueFrom.
	IsResume() bool

	// PromptTurn starts one prompt turn and returns its control handle. Ctx
	// governs the whole turn: when it is cancelled the adapter must cancel the
	// underlying work and end the turn, reporting the failure through the
	// control's Err.
	PromptTurn(ctx context.Context, options PromptTurnOptions) (PromptControl, error)

	// ContinueTurn continues the in-flight turn without a new prompt, after
	// the session was started from ContinueFrom. Adapters whose turn is still
	// live attach to it without re-driving it; others re-drive the runtime's
	// thread from its persisted state. Ctx governs the continued turn like
	// PromptTurn's.
	ContinueTurn(ctx context.Context, options ContinueTurnOptions) (PromptControl, error)

	// Compact asks the runtime to compact its context. The runtime owns the
	// compaction and reports it as a CompactionPart on the active or next
	// turn. Adapters without manual compaction return
	// *CapabilityUnsupportedError. Empty customInstructions use the runtime
	// default.
	Compact(ctx context.Context, customInstructions string) error

	// SuspendTurn freezes the active turn at a precise cursor while keeping
	// the runtime alive, and returns the state to continue it later. The
	// active turn's control then ends cleanly, without an error, once the
	// buffered events were delivered.
	SuspendTurn(ctx context.Context) (ContinueTurnState, error)

	// Detach disconnects from the runtime without tearing it down and returns
	// the state to resume the session before a later turn. The sandbox keeps
	// running.
	Detach(ctx context.Context) (ResumeSessionState, error)

	// Stop persists the state to resume the session later, then stops the
	// runtime.
	Stop(ctx context.Context) (ResumeSessionState, error)

	// Destroy stops the runtime without returning lifecycle state.
	Destroy(ctx context.Context) error
}

// HistoryReader is implemented by sessions that can read the conversation
// history the runtime persisted, including exchanges outside the agent, such
// as the same conversation continued in the runtime's own CLI. Sessions that
// cannot reach the store from the current environment return
// *HistoryUnavailableError; an empty conversation is not an error.
type HistoryReader interface {
	ReadHistory(ctx context.Context, options ReadHistoryOptions) (ReadHistoryResult, error)
}

// ReadHistoryOptions configure HistoryReader.ReadHistory.
type ReadHistoryOptions struct {
	// Since is the Cursor of a previous read. Only later messages are
	// returned. Empty reads the whole history.
	Since string
}

// ReadHistoryResult is the outcome of HistoryReader.ReadHistory.
type ReadHistoryResult struct {
	Messages []Message

	// Cursor is an adapter-owned position to pass as Since.
	Cursor string
}

// TurnSettings are the agent-owned settings captured when a turn begins. The
// same settings are passed to fresh and continued turns and persisted with
// unfinished-turn state, so that a resumed continuation cannot pick up the
// configuration of a later turn.
type TurnSettings struct {
	// Model selects the runtime's model for the turn. Empty uses the runtime
	// default.
	Model string `json:"model,omitempty"`

	// Skills are made available to the runtime for the turn. Adapters replace
	// the skills of the preceding turn before a fresh turn.
	Skills []Skill `json:"skills"`

	// Instructions are applied through the runtime's system or developer
	// instructions when supported.
	Instructions string `json:"instructions,omitempty"`

	// Tools are the application tools of the turn. The adapter emits a
	// ToolCallPart when the runtime calls one and waits for
	// PromptControl.SubmitToolResult.
	Tools []ToolSpec `json:"tools"`
}

// PromptTurnOptions configure Session.PromptTurn.
type PromptTurnOptions struct {
	TurnSettings

	// Prompt is the fresh user input of the turn. The session owns its
	// conversation history, so earlier turns are never replayed. A plain
	// string prompt is a message with a single text part.
	Prompt model.UserMessage

	// ResponseFormat requests a response format for the turn. Nil requests
	// text. Adapters that cannot honor a JSON format return
	// *CapabilityUnsupportedError.
	ResponseFormat *model.ResponseFormat

	// Emit receives each event of the turn, in order. It does not block and
	// may be called from any goroutine, but not after the control's Done
	// channel is closed.
	Emit func(part StreamPart)
}

// ContinueTurnOptions configure Session.ContinueTurn.
type ContinueTurnOptions struct {
	TurnSettings

	// ResponseFormat is the response format of the in-flight turn, for
	// adapters that re-drive the turn.
	ResponseFormat *model.ResponseFormat

	// Emit receives each event of the continued turn, as for
	// PromptTurnOptions.
	Emit func(part StreamPart)
}

// PromptControl is the control surface of a running turn. The agent uses it
// to feed results back into the turn. Implementations must be safe for
// concurrent use.
type PromptControl interface {
	// SubmitToolResult provides the result of an application tool call the
	// adapter emitted.
	SubmitToolResult(ctx context.Context, submission ToolResultSubmission) error

	// Done is closed when the adapter has emitted every event of the turn,
	// whether it succeeded or failed.
	Done() <-chan struct{}

	// Err returns the failure of the turn after Done is closed, and nil when
	// it succeeded or was suspended.
	Err() error
}

// ToolApprovalSubmitter is implemented by controls that accept answers to
// the ToolApprovalRequestPart events of built-in tools.
type ToolApprovalSubmitter interface {
	SubmitToolApproval(ctx context.Context, submission ToolApprovalSubmission) error
}

// UserMessageSubmitter is implemented by controls whose runtime accepts
// interactive input while a turn runs.
type UserMessageSubmitter interface {
	// SubmitUserMessage returns once the runtime has accepted the message
	// for its next safe input boundary.
	SubmitUserMessage(ctx context.Context, text string) error
}

// ToolResultSubmission is the result of one application tool call.
type ToolResultSubmission struct {
	ToolCallID string

	// Output is the raw execution result, as reported to consumers. It must
	// encode to JSON.
	Output any

	IsError bool

	// ToolResult is the model-facing representation of the result. When set,
	// its output takes precedence over Output when the adapter builds the
	// runtime's tool response.
	ToolResult *model.ToolResultPart
}

// ToolApprovalSubmission answers one built-in tool approval request.
type ToolApprovalSubmission struct {
	ApprovalID string
	Approved   bool

	// Reason optionally explains the decision.
	Reason string
}
