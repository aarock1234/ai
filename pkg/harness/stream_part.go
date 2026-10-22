package harness

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// Metadata is adapter-namespaced opaque data attached to harness events,
// the harness counterpart of model.ProviderMetadata. Keys are harness IDs;
// values are JSON objects the adapter chooses to surface.
type Metadata map[string]any

// StreamPart is one event an adapter emits during a turn. The variants
// mirror model.StreamPart where they overlap and reuse its tool types, so
// that the agent can forward them with little translation: StreamStartPart,
// TextStartPart, TextDeltaPart, TextEndPart, ReasoningStartPart,
// ReasoningDeltaPart, ReasoningEndPart, ToolInputStartPart,
// ToolInputDeltaPart, ToolInputEndPart, ToolCallPart,
// ToolApprovalRequestPart, ToolResultPart, FinishStepPart, FinishPart,
// FileChangePart, CompactionPart, ErrorPart and RawPart.
type StreamPart interface {
	isStreamPart()
}

// StreamStartPart opens the turn.
type StreamStartPart struct {
	// Warnings are non-fatal issues of the call, such as unsupported
	// settings. Upstream's harness call warnings map to model.Warning: an
	// unsupported setting or tool is model.WarningUnsupported with Setting or
	// Feature, and other warnings are model.WarningOther with Message.
	Warnings []model.Warning

	// ModelID is the model the runtime resolved for the turn, when the
	// adapter learns it at the start.
	ModelID string
}

// TextStartPart opens a text block.
type TextStartPart struct {
	ID              string
	HarnessMetadata Metadata
}

// TextDeltaPart appends to a text block.
type TextDeltaPart struct {
	ID              string
	Delta           string
	HarnessMetadata Metadata
}

// TextEndPart closes a text block.
type TextEndPart struct {
	ID              string
	HarnessMetadata Metadata
}

// ReasoningStartPart opens a reasoning block.
type ReasoningStartPart struct {
	ID              string
	HarnessMetadata Metadata
}

// ReasoningDeltaPart appends to a reasoning block.
type ReasoningDeltaPart struct {
	ID              string
	Delta           string
	HarnessMetadata Metadata
}

// ReasoningEndPart closes a reasoning block.
type ReasoningEndPart struct {
	ID              string
	HarnessMetadata Metadata
}

// ToolInputStartPart opens the streamed input of a tool call.
type ToolInputStartPart struct {
	model.ToolInputStart
}

// ToolInputDeltaPart appends to a tool call's input.
type ToolInputDeltaPart struct {
	model.ToolInputDelta
}

// ToolInputEndPart closes a tool call's input.
type ToolInputEndPart struct {
	model.ToolInputEnd
}

// ToolCallPart is a tool call of the runtime. ProviderExecuted is true for
// built-in tools the runtime runs itself and false for application tools the
// agent must run.
type ToolCallPart struct {
	model.ToolCall

	// NativeName is the runtime's name of a built-in tool when it differs
	// from ToolName, for example "Bash" for "bash".
	NativeName string
}

// ToolApprovalRequestPart asks for the approval of a built-in tool call.
type ToolApprovalRequestPart struct {
	model.ToolApprovalRequest
}

// ToolResultPart is the result of a tool call. Results of application tools
// submitted by the agent are echoed back as tool results too.
type ToolResultPart struct {
	model.ToolResult
}

// FinishStepPart closes a step of a multi-step turn.
//
// Adapters also emit it when the runtime cannot progress without host input:
// every unsettled tool call of the step waits for a tool result, an
// approval or an answer. The runtime's step stays open and another
// FinishStepPart follows once those calls settle; it is only emitted when a
// waiting call was not covered by an earlier one in the same step.
type FinishStepPart struct {
	FinishReason    model.FinishReason
	Usage           model.Usage
	HarnessMetadata Metadata

	// Checkpoint optionally pins the adapter's event cursor at this step
	// boundary. The agent pins it while it evaluates stop conditions, so that
	// suspending at a stop condition resumes right after this event. Bridge
	// adapters set it from the event's sequence number; others leave it nil.
	Checkpoint EventCheckpoint
}

// EventCheckpoint pins an adapter's event cursor at one event, so that a
// later Session.SuspendTurn reports that event as the last delivered one.
type EventCheckpoint interface {
	// Pin pins the cursor and returns the function that releases the pin.
	Pin() (release func())
}

// FinishPart ends the turn.
type FinishPart struct {
	FinishReason    model.FinishReason
	TotalUsage      model.Usage
	HarnessMetadata Metadata
}

// FileChangeEvent is the kind of a workspace file change.
type FileChangeEvent string

// File change events.
const (
	FileCreated  FileChangeEvent = "create"
	FileModified FileChangeEvent = "modify"
	FileDeleted  FileChangeEvent = "delete"
)

// FileChangePart reports a workspace file change made through an opaque
// mechanism with no visible tool call, such as Codex's internal patch tool.
// It carries the path only.
type FileChangePart struct {
	Event           FileChangeEvent
	Path            string
	HarnessMetadata Metadata
}

// CompactionTrigger is what caused a context compaction.
type CompactionTrigger string

// Compaction triggers.
const (
	CompactionManual CompactionTrigger = "manual"
	CompactionAuto   CompactionTrigger = "auto"
)

// CompactionPart reports a context compaction the runtime completed. It is
// an observation only; the runtime owns the compaction.
type CompactionPart struct {
	Trigger CompactionTrigger
	Summary string

	// TokensBefore and TokensAfter are nil when the runtime does not report
	// them.
	TokensBefore *int
	TokensAfter  *int

	HarnessMetadata Metadata
}

// ErrorPart reports a failure. A turn may report several.
type ErrorPart struct {
	Err error
}

// RawPart carries adapter-specific data. The agent forwards it as an
// ai.RawPart.
type RawPart struct {
	Value json.RawMessage
}

func (StreamStartPart) isStreamPart()         {}
func (TextStartPart) isStreamPart()           {}
func (TextDeltaPart) isStreamPart()           {}
func (TextEndPart) isStreamPart()             {}
func (ReasoningStartPart) isStreamPart()      {}
func (ReasoningDeltaPart) isStreamPart()      {}
func (ReasoningEndPart) isStreamPart()        {}
func (ToolInputStartPart) isStreamPart()      {}
func (ToolInputDeltaPart) isStreamPart()      {}
func (ToolInputEndPart) isStreamPart()        {}
func (ToolCallPart) isStreamPart()            {}
func (ToolApprovalRequestPart) isStreamPart() {}
func (ToolResultPart) isStreamPart()          {}
func (FinishStepPart) isStreamPart()          {}
func (FinishPart) isStreamPart()              {}
func (FileChangePart) isStreamPart()          {}
func (CompactionPart) isStreamPart()          {}
func (ErrorPart) isStreamPart()               {}
func (RawPart) isStreamPart()                 {}
