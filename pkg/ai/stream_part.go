package ai

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// StreamPart is one event of a StreamText stream. A stream starts with
// StartPart and, unless it fails or is aborted, ends with FinishPart. Each
// step is framed by StartStepPart and FinishStepPart. Text, reasoning and
// tool input arrive as start, delta and end parts sharing an ID.
//
// Complete items reuse the content types: CustomContent, SourceContent,
// FileContent, ReasoningFileContent, ToolCall, ToolResult (including
// preliminary results), ToolError, ToolApprovalRequest and
// ToolApprovalResponse. The other parts are StartPart, StartStepPart,
// TextStartPart, TextDeltaPart, TextEndPart, ReasoningStartPart,
// ReasoningDeltaPart, ReasoningEndPart, ToolInputStartPart,
// ToolInputDeltaPart, ToolInputEndPart, ToolOutputDeniedPart,
// FinishStepPart, FinishPart, AbortPart, ErrorPart and RawPart.
type StreamPart interface {
	isStreamPart()
}

// StartPart opens the stream.
type StartPart struct{}

// StartStepPart opens a step.
type StartStepPart struct {
	Request  StepRequest
	Warnings []model.Warning
}

// TextStartPart opens a text block.
type TextStartPart struct {
	ID               string
	ProviderMetadata model.ProviderMetadata
}

// TextDeltaPart appends to a text block.
type TextDeltaPart struct {
	ID               string
	Text             string
	ProviderMetadata model.ProviderMetadata
}

// TextEndPart closes a text block.
type TextEndPart struct {
	ID               string
	Citations        []model.Citation
	ProviderMetadata model.ProviderMetadata
}

// ReasoningStartPart opens a reasoning block.
type ReasoningStartPart struct {
	ID               string
	ProviderMetadata model.ProviderMetadata
}

// ReasoningDeltaPart appends to a reasoning block.
type ReasoningDeltaPart struct {
	ID               string
	Text             string
	ProviderMetadata model.ProviderMetadata
}

// ReasoningEndPart closes a reasoning block.
type ReasoningEndPart struct {
	ID               string
	ProviderMetadata model.ProviderMetadata
}

// ToolInputStartPart opens the streamed input of a tool call. ID is the
// tool call ID.
type ToolInputStartPart struct {
	ID       string
	ToolName string

	// InputEncoding is empty for JSON.
	InputEncoding model.ToolInputEncoding

	ProviderExecuted bool
	Dynamic          bool
	Title            string
	ProviderMetadata model.ProviderMetadata
	ToolMetadata     map[string]any
}

// ToolInputDeltaPart appends to a tool call's input.
type ToolInputDeltaPart struct {
	ID               string
	Delta            string
	ProviderMetadata model.ProviderMetadata
}

// ToolInputEndPart closes a tool call's input. A ToolCall with the same ID
// follows.
type ToolInputEndPart struct {
	ID               string
	ProviderMetadata model.ProviderMetadata
}

// ToolOutputDeniedPart reports that a tool call was denied and will not
// run.
type ToolOutputDeniedPart struct {
	ToolCallID       string
	ToolName         string
	ProviderExecuted bool
}

// FinishStepPart closes a step. Tool results of the step precede it.
type FinishStepPart struct {
	// Response describes the model response. Its body is not included.
	Response model.ResponseInfo

	Usage           LanguageModelUsage
	Performance     StepPerformance
	FinishReason    model.UnifiedFinishReason
	RawFinishReason string

	ProviderMetadata model.ProviderMetadata
}

// FinishPart ends the stream.
type FinishPart struct {
	FinishReason    model.UnifiedFinishReason
	RawFinishReason string

	// TotalUsage is the summed usage of every step.
	TotalUsage LanguageModelUsage
}

// AbortPart reports that the stream was cancelled through its context or
// Close. It is the last part.
type AbortPart struct {
	// Reason describes the cancellation cause.
	Reason string
}

// ErrorPart reports a failure. Errors inside a step, such as a provider
// error event, do not end the stream by themselves.
type ErrorPart struct {
	Err error

	// callbacksHandled reports that OnChunk and OnError already received the
	// error while the stream retry decision was made.
	callbacksHandled bool
}

// RawPart carries a native provider event. It is only emitted when
// StreamTextInclude.RawChunks is set.
type RawPart struct {
	Value json.RawMessage
}

// streamRetryBoundaryPart separates a failed model call attempt from its
// retry. It resets the recorded content of the step and restarts the stream
// transforms; consumers never receive it.
type streamRetryBoundaryPart struct {
	request  StepRequest
	warnings []model.Warning
}

func (StartPart) isStreamPart()            {}
func (StartStepPart) isStreamPart()        {}
func (TextStartPart) isStreamPart()        {}
func (TextDeltaPart) isStreamPart()        {}
func (TextEndPart) isStreamPart()          {}
func (ReasoningStartPart) isStreamPart()   {}
func (ReasoningDeltaPart) isStreamPart()   {}
func (ReasoningEndPart) isStreamPart()     {}
func (ToolInputStartPart) isStreamPart()   {}
func (ToolInputDeltaPart) isStreamPart()   {}
func (ToolInputEndPart) isStreamPart()     {}
func (ToolOutputDeniedPart) isStreamPart() {}
func (FinishStepPart) isStreamPart()       {}
func (FinishPart) isStreamPart()           {}
func (AbortPart) isStreamPart()            {}
func (ErrorPart) isStreamPart()            {}
func (RawPart) isStreamPart()              {}

func (streamRetryBoundaryPart) isStreamPart() {}

func (CustomContent) isStreamPart()        {}
func (SourceContent) isStreamPart()        {}
func (FileContent) isStreamPart()          {}
func (ReasoningFileContent) isStreamPart() {}
func (ToolCall) isStreamPart()             {}
func (ToolResult) isStreamPart()           {}
func (ToolError) isStreamPart()            {}
func (ToolApprovalRequest) isStreamPart()  {}
func (ToolApprovalResponse) isStreamPart() {}
