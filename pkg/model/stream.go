package model

import "encoding/json"

// StreamPart is one event of a streaming call. Text, reasoning and tool input
// arrive as start/delta/end sequences sharing an ID; complete items reuse the
// Content types. A stream begins with StreamStart and ends with Finish.
type StreamPart interface {
	isStreamPart()
}

// StreamStart opens the stream and reports call warnings.
type StreamStart struct {
	Warnings []Warning
}

// TextStart opens a text block.
type TextStart struct {
	ID               string
	ProviderMetadata ProviderMetadata
}

// TextDelta appends to a text block.
type TextDelta struct {
	ID               string
	Delta            string
	ProviderMetadata ProviderMetadata
}

// TextEnd closes a text block.
type TextEnd struct {
	ID               string
	Citations        []Citation
	ProviderMetadata ProviderMetadata
}

// ReasoningStart opens a reasoning block.
type ReasoningStart struct {
	ID               string
	ProviderMetadata ProviderMetadata
}

// ReasoningDelta appends to a reasoning block.
type ReasoningDelta struct {
	ID               string
	Delta            string
	ProviderMetadata ProviderMetadata
}

// ReasoningEnd closes a reasoning block.
type ReasoningEnd struct {
	ID               string
	ProviderMetadata ProviderMetadata
}

// ToolInputEncoding describes how tool input deltas are encoded.
type ToolInputEncoding string

// Tool input encodings.
const (
	ToolInputJSON ToolInputEncoding = "json"
	ToolInputText ToolInputEncoding = "text"
)

// ToolInputStart opens the input of a tool call. ID is the tool call ID.
type ToolInputStart struct {
	ID       string
	ToolName string

	// InputEncoding is empty for JSON.
	InputEncoding ToolInputEncoding

	ProviderExecuted bool
	Dynamic          bool

	// Title optionally describes the call for display.
	Title string

	ProviderMetadata ProviderMetadata
}

// ToolInputDelta appends to a tool call's input.
type ToolInputDelta struct {
	ID               string
	Delta            string
	ProviderMetadata ProviderMetadata
}

// ToolInputEnd closes a tool call's input. A ToolCall with the same ID
// follows.
type ToolInputEnd struct {
	ID               string
	ProviderMetadata ProviderMetadata
}

// Finish ends the stream with usage and the finish reason.
type Finish struct {
	Usage            Usage
	FinishReason     FinishReason
	ProviderMetadata ProviderMetadata
}

// Raw carries one native provider event. It is emitted only when the call
// options request raw chunks.
type Raw struct {
	Value json.RawMessage
}

// StreamError reports a failure inside the stream.
type StreamError struct {
	Err error
}

func (StreamStart) isStreamPart()         {}
func (TextStart) isStreamPart()           {}
func (TextDelta) isStreamPart()           {}
func (TextEnd) isStreamPart()             {}
func (ReasoningStart) isStreamPart()      {}
func (ReasoningDelta) isStreamPart()      {}
func (ReasoningEnd) isStreamPart()        {}
func (ToolInputStart) isStreamPart()      {}
func (ToolInputDelta) isStreamPart()      {}
func (ToolInputEnd) isStreamPart()        {}
func (ToolApprovalRequest) isStreamPart() {}
func (ToolCall) isStreamPart()            {}
func (ToolResult) isStreamPart()          {}
func (Custom) isStreamPart()              {}
func (File) isStreamPart()                {}
func (ReasoningFile) isStreamPart()       {}
func (Source) isStreamPart()              {}
func (ResponseMetadata) isStreamPart()    {}
func (Finish) isStreamPart()              {}
func (Raw) isStreamPart()                 {}
func (StreamError) isStreamPart()         {}
