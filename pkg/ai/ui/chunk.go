package ui

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// Chunk is one event of a UI message stream, the upstream UIMessageChunk.
// Each chunk type encodes with its upstream "type" discriminator:
//
//   - text-start, text-delta, text-end: TextStartChunk, TextDeltaChunk,
//     TextEndChunk
//   - reasoning-start, reasoning-delta, reasoning-end: ReasoningStartChunk,
//     ReasoningDeltaChunk, ReasoningEndChunk
//   - custom, error, file, reasoning-file, source-url, source-document:
//     CustomChunk, ErrorChunk, FileChunk, ReasoningFileChunk, SourceURLChunk,
//     SourceDocumentChunk
//   - tool-input-start, tool-input-delta, tool-input-available,
//     tool-input-error: ToolInputStartChunk, ToolInputDeltaChunk,
//     ToolInputAvailableChunk, ToolInputErrorChunk
//   - tool-approval-request, tool-approval-response:
//     ToolApprovalRequestChunk, ToolApprovalResponseChunk
//   - tool-output-available, tool-output-error, tool-output-denied:
//     ToolOutputAvailableChunk, ToolOutputErrorChunk, ToolOutputDeniedChunk
//   - data-*: DataChunk
//   - start-step, finish-step, reset-step: StartStepChunk, FinishStepChunk,
//     ResetStepChunk
//   - start, finish, abort, message-metadata: StartChunk, FinishChunk,
//     AbortChunk, MessageMetadataChunk
type Chunk interface {
	isChunk()
}

// TextStartChunk opens a text part.
type TextStartChunk struct {
	ID               string                 `json:"id"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// TextDeltaChunk appends to a text part.
type TextDeltaChunk struct {
	ID               string                 `json:"id"`
	Delta            string                 `json:"delta"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// TextEndChunk closes a text part.
type TextEndChunk struct {
	ID               string                 `json:"id"`
	Citations        []Citation             `json:"citations,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ReasoningStartChunk opens a reasoning part.
type ReasoningStartChunk struct {
	ID               string                 `json:"id"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ReasoningDeltaChunk appends to a reasoning part.
type ReasoningDeltaChunk struct {
	ID               string                 `json:"id"`
	Delta            string                 `json:"delta"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ReasoningEndChunk closes a reasoning part.
type ReasoningEndChunk struct {
	ID               string                 `json:"id"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// CustomChunk adds provider-specific content. Kind has the form
// "provider.type".
type CustomChunk struct {
	Kind             string                 `json:"kind"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ErrorChunk reports an error to the client.
type ErrorChunk struct {
	ErrorText string `json:"errorText"`
}

// ToolInputStartChunk opens the streamed input of a tool call.
type ToolInputStartChunk struct {
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`

	// InputEncoding is the encoding of the input deltas. Empty means JSON.
	InputEncoding model.ToolInputEncoding `json:"inputEncoding,omitempty"`

	ProviderExecuted bool                   `json:"providerExecuted,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
	ToolMetadata     map[string]any         `json:"toolMetadata,omitempty"`
	Dynamic          bool                   `json:"dynamic,omitempty"`
	Title            string                 `json:"title,omitempty"`
}

// ToolInputDeltaChunk appends to the input of a tool call.
type ToolInputDeltaChunk struct {
	ToolCallID     string `json:"toolCallId"`
	InputTextDelta string `json:"inputTextDelta"`
}

// ToolInputAvailableChunk carries the complete input of a tool call.
type ToolInputAvailableChunk struct {
	ToolCallID       string                 `json:"toolCallId"`
	ToolName         string                 `json:"toolName"`
	Input            json.RawMessage        `json:"input"`
	ProviderExecuted bool                   `json:"providerExecuted,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
	ToolMetadata     map[string]any         `json:"toolMetadata,omitempty"`
	Dynamic          bool                   `json:"dynamic,omitempty"`
	Title            string                 `json:"title,omitempty"`
}

// ToolInputErrorChunk reports a tool call whose input is invalid.
type ToolInputErrorChunk struct {
	ToolCallID       string                 `json:"toolCallId"`
	ToolName         string                 `json:"toolName"`
	Input            json.RawMessage        `json:"input"`
	ProviderExecuted bool                   `json:"providerExecuted,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
	ToolMetadata     map[string]any         `json:"toolMetadata,omitempty"`
	Dynamic          bool                   `json:"dynamic,omitempty"`
	ErrorText        string                 `json:"errorText"`
	Title            string                 `json:"title,omitempty"`
}

// ToolApprovalRequestChunk asks for approval of a tool call.
type ToolApprovalRequestChunk struct {
	ApprovalID         string          `json:"approvalId"`
	ToolCallID         string          `json:"toolCallId"`
	ApprovalDescriptor json.RawMessage `json:"approvalDescriptor,omitempty"`

	// InputSchemaInput is the validated input before tool input refinement,
	// set only when refinement changed the input.
	InputSchemaInput json.RawMessage `json:"inputSchemaInput,omitempty"`

	Reason      string `json:"reason,omitempty"`
	IsAutomatic bool   `json:"isAutomatic,omitempty"`
	Signature   string `json:"signature,omitempty"`
}

// ToolApprovalResponseChunk carries the decision on an approval request.
type ToolApprovalResponseChunk struct {
	ApprovalID       string                 `json:"approvalId"`
	Approved         bool                   `json:"approved"`
	Reason           string                 `json:"reason,omitempty"`
	ProviderExecuted bool                   `json:"providerExecuted,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ToolOutputAvailableChunk carries the output of a tool call.
type ToolOutputAvailableChunk struct {
	ToolCallID string `json:"toolCallId"`

	// Output is the JSON output. Nil encodes as null.
	Output json.RawMessage `json:"output"`

	ProviderExecuted bool                   `json:"providerExecuted,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
	ToolMetadata     map[string]any         `json:"toolMetadata,omitempty"`
	Dynamic          bool                   `json:"dynamic,omitempty"`
	Preliminary      bool                   `json:"preliminary,omitempty"`
}

// ToolOutputErrorChunk reports a failed tool call.
type ToolOutputErrorChunk struct {
	ToolCallID       string                 `json:"toolCallId"`
	ErrorText        string                 `json:"errorText"`
	ProviderExecuted bool                   `json:"providerExecuted,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
	ToolMetadata     map[string]any         `json:"toolMetadata,omitempty"`
	Dynamic          bool                   `json:"dynamic,omitempty"`
}

// ToolOutputDeniedChunk reports a denied tool call.
type ToolOutputDeniedChunk struct {
	ToolCallID string `json:"toolCallId"`
}

// SourceURLChunk adds a URL source.
type SourceURLChunk struct {
	SourceID         string                 `json:"sourceId"`
	URL              string                 `json:"url"`
	Title            string                 `json:"title,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// SourceDocumentChunk adds a document source.
type SourceDocumentChunk struct {
	SourceID         string                 `json:"sourceId"`
	MediaType        string                 `json:"mediaType"`
	Title            string                 `json:"title"`
	Filename         string                 `json:"filename,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// FileChunk adds a file.
type FileChunk struct {
	URL              string                 `json:"url"`
	MediaType        string                 `json:"mediaType"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ReasoningFileChunk adds a file produced during reasoning.
type ReasoningFileChunk struct {
	URL              string                 `json:"url"`
	MediaType        string                 `json:"mediaType"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// DataChunk adds or replaces a data part, encoded with the type
// "data-" + Name. A chunk whose ID matches an existing data part of the same
// name replaces its data.
type DataChunk struct {
	Name string
	ID   string
	Data json.RawMessage

	// Transient chunks reach the client but are not added to the message.
	Transient bool
}

// StartStepChunk opens a step.
type StartStepChunk struct{}

// FinishStepChunk closes a step.
type FinishStepChunk struct{}

// ResetStepChunk removes the parts added during the current step.
type ResetStepChunk struct{}

// StartChunk opens the response message.
type StartChunk struct {
	MessageID       string          `json:"messageId,omitempty"`
	MessageMetadata json.RawMessage `json:"messageMetadata,omitempty"`
}

// FinishChunk ends the response message.
type FinishChunk struct {
	FinishReason    model.UnifiedFinishReason `json:"finishReason,omitempty"`
	MessageMetadata json.RawMessage           `json:"messageMetadata,omitempty"`
}

// AbortChunk reports that the response was aborted.
type AbortChunk struct {
	Reason string `json:"reason,omitempty"`
}

// MessageMetadataChunk updates the metadata of the response message. Object
// metadata is merged into the existing metadata.
type MessageMetadataChunk struct {
	MessageMetadata json.RawMessage `json:"messageMetadata"`
}

func (TextStartChunk) isChunk()            {}
func (TextDeltaChunk) isChunk()            {}
func (TextEndChunk) isChunk()              {}
func (ReasoningStartChunk) isChunk()       {}
func (ReasoningDeltaChunk) isChunk()       {}
func (ReasoningEndChunk) isChunk()         {}
func (CustomChunk) isChunk()               {}
func (ErrorChunk) isChunk()                {}
func (ToolInputStartChunk) isChunk()       {}
func (ToolInputDeltaChunk) isChunk()       {}
func (ToolInputAvailableChunk) isChunk()   {}
func (ToolInputErrorChunk) isChunk()       {}
func (ToolApprovalRequestChunk) isChunk()  {}
func (ToolApprovalResponseChunk) isChunk() {}
func (ToolOutputAvailableChunk) isChunk()  {}
func (ToolOutputErrorChunk) isChunk()      {}
func (ToolOutputDeniedChunk) isChunk()     {}
func (SourceURLChunk) isChunk()            {}
func (SourceDocumentChunk) isChunk()       {}
func (FileChunk) isChunk()                 {}
func (ReasoningFileChunk) isChunk()        {}
func (DataChunk) isChunk()                 {}
func (StartStepChunk) isChunk()            {}
func (FinishStepChunk) isChunk()           {}
func (ResetStepChunk) isChunk()            {}
func (StartChunk) isChunk()                {}
func (FinishChunk) isChunk()               {}
func (AbortChunk) isChunk()                {}
func (MessageMetadataChunk) isChunk()      {}
