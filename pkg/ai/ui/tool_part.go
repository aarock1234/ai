package ui

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// ToolState is the lifecycle state of a tool part.
type ToolState string

// Tool part states.
const (
	ToolInputStreaming    ToolState = "input-streaming"
	ToolInputAvailable    ToolState = "input-available"
	ToolApprovalRequested ToolState = "approval-requested"
	ToolApprovalResponded ToolState = "approval-responded"
	ToolOutputAvailable   ToolState = "output-available"
	ToolOutputError       ToolState = "output-error"
	ToolOutputDenied      ToolState = "output-denied"
)

// ToolPart is a tool invocation. Static tool parts, whose tool types are
// known at development time, are encoded with the type "tool-" + ToolName;
// dynamic tool parts are encoded with the type "dynamic-tool".
type ToolPart struct {
	ToolName   string
	ToolCallID string

	// Dynamic selects the dynamic tool encoding.
	Dynamic bool

	// Unavailable marks a static tool part that was converted to a dynamic
	// part because its tool is no longer available or its input no longer
	// matches the tool schema. It implies Dynamic.
	Unavailable bool

	State ToolState

	Title        string
	ToolMetadata map[string]any

	// ProviderExecuted marks calls that the provider ran.
	ProviderExecuted bool

	// Input is the tool input. While the input streams it is the partial
	// input parsed so far.
	Input json.RawMessage

	// RawInput is the input text received so far, as a JSON string, while
	// the input streams. Older output-error parts used it for the input.
	RawInput json.RawMessage

	// InputEncoding is the encoding of RawInput. Empty means JSON.
	InputEncoding model.ToolInputEncoding

	// Output is the tool output in the output-available state.
	Output json.RawMessage

	// ErrorText describes the failure in the output-error state.
	ErrorText string

	CallProviderMetadata   model.ProviderMetadata
	ResultProviderMetadata model.ProviderMetadata

	// Preliminary marks an intermediate output that a later output replaces.
	Preliminary bool

	// Approval is the approval request and its response, if any.
	Approval *ToolApproval
}

// ToolApproval is the approval request of a tool call and its response.
type ToolApproval struct {
	ID string `json:"id"`

	// Approved is nil until the approval is responded to.
	Approved *bool `json:"approved,omitempty"`

	Descriptor json.RawMessage `json:"descriptor,omitempty"`

	// RequestReason explains why approval is needed.
	RequestReason string `json:"requestReason,omitempty"`

	// Reason explains the response.
	Reason string `json:"reason,omitempty"`

	IsAutomatic bool   `json:"isAutomatic,omitempty"`
	Signature   string `json:"signature,omitempty"`

	// InputSchemaInput is the validated input before tool input refinement.
	// It is set only when refinement changed the input.
	InputSchemaInput json.RawMessage `json:"inputSchemaInput,omitempty"`
}

// toolPartFields are the encoded fields of a ToolPart besides its type.
type toolPartFields struct {
	ToolName               string                  `json:"toolName,omitempty"`
	Dynamic                *bool                   `json:"dynamic,omitempty"`
	ToolCallID             string                  `json:"toolCallId"`
	Title                  string                  `json:"title,omitempty"`
	ToolMetadata           map[string]any          `json:"toolMetadata,omitempty"`
	ProviderExecuted       bool                    `json:"providerExecuted,omitempty"`
	State                  ToolState               `json:"state"`
	Input                  json.RawMessage         `json:"input,omitempty"`
	RawInput               json.RawMessage         `json:"rawInput,omitempty"`
	InputEncoding          model.ToolInputEncoding `json:"inputEncoding,omitempty"`
	Output                 json.RawMessage         `json:"output,omitempty"`
	ErrorText              *string                 `json:"errorText,omitempty"`
	CallProviderMetadata   model.ProviderMetadata  `json:"callProviderMetadata,omitempty"`
	ResultProviderMetadata model.ProviderMetadata  `json:"resultProviderMetadata,omitempty"`
	Preliminary            bool                    `json:"preliminary,omitempty"`
	Approval               *ToolApproval           `json:"approval,omitempty"`
}

// isDynamicPart reports whether the part uses the dynamic tool encoding.
func (p ToolPart) isDynamicPart() bool {
	return p.Dynamic || p.Unavailable
}

// hasOutputState reports whether the part holds a tool output or failure.
func (p ToolPart) hasOutputState() bool {
	return isOutputState(p.State)
}

func isOutputState(state ToolState) bool {
	return state == ToolOutputAvailable || state == ToolOutputError
}

// MarshalJSON encodes the part with the type "tool-" + ToolName, or
// "dynamic-tool" for dynamic parts. The error text is always encoded in the
// output-error state.
func (p ToolPart) MarshalJSON() ([]byte, error) {
	fields := toolPartFields{
		ToolCallID:             p.ToolCallID,
		Title:                  p.Title,
		ToolMetadata:           p.ToolMetadata,
		ProviderExecuted:       p.ProviderExecuted,
		State:                  p.State,
		Input:                  p.Input,
		RawInput:               p.RawInput,
		InputEncoding:          p.InputEncoding,
		Output:                 p.Output,
		CallProviderMetadata:   p.CallProviderMetadata,
		ResultProviderMetadata: p.ResultProviderMetadata,
		Preliminary:            p.Preliminary,
		Approval:               p.Approval,
	}

	if p.ErrorText != "" || p.State == ToolOutputError {
		fields.ErrorText = &p.ErrorText
	}

	if !p.isDynamicPart() {
		return marshalWithType(toolTypePrefix+p.ToolName, fields)
	}

	fields.ToolName = p.ToolName
	if p.Unavailable {
		fields.Dynamic = new(false)
	}

	return marshalWithType(dynamicToolType, fields)
}

// decodeToolPart decodes a tool part. Static parts take their tool name from
// their type; dynamic parts carry theirs in toolName.
func decodeToolPart(data []byte, staticName string, dynamic bool) (Part, error) {
	var fields toolPartFields
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}

	part := ToolPart{
		ToolName:               staticName,
		ToolCallID:             fields.ToolCallID,
		State:                  fields.State,
		Title:                  fields.Title,
		ToolMetadata:           fields.ToolMetadata,
		ProviderExecuted:       fields.ProviderExecuted,
		Input:                  fields.Input,
		RawInput:               fields.RawInput,
		InputEncoding:          fields.InputEncoding,
		Output:                 fields.Output,
		CallProviderMetadata:   fields.CallProviderMetadata,
		ResultProviderMetadata: fields.ResultProviderMetadata,
		Preliminary:            fields.Preliminary,
		Approval:               fields.Approval,
	}

	if fields.ErrorText != nil {
		part.ErrorText = *fields.ErrorText
	}

	if dynamic {
		part.ToolName = fields.ToolName
		part.Dynamic = true
		part.Unavailable = fields.Dynamic != nil && !*fields.Dynamic
	}

	return part, nil
}
