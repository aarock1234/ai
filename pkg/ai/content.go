package ai

import (
	"encoding/json"
	"errors"

	"github.com/aarock1234/ai/pkg/model"
)

// ContentPart is one item of generated content: TextContent, CustomContent,
// ReasoningContent, ReasoningFileContent, SourceContent, FileContent,
// ToolCall, ToolResult, ToolError, ToolApprovalRequest or
// ToolApprovalResponse.
type ContentPart interface {
	isContentPart()
}

// ToolOutput is the outcome of a tool call: a ToolResult or a ToolError.
type ToolOutput interface {
	ContentPart
	StreamPart
	LanguageModelStreamPart

	isToolOutput()
}

// TextContent is generated text.
type TextContent struct {
	Text             string
	Citations        []model.Citation
	ProviderMetadata model.ProviderMetadata
}

// CustomContent is provider-specific content. Kind has the form
// "provider.type"; the content lives in ProviderMetadata.
type CustomContent struct {
	Kind             string
	ProviderMetadata model.ProviderMetadata
}

// ReasoningContent is reasoning text.
type ReasoningContent struct {
	Text             string
	ProviderMetadata model.ProviderMetadata
}

// ReasoningFileContent is a file generated during reasoning.
type ReasoningFileContent struct {
	File             GeneratedFile
	ProviderMetadata model.ProviderMetadata
}

// SourceContent is a source used to generate the content.
type SourceContent struct {
	model.Source
}

// FileContent is a generated file.
type FileContent struct {
	File             GeneratedFile
	ProviderMetadata model.ProviderMetadata
}

// ToolCall is a parsed tool call.
type ToolCall struct {
	ToolCallID string
	ToolName   string

	// Input is the validated JSON input. For invalid calls it is the input
	// as the model produced it: its JSON value when it parses, and otherwise
	// the raw text encoded as a JSON string.
	Input json.RawMessage

	// ProviderExecuted marks calls that the provider runs itself.
	ProviderExecuted bool

	// Dynamic marks calls of dynamic tools and invalid calls, whose types are
	// only known at runtime.
	Dynamic bool

	// Invalid marks calls that could not be parsed or that name an unknown
	// tool. Error holds the reason.
	Invalid bool
	Error   error

	// Title is the display title of the tool.
	Title string

	ProviderMetadata model.ProviderMetadata

	// ToolMetadata is the Metadata of the called tool.
	ToolMetadata map[string]any

	// schemaInput is the validated input before refinement, set when a
	// refinement is configured for the tool. Approval requests record it so
	// that approvals revalidate the input the schema accepted.
	schemaInput json.RawMessage
}

// ToolResult is the successful outcome of a tool call.
type ToolResult struct {
	ToolCallID string
	ToolName   string

	// Input is the input of the call. It is empty for deferred results of
	// provider-executed tools whose call arrived in an earlier response.
	Input json.RawMessage

	// Output is the value returned by the tool's Execute function, or the
	// JSON result of a provider-executed tool.
	Output any

	ProviderExecuted bool
	Dynamic          bool

	// Preliminary marks an intermediate output that a later result replaces.
	Preliminary bool

	Title            string
	ProviderMetadata model.ProviderMetadata
	ToolMetadata     map[string]any
}

// ToolError is the failed outcome of a tool call.
type ToolError struct {
	ToolCallID string
	ToolName   string
	Input      json.RawMessage

	// Error is the failure. Failed provider-executed tools report a
	// *ProviderToolError carrying the provider's JSON result.
	Error error

	ProviderExecuted bool
	Dynamic          bool
	Title            string
	ProviderMetadata model.ProviderMetadata
	ToolMetadata     map[string]any
}

// ToolApprovalRequest asks for approval of a tool call before it runs.
type ToolApprovalRequest struct {
	ApprovalID string
	ToolCall   ToolCall

	// Reason optionally explains why approval is needed.
	Reason string

	// IsAutomatic marks requests that an approval policy resolved without
	// user input.
	IsAutomatic bool

	// Signature authenticates the request when a tool approval secret is
	// configured.
	Signature string
}

// ToolApprovalResponse is the decision on a tool approval request.
type ToolApprovalResponse struct {
	ApprovalID string
	ToolCall   ToolCall
	Approved   bool

	// Reason optionally explains the decision.
	Reason string

	// ProviderExecuted marks responses to provider-executed tool calls. Only
	// those are sent to the model.
	ProviderExecuted bool
}

// ProviderToolError is the failure reported by a provider-executed tool.
type ProviderToolError struct {
	// Result is the provider's JSON description of the failure.
	Result json.RawMessage
}

// Error returns the provider's result as text.
func (e *ProviderToolError) Error() string {
	return string(e.Result)
}

// InputSchemaInput returns the validated input before tool input refinement
// when a refinement changed it, and nil otherwise. Approval requests record
// it so that approvals can be revalidated against the input the schema
// accepted.
func (c ToolCall) InputSchemaInput() json.RawMessage {
	return refinedFromInput(c)
}

func (TextContent) isContentPart()          {}
func (CustomContent) isContentPart()        {}
func (ReasoningContent) isContentPart()     {}
func (ReasoningFileContent) isContentPart() {}
func (SourceContent) isContentPart()        {}
func (FileContent) isContentPart()          {}
func (ToolCall) isContentPart()             {}
func (ToolResult) isContentPart()           {}
func (ToolError) isContentPart()            {}
func (ToolApprovalRequest) isContentPart()  {}
func (ToolApprovalResponse) isContentPart() {}

func (ToolResult) isToolOutput() {}
func (ToolError) isToolOutput()  {}

// toolOutputCallID returns the tool call ID of an output.
func toolOutputCallID(output ToolOutput) string {
	switch typed := output.(type) {
	case ToolResult:
		return typed.ToolCallID
	case ToolError:
		return typed.ToolCallID
	default:
		return ""
	}
}

// toolOutputIsProviderExecuted reports whether the provider produced an
// output.
func toolOutputIsProviderExecuted(output ToolOutput) bool {
	switch typed := output.(type) {
	case ToolResult:
		return typed.ProviderExecuted
	case ToolError:
		return typed.ProviderExecuted
	default:
		return false
	}
}

// toolErrorValue returns the value that describes a tool failure to the
// model: the provider's JSON result for provider-executed tools and the
// error itself otherwise.
func toolErrorValue(err error) any {
	if providerErr, ok := errors.AsType[*ProviderToolError](err); ok {
		return providerErr.Result
	}

	return err
}
