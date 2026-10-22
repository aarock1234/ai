package model

import "encoding/json"

// Role identifies the author of a message.
type Role string

// Roles in a prompt.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is one entry in a prompt: a SystemMessage, UserMessage,
// AssistantMessage or ToolMessage. Each role accepts only the parts that the
// specification allows for it, so invalid prompts do not compile.
type Message interface {
	Role() Role
}

// SystemMessage carries instructions for the model.
type SystemMessage struct {
	Content         string
	ProviderOptions ProviderOptions
}

// Role returns RoleSystem.
func (SystemMessage) Role() Role {
	return RoleSystem
}

// UserMessage carries user input.
type UserMessage struct {
	Content         []UserPart
	ProviderOptions ProviderOptions
}

// Role returns RoleUser.
func (UserMessage) Role() Role {
	return RoleUser
}

// AssistantMessage carries earlier model output, including provider-executed
// tool results.
type AssistantMessage struct {
	Content         []AssistantPart
	ProviderOptions ProviderOptions
}

// Role returns RoleAssistant.
func (AssistantMessage) Role() Role {
	return RoleAssistant
}

// ToolMessage carries results of locally executed tools and approval
// decisions.
type ToolMessage struct {
	Content         []ToolPart
	ProviderOptions ProviderOptions
}

// Role returns RoleTool.
func (ToolMessage) Role() Role {
	return RoleTool
}

// UserPart is a TextPart or FilePart.
type UserPart interface {
	isUserPart()
}

// AssistantPart is a TextPart, FilePart, CustomPart, ReasoningPart,
// ReasoningFilePart, ToolCallPart, ToolResultPart or ToolApprovalRequestPart.
type AssistantPart interface {
	isAssistantPart()
}

// ToolPart is a ToolResultPart or ToolApprovalResponsePart.
type ToolPart interface {
	isToolPart()
}

// TextPart is text content.
type TextPart struct {
	Text            string
	ProviderOptions ProviderOptions
}

func (TextPart) isUserPart()      {}
func (TextPart) isAssistantPart() {}

// FilePart is file content such as an image, PDF or audio clip.
type FilePart struct {
	// Filename is optional.
	Filename string

	// MediaType is an IANA media type. It may contain a wildcard such as
	// "image/*" when the exact type is unknown.
	MediaType string

	Data            FileData
	ProviderOptions ProviderOptions
}

func (FilePart) isUserPart()      {}
func (FilePart) isAssistantPart() {}

// ReasoningPart is reasoning text from an earlier assistant turn. Signatures
// and encrypted content travel in ProviderOptions.
type ReasoningPart struct {
	Text            string
	ProviderOptions ProviderOptions
}

func (ReasoningPart) isAssistantPart() {}

// ReasoningFilePart is a file produced during reasoning. Its data is inline
// bytes or a URL.
type ReasoningFilePart struct {
	MediaType       string
	Data            FileData
	ProviderOptions ProviderOptions
}

func (ReasoningFilePart) isAssistantPart() {}

// CustomPart is provider-specific content with no standard meaning. Kind has
// the form "provider.type"; the content itself lives in ProviderOptions.
type CustomPart struct {
	Kind            string
	ProviderOptions ProviderOptions
}

func (CustomPart) isAssistantPart() {}

// ToolCallPart is a tool call from an earlier assistant turn.
type ToolCallPart struct {
	ToolCallID string
	ToolName   string

	// Input is the JSON arguments object.
	Input json.RawMessage

	// ProviderExecuted marks calls that the provider ran itself.
	ProviderExecuted bool

	ProviderOptions ProviderOptions
}

func (ToolCallPart) isAssistantPart() {}

// ToolResultPart is the outcome of a tool call.
type ToolResultPart struct {
	ToolCallID      string
	ToolName        string
	Output          ToolResultOutput
	ProviderOptions ProviderOptions
}

func (ToolResultPart) isAssistantPart() {}
func (ToolResultPart) isToolPart()      {}

// ToolApprovalRequestPart records that a tool call awaited approval. It is
// part of conversation history only: the ai package removes it before calling
// a model, and providers ignore it.
type ToolApprovalRequestPart struct {
	ApprovalID string
	ToolCallID string

	// Reason optionally explains why approval is needed.
	Reason string

	// IsAutomatic marks requests resolved by policy without user input.
	IsAutomatic bool

	// Signature authenticates the request when approvals are signed.
	Signature string

	// InputSchemaInput is the validated input before tool input refinement.
	// It is set only when refinement changed the input, so that the approval
	// can be revalidated against the input the schema accepted.
	InputSchemaInput json.RawMessage
}

func (ToolApprovalRequestPart) isAssistantPart() {}

// ToolApprovalResponsePart answers a tool approval request. Providers that
// run approved tools themselves forward it; others ignore it.
type ToolApprovalResponsePart struct {
	ApprovalID string
	Approved   bool

	// Reason optionally explains the decision.
	Reason string

	// ProviderExecuted marks responses to provider-executed tool calls.
	ProviderExecuted bool

	ProviderOptions ProviderOptions
}

func (ToolApprovalResponsePart) isToolPart() {}
