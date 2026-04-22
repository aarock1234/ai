package ai

import "encoding/json"

// Role identifies the speaker of a message.
type Role string

const (
	// RoleSystem identifies a system-level message that sets context.
	RoleSystem Role = "system"

	// RoleUser identifies a message from the user.
	RoleUser Role = "user"

	// RoleAssistant identifies a message from the model.
	RoleAssistant Role = "assistant"

	// RoleTool identifies a message carrying a tool execution result.
	RoleTool Role = "tool"
)

// Message is a single message in a conversation. For simple text
// messages, set Content directly. For multi-part messages (text +
// images, tool results), use the Parts field instead.
type Message struct {
	Role    Role
	Content string
	Parts   []Part
}

// Part is the interface for message content segments.
type Part interface {
	partType()
}

// TextPart is a plain text segment in a message.
type TextPart struct {
	Text string
}

func (TextPart) partType() {}

// ImagePart is an image segment, specified by URL or inline data.
type ImagePart struct {
	URL      string
	Data     []byte
	MIMEType string
}

func (ImagePart) partType() {}

// FilePart is a generic file attachment (PDF, audio, etc.). Images
// should use ImagePart; FilePart covers the other MIME types.
type FilePart struct {
	URL      string
	Data     []byte
	MIMEType string
	Filename string
}

func (FilePart) partType() {}

// ReasoningPart carries a reasoning block from a prior assistant turn
// back to the provider. Providers that support round-tripping reasoning
// (Anthropic extended thinking, OpenRouter reasoning_details) honor the
// Signature and Format fields.
type ReasoningPart struct {
	Text      string
	Signature string
	ID        string
	Format    string
	Redacted  bool
}

func (ReasoningPart) partType() {}

// ToolCallPart represents a tool call made by the assistant.
type ToolCallPart struct {
	ToolCallID       string
	ToolName         string
	Args             json.RawMessage
	ProviderExecuted bool
}

func (ToolCallPart) partType() {}

// ToolResultPart carries the result of a tool invocation.
type ToolResultPart struct {
	ToolCallID string
	ToolName   string
	Content    string
	IsError    bool
}

func (ToolResultPart) partType() {}
