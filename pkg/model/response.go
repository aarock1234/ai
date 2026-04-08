package model

// Response is the normalized response from a non-streaming generation
// request. Providers populate this from their API-specific response format.
type Response struct {
	// Content contains the generated content segments (text, tool calls).
	Content []Content

	// FinishReason indicates why the model stopped generating.
	FinishReason FinishReason

	// Usage tracks token consumption for this request.
	Usage Usage
}

// Content is the interface for response content segments.
type Content interface {
	contentType()
}

// TextContent is a plain text segment in the response.
type TextContent struct {
	Text string
}

func (TextContent) contentType() {}

// ToolCallContent represents a tool invocation requested by the model.
type ToolCallContent struct {
	ToolCallID string
	ToolName   string
	Args       string
}

func (ToolCallContent) contentType() {}

// FinishReason indicates why the model stopped generating.
type FinishReason string

const (
	// FinishReasonStop indicates the model naturally completed generation.
	FinishReasonStop FinishReason = "stop"

	// FinishReasonLength indicates the model hit the token limit.
	FinishReasonLength FinishReason = "length"

	// FinishReasonToolCalls indicates the model is requesting tool execution.
	FinishReasonToolCalls FinishReason = "tool_calls"

	// FinishReasonContentFilter indicates the response was filtered.
	FinishReasonContentFilter FinishReason = "content_filter"

	// FinishReasonUnknown indicates an unrecognized finish reason.
	FinishReasonUnknown FinishReason = "unknown"
)

// Usage tracks token consumption for a single request.
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// TotalTokens returns the sum of input and output tokens.
func (u Usage) TotalTokens() int {
	return u.InputTokens + u.OutputTokens
}
