package model

// ProviderMetadata carries normalized provider/model response metadata.
// Known cross-provider fields live at the top level; provider-specific
// extras live in Provider.
type ProviderMetadata struct {
	// RequestID is the provider-generated request or message identifier,
	// when available.
	RequestID string

	// ModelID is the provider-reported model identifier for the response,
	// when available.
	ModelID string

	// Provider carries provider-specific response metadata keyed by field
	// name. Values are kept stringly typed until the SDK has stronger
	// cross-provider guarantees.
	Provider map[string]string
}

// Clone returns a deep copy of the metadata.
func (m ProviderMetadata) Clone() ProviderMetadata {
	cloned := ProviderMetadata{
		RequestID: m.RequestID,
		ModelID:   m.ModelID,
	}

	if len(m.Provider) > 0 {
		cloned.Provider = make(map[string]string, len(m.Provider))
		for key, value := range m.Provider {
			cloned.Provider[key] = value
		}
	}

	return cloned
}

// Response is the normalized response from a non-streaming generation
// request. Providers populate this from their API-specific response format.
type Response struct {
	// Content contains the generated content segments (text, tool calls).
	Content []Content

	// FinishReason indicates why the model stopped generating.
	FinishReason FinishReason

	// Usage tracks token consumption for this request.
	Usage Usage

	// Metadata carries provider/model response metadata when available.
	Metadata ProviderMetadata
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
