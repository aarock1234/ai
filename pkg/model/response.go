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
	// Content contains the generated content segments. Providers append
	// TextContent, ReasoningContent, ToolCallContent, FileContent, and
	// SourceContent in the order produced by the model.
	Content []Content

	// FinishReason indicates why the model stopped generating.
	FinishReason FinishReason

	// RawFinishReason is the provider-native finish reason string.
	RawFinishReason string

	// Usage tracks token consumption for this request.
	Usage Usage

	// Metadata carries provider/model response metadata when available.
	Metadata ProviderMetadata

	// Warnings carries provider warnings when available. Providers use
	// these to surface unsupported settings or partially applied options.
	Warnings []string
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

// ReasoningContent is a reasoning segment emitted by the model. Providers
// may emit multiple reasoning blocks per response (e.g. Anthropic extended
// thinking, OpenRouter reasoning_details, DeepSeek-R1). Signature and ID
// are preserved so the block can round-trip back to the provider.
type ReasoningContent struct {
	// Text is the plaintext reasoning content. Empty for redacted blocks.
	Text string

	// Signature is the provider's opaque signature used to verify the
	// block when sent back in a subsequent request. Populated by
	// Anthropic extended thinking and some OpenRouter reasoning formats.
	Signature string

	// ID is the provider-assigned identifier for this reasoning block,
	// when available.
	ID string

	// Format identifies the reasoning block's provider-specific shape
	// (e.g. "anthropic-claude-v1", "reasoning.text", "reasoning.encrypted").
	// Empty when the provider does not supply one.
	Format string

	// Redacted indicates the reasoning text is unavailable (e.g. encrypted
	// or filtered by the provider). Text is empty in this case.
	Redacted bool

	// Metadata carries provider-specific metadata for this reasoning block.
	Metadata ProviderMetadata
}

func (ReasoningContent) contentType() {}

// ToolCallContent represents a tool invocation requested by the model.
type ToolCallContent struct {
	ToolCallID string
	ToolName   string
	Args       string

	// ProviderExecuted is true when the provider ran the tool itself
	// (e.g. OpenAI's built-in web_search). Consumers should not execute
	// the call locally.
	ProviderExecuted bool
}

func (ToolCallContent) contentType() {}

// ToolResultContent is the result of a provider-executed tool call that
// arrives inline with the assistant response. Client-executed tool
// results flow through ToolResultPart on subsequent requests instead.
type ToolResultContent struct {
	ToolCallID string
	ToolName   string
	Content    string
	IsError    bool
}

func (ToolResultContent) contentType() {}

// FileContent is a file emitted by the model (images, audio, documents).
// Either Data or URL is populated; providers return whichever the
// upstream API produces without unnecessary conversion.
type FileContent struct {
	// Data is the raw file bytes. Mutually exclusive with URL.
	Data []byte

	// URL is the file URL. Mutually exclusive with Data.
	URL string

	// MIMEType is the IANA media type (e.g. "image/png", "audio/wav").
	MIMEType string

	// Filename is the provider-reported filename, when available.
	Filename string

	// Metadata carries provider-specific metadata for this file.
	Metadata ProviderMetadata
}

func (FileContent) contentType() {}

// SourceType identifies the kind of source referenced in a SourceContent
// segment.
type SourceType string

const (
	// SourceTypeURL indicates a web URL source (e.g. web search citation).
	SourceTypeURL SourceType = "url"

	// SourceTypeDocument indicates a document source (e.g. PDF citation).
	SourceTypeDocument SourceType = "document"
)

// SourceContent is a reference to a source used to generate the
// response, such as a web search citation or document excerpt.
type SourceContent struct {
	// SourceType is "url" or "document".
	SourceType SourceType

	// ID is the provider-assigned identifier for this source.
	ID string

	// URL is the source URL. Populated for SourceTypeURL.
	URL string

	// Title is the human-readable title of the source, when available.
	Title string

	// MIMEType is the IANA media type of the source. Populated for
	// SourceTypeDocument (e.g. "application/pdf").
	MIMEType string

	// Filename is the source filename, when available.
	Filename string

	// StartIndex is the start offset into the response text that this
	// citation annotates, when supplied by the provider.
	StartIndex int

	// EndIndex is the end offset into the response text that this
	// citation annotates, when supplied by the provider.
	EndIndex int

	// Metadata carries provider-specific metadata for this source.
	Metadata ProviderMetadata
}

func (SourceContent) contentType() {}

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

	// FinishReasonError indicates the provider returned an error mid-stream.
	FinishReasonError FinishReason = "error"

	// FinishReasonUnknown indicates an unrecognized finish reason.
	FinishReasonUnknown FinishReason = "unknown"
)

// Usage tracks token consumption for a single request.
type Usage struct {
	InputTokens       int
	OutputTokens      int
	CachedInputTokens int
	ReasoningTokens   int
	ImageInputTokens  int
	ImageOutputTokens int
	AudioInputTokens  int
	AudioOutputTokens int
}

// TotalTokens returns the sum of input and output tokens.
func (u Usage) TotalTokens() int {
	return u.InputTokens + u.OutputTokens
}
