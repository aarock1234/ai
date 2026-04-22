package model

import "encoding/json"

// Chunk is the interface for streaming response segments. Each chunk
// carries a delta of the response as it arrives from the provider.
type Chunk interface {
	chunkType()
}

// StreamStartChunk signals the stream has begun and carries any warnings
// the provider surfaced up front (e.g. unsupported settings).
type StreamStartChunk struct {
	Warnings []string
}

func (StreamStartChunk) chunkType() {}

// TextDelta is a partial text segment from a streaming response.
type TextDelta struct {
	Text string
}

func (TextDelta) chunkType() {}

// ReasoningDelta is a partial reasoning segment from a streaming response.
// Providers may interleave multiple reasoning streams; ID disambiguates
// them when the provider surfaces per-block identifiers. Signature and
// Format preserve provider-native reasoning metadata so the block can
// be round-tripped back in a subsequent request.
type ReasoningDelta struct {
	// Text is the plaintext reasoning fragment. Empty for redacted blocks.
	Text string

	// ID is the provider-assigned identifier for the reasoning block
	// this delta belongs to, when available.
	ID string

	// Signature is the provider's opaque signature for the block, when
	// supplied.
	Signature string

	// Format identifies the reasoning block's provider-specific shape,
	// when supplied.
	Format string

	// Redacted indicates the reasoning is encrypted or filtered. Text
	// is empty in this case.
	Redacted bool
}

func (ReasoningDelta) chunkType() {}

// ToolCallDelta is a partial tool call from a streaming response.
// Tool calls may arrive across multiple deltas; the consumer
// accumulates them by ToolCallID.
type ToolCallDelta struct {
	ToolCallID string
	ToolName   string
	ArgsJSON   string

	// ProviderExecuted is true when the provider ran the tool itself.
	// Consumers should not execute the call locally.
	ProviderExecuted bool
}

func (ToolCallDelta) chunkType() {}

// ToolResultChunk delivers the result of a provider-executed tool call
// inline with the stream.
type ToolResultChunk struct {
	ToolCallID string
	ToolName   string
	Content    string
	IsError    bool
}

func (ToolResultChunk) chunkType() {}

// FileChunk delivers a complete file emitted by the model (typically a
// generated image or audio clip). Providers deliver files as whole units
// rather than as byte deltas.
type FileChunk struct {
	Data     []byte
	URL      string
	MIMEType string
	Filename string
	Metadata ProviderMetadata
}

func (FileChunk) chunkType() {}

// SourceChunk delivers a source citation observed mid-stream (e.g. a
// web search citation).
type SourceChunk struct {
	SourceType SourceType
	ID         string
	URL        string
	Title      string
	MIMEType   string
	Filename   string
	StartIndex int
	EndIndex   int
	Metadata   ProviderMetadata
}

func (SourceChunk) chunkType() {}

// ProviderMetadataChunk carries provider/model streaming metadata that
// becomes available mid-stream.
type ProviderMetadataChunk struct {
	Metadata ProviderMetadata
}

func (ProviderMetadataChunk) chunkType() {}

// FinishChunk signals that the model has stopped generating.
type FinishChunk struct {
	FinishReason    FinishReason
	RawFinishReason string
}

func (FinishChunk) chunkType() {}

// UsageChunk carries token usage information, typically sent as the
// final chunk in a stream.
type UsageChunk struct {
	Usage Usage
}

func (UsageChunk) chunkType() {}

// ErrorChunk signals that the provider reported a recoverable error
// during the stream. Callers may continue reading subsequent chunks,
// but a FinishChunk with FinishReasonError is expected to follow.
type ErrorChunk struct {
	Err error
}

// Error reports the underlying provider error.
func (c ErrorChunk) Error() string {
	if c.Err == nil {
		return "stream error"
	}

	return c.Err.Error()
}

// Unwrap exposes the underlying error for errors.Is / errors.As.
func (c ErrorChunk) Unwrap() error { return c.Err }

func (ErrorChunk) chunkType() {}

// RawChunk is a passthrough of the provider's raw event payload. It is
// emitted only when a provider is configured to include raw frames and
// is primarily useful for debugging or implementing provider-specific
// functionality on top of the SDK.
type RawChunk struct {
	Data json.RawMessage
}

func (RawChunk) chunkType() {}
