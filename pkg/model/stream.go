package model

// ProviderMetadataChunk carries provider/model streaming metadata that
// becomes available mid-stream.
type ProviderMetadataChunk struct {
	Metadata ProviderMetadata
}

func (ProviderMetadataChunk) chunkType() {}

// Chunk is the interface for streaming response segments. Each chunk
// carries a delta of the response as it arrives from the provider.
type Chunk interface {
	chunkType()
}

// TextDelta is a partial text segment from a streaming response.
type TextDelta struct {
	Text string
}

func (TextDelta) chunkType() {}

// ToolCallDelta is a partial tool call from a streaming response.
// Tool calls may arrive across multiple deltas; the consumer
// accumulates them by ToolCallID.
type ToolCallDelta struct {
	ToolCallID string
	ToolName   string
	ArgsJSON   string
}

func (ToolCallDelta) chunkType() {}

// FinishChunk signals that the model has stopped generating.
type FinishChunk struct {
	FinishReason FinishReason
}

func (FinishChunk) chunkType() {}

// UsageChunk carries token usage information, typically sent as the
// final chunk in a stream.
type UsageChunk struct {
	Usage Usage
}

func (UsageChunk) chunkType() {}
