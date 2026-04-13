// Package model defines the provider-facing interfaces and types for the
// AI SDK. Providers implement Model and EmbeddingModel; the core SDK
// orchestrates calls through these interfaces.
package model

import (
	"context"
	"iter"
)

// Model is the central interface that every language model provider
// implements. It supports both non-streaming and streaming generation.
// Keep this interface minimal — provider-specific capabilities flow
// through ProviderOptions on the request and ProviderMetadata on the
// response rather than additional methods.
type Model interface {
	// Provider returns the provider name (e.g. "openai", "openrouter").
	Provider() string

	// ModelID returns the model identifier (e.g. "gpt-5.4", "claude-sonnet-4-20250514").
	ModelID() string

	// DoGenerate performs a non-streaming generation request and returns
	// the complete response.
	DoGenerate(ctx context.Context, req *Request) (*Response, error)

	// DoStream performs a streaming generation request and returns an
	// iterator that yields chunks as they arrive from the provider.
	DoStream(ctx context.Context, req *Request) (*StreamResponse, error)
}

// ResponseFormatSupport reports whether a model supports a given
// response format natively.
type ResponseFormatSupport interface {
	SupportsResponseFormat(rf *ResponseFormat) bool
}

// StreamResponse wraps the streaming iterator with response metadata
// that is available before the stream is consumed.
type StreamResponse struct {
	// Stream yields chunks as they arrive from the provider. The caller
	// must range over this to consume the response. Breaking out of the
	// range early is safe and signals the provider to stop.
	Stream iter.Seq[Chunk]

	// Metadata carries provider/model response metadata known at stream
	// creation time.
	Metadata ProviderMetadata
}
