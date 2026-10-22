package model

import (
	"context"
	"net/http"
)

// EmbeddingModel converts text into vectors.
type EmbeddingModel interface {
	Provider() string
	ModelID() string

	// MaxEmbeddingsPerCall returns the provider's batch limit, or zero when
	// there is none.
	MaxEmbeddingsPerCall() int

	// SupportsParallelCalls reports whether batches may be sent concurrently.
	SupportsParallelCalls() bool

	Embed(ctx context.Context, options EmbedOptions) (*EmbedResult, error)
}

// EmbedOptions are the inputs of one embedding call.
type EmbedOptions struct {
	Values []string

	// Dimensions optionally requests a vector size.
	Dimensions *int

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// EmbedResult is the outcome of one embedding call.
type EmbedResult struct {
	// Embeddings has one vector per input value, in input order.
	Embeddings [][]float64

	// Usage is nil when the provider does not report it.
	Usage *EmbeddingUsage

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
	Response         ResponseInfo
}

// EmbeddingUsage reports embedding token consumption.
type EmbeddingUsage struct {
	Tokens int
}
