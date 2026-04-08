package model

import "context"

// EmbeddingModel is the interface for text embedding providers.
// It is separate from Model because embedding and generation are
// distinct capabilities.
type EmbeddingModel interface {
	// Provider returns the provider name (e.g. "openai").
	Provider() string

	// ModelID returns the model identifier (e.g. "text-embedding-3-small").
	ModelID() string

	// DoEmbed embeds the given texts and returns a vector for each input.
	// Providers may impose their own limits on batch size.
	DoEmbed(ctx context.Context, texts []string) (*EmbedResponse, error)
}

// EmbedResponse is the normalized response from an embedding request.
type EmbedResponse struct {
	// Embeddings contains one float64 slice per input text, in the same
	// order as the input.
	Embeddings [][]float64

	// Usage tracks token consumption for the embedding request.
	Usage Usage
}
