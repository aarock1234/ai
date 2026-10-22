package openrouter

import (
	"context"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

// EmbeddingModel is an OpenRouter embedding model. It sends
// OpenAI-compatible embedding requests.
type EmbeddingModel struct {
	provider  string
	embedding *openaicompatible.EmbeddingModel
}

// Provider returns the provider identifier, "openrouter.embedding" by
// default.
func (m *EmbeddingModel) Provider() string {
	return m.provider
}

// ModelID returns the OpenRouter model identifier.
func (m *EmbeddingModel) ModelID() string {
	return m.embedding.ModelID()
}

// MaxEmbeddingsPerCall returns the per-call embedding limit.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	return m.embedding.MaxEmbeddingsPerCall()
}

// SupportsParallelCalls reports whether calls may run concurrently.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return m.embedding.SupportsParallelCalls()
}

// Embed embeds the values in one call.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	return m.embedding.Embed(ctx, options)
}

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)
