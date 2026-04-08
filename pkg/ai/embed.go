package ai

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// EmbedResult is the result of a single text embedding.
type EmbedResult struct {
	// Embedding is the vector representation of the input text.
	Embedding []float64

	// Usage tracks token consumption.
	Usage Usage
}

// EmbedManyResult is the result of a batch text embedding.
type EmbedManyResult struct {
	// Embeddings contains one vector per input text, in the same order
	// as the input.
	Embeddings [][]float64

	// Usage tracks total token consumption across all inputs.
	Usage Usage
}

// Embed embeds a single text using the given embedding model.
func Embed(ctx context.Context, m model.EmbeddingModel, text string) (*EmbedResult, error) {
	resp, err := m.DoEmbed(ctx, []string{text})
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("embed: %w", ErrNoContent)
	}

	return &EmbedResult{
		Embedding: resp.Embeddings[0],
		Usage:     resp.Usage,
	}, nil
}

// EmbedMany embeds multiple texts using the given embedding model.
// The returned embeddings are in the same order as the input texts.
func EmbedMany(ctx context.Context, m model.EmbeddingModel, texts []string) (*EmbedManyResult, error) {
	resp, err := m.DoEmbed(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("embed many: %w", err)
	}

	return &EmbedManyResult{
		Embeddings: resp.Embeddings,
		Usage:      resp.Usage,
	}, nil
}
