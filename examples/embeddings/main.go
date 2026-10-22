// Command embeddings embeds a query and a few documents in one EmbedMany call
// and ranks the documents by their cosine similarity to the query.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/embeddings
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	provider := openai.New()

	query := "How do I cancel a running operation?"
	documents := []string{
		"A context carries deadlines and cancellation signals across API boundaries.",
		"Slices are views over an underlying array.",
		"Goroutines are lightweight threads managed by the Go runtime.",
	}

	result, err := ai.EmbedMany(ctx, ai.EmbedManyOptions{
		Model:  provider.Embedding("text-embedding-3-small"),
		Values: append([]string{query}, documents...),
	})
	if err != nil {
		return fmt.Errorf("embed values: %w", err)
	}

	queryEmbedding := result.Embeddings[0]
	documentEmbeddings := result.Embeddings[1:]

	fmt.Printf("query: %s\n\n", query)

	for index, document := range documents {
		similarity, err := ai.CosineSimilarity(queryEmbedding, documentEmbeddings[index])
		if err != nil {
			return fmt.Errorf("compare embeddings: %w", err)
		}

		fmt.Printf("%.3f  %s\n", similarity, document)
	}

	if result.Usage != nil {
		fmt.Printf("\ntokens: %d\n", result.Usage.Tokens)
	}

	return nil
}
