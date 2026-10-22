// Command rerank orders documents by their relevance to a query with a Cohere
// reranking model and prints the most relevant ones.
//
// Usage:
//
//	COHERE_API_KEY=... go run ./examples/rerank
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/cohere"
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

	provider := cohere.New()

	query := "What is the capital of France?"

	result, err := ai.Rerank(ctx, ai.RerankOptions[string]{
		Model: provider.Reranking("rerank-v3.5"),
		Query: query,
		Documents: []string{
			"Berlin is the capital of Germany.",
			"Paris is the capital and largest city of France.",
			"The Eiffel Tower was completed in 1889.",
			"France borders Spain, Italy and Germany.",
		},
		TopN: new(3),
	})
	if err != nil {
		return fmt.Errorf("rerank documents: %w", err)
	}

	fmt.Printf("query: %s\n\n", query)

	for rank, ranked := range result.Ranking {
		fmt.Printf("%d. %.3f  %s (document %d)\n", rank+1, ranked.Score, ranked.Document, ranked.OriginalIndex)
	}

	return nil
}
