// Rerank a set of documents for a query.
//
// Usage:
//
//	COHERE_API_KEY=... go run ./examples/rerank
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/cohere"
)

func main() {
	ctx := context.Background()

	result, err := ai.Rerank(ctx, cohere.Rerank("rerank-v3.5"), model.RerankRequest{
		Query: "golang concurrency patterns",
		Documents: []string{
			"Introduction to cooking with cast iron.",
			"Go concurrency patterns with channels and worker pools.",
			"Travel guide to Lisbon neighborhoods.",
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, item := range result.Results {
		fmt.Printf("%d %.3f %s\n", item.Index, item.Score, item.Document)
	}
}
