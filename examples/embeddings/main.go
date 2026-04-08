// Embed texts and compute cosine similarity between them.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run ./examples/embeddings
package main

import (
	"context"
	"fmt"
	"log"
	"math"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func main() {
	ctx := context.Background()
	model := openai.Embedding("text-embedding-3-small")

	texts := []string{
		"The cat sat on the mat.",
		"A kitten rested on the rug.",
		"The stock market closed higher today.",
	}

	result, err := ai.EmbedMany(ctx, model, texts)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Cosine similarity:")
	for i := range texts {
		for j := i + 1; j < len(texts); j++ {
			sim := cosineSimilarity(result.Embeddings[i], result.Embeddings[j])
			fmt.Printf("  [%d] vs [%d]: %.4f  (%q vs %q)\n", i, j, sim, texts[i], texts[j])
		}
	}

	fmt.Printf("\nTokens used: %d\n", result.Usage.InputTokens)
}

func cosineSimilarity(a, b []float64) float64 {
	var dot, normA, normB float64

	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
