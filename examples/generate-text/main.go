// Generate text with a simple prompt and print the result.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run ./examples/generate-text
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func main() {
	ctx := context.Background()

	result, err := ai.GenerateText(ctx, openai.Chat(""),
		ai.WithSystem("You are a helpful assistant that responds concisely."),
		ai.WithPrompt("Explain the difference between concurrency and parallelism in two sentences."),
		ai.WithTemperature(0.7),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Text)
	fmt.Printf("\nTokens: %d input, %d output\n", result.Usage.InputTokens, result.Usage.OutputTokens)
}
