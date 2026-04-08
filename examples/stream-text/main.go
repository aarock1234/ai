// Stream text from a model and print deltas as they arrive.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run ./examples/stream-text
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

	stream, err := ai.StreamText(ctx, openai.Chat(""),
		ai.WithPrompt("Write a short poem about Go programming."),
	)
	if err != nil {
		log.Fatal(err)
	}

	for delta := range stream.Text() {
		fmt.Print(delta)
	}
	fmt.Println()

	// After the stream finishes, you can access the full result.
	result, err := stream.Result()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nTokens: %d input, %d output\n", result.Usage.InputTokens, result.Usage.OutputTokens)
}
