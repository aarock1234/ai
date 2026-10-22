// Command stream-text streams text from a model, printing each delta as it
// arrives, then prints the finish reason of the completed stream.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/stream-text
package main

import (
	"context"
	"fmt"
	"io"
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

	stream, err := ai.StreamText(ctx, ai.StreamTextOptions{
		Model: provider.Responses("gpt-5.4"),
		Prompt: ai.Prompt{
			Prompt: "Write a short poem about the Go gopher.",
		},
	})
	if err != nil {
		return fmt.Errorf("start stream: %w", err)
	}
	defer closeStream(stream)

	// The generation runs while the loop pulls deltas. Pressing Ctrl+C
	// cancels ctx, which ends the stream early.
	for delta := range stream.TextStream() {
		fmt.Print(delta)
	}

	fmt.Println()

	result, err := stream.Result()
	if err != nil {
		return fmt.Errorf("stream text: %w", err)
	}

	fmt.Println()
	fmt.Printf("finish reason: %s\n", result.FinishReason)

	return nil
}

// closeStream releases the stream and reports a failure to close it.
func closeStream(stream io.Closer) {
	if err := stream.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "error: close stream: %v\n", err)
	}
}
