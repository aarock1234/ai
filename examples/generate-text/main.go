// Command generate-text generates text from a prompt and prints it with the
// token usage.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/generate-text
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strconv"

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

	result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
		Model: provider.Responses("gpt-5.4"),
		Prompt: ai.Prompt{
			System: "You are a concise assistant.",
			Prompt: "Explain the difference between concurrency and parallelism in two sentences.",
		},
	})
	if err != nil {
		return fmt.Errorf("generate text: %w", err)
	}

	fmt.Println(result.Text)
	fmt.Println()
	fmt.Printf("finish reason: %s\n", result.FinishReason)
	fmt.Printf(
		"tokens: %s input, %s output\n",
		formatTokens(result.Usage.InputTokens),
		formatTokens(result.Usage.OutputTokens),
	)

	return nil
}

// formatTokens prints a token count that the provider may not report.
func formatTokens(count *int) string {
	if count == nil {
		return "unknown"
	}

	return strconv.Itoa(*count)
}
