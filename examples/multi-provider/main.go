// Command multi-provider registers several providers in a Registry and selects
// the language model at runtime from a combined "provider:model" ID.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/multi-provider openai:gpt-5.4
//	ANTHROPIC_API_KEY=... go run ./examples/multi-provider anthropic:claude-sonnet-4-5
//	GOOGLE_GENERATIVE_AI_API_KEY=... go run ./examples/multi-provider google:gemini-3-flash-preview
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
	"github.com/aarock1234/ai/pkg/provider/google"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

const defaultModelID = "openai:gpt-5.4"

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelID := defaultModelID
	if len(os.Args) > 1 {
		modelID = os.Args[1]
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Providers read their API keys from the environment when a request is
	// sent, so registering a provider without a key is harmless.
	registry := ai.NewRegistry(ai.RegistryOptions{
		Providers: map[string]model.Provider{
			"anthropic": anthropic.New(),
			"google":    google.New(),
			"openai":    openai.New(),
		},
	})

	languageModel, err := registry.LanguageModel(modelID)
	if err != nil {
		return fmt.Errorf("resolve model %s: %w", modelID, err)
	}

	fmt.Printf("using %s (%s)\n\n", languageModel.ModelID(), languageModel.Provider())

	result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
		Model: languageModel,
		Prompt: ai.Prompt{
			Prompt: "Introduce yourself in one sentence.",
		},
	})
	if err != nil {
		return fmt.Errorf("generate text with %s: %w", modelID, err)
	}

	fmt.Println(result.Text)

	return nil
}
