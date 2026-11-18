// Command openrouter runs the SDK's main features live against OpenRouter
// for each model named on the command line: text, object and array output
// with nested, enum and nullable fields, a two-tool agent, and usage cost.
// It finishes by asking for a model that does not exist and checking that
// the call fails with an APICallError.
//
// Every request asks OpenRouter to route only to providers that do not
// collect data.
//
// Usage:
//
//	OPENROUTER_API_KEY=... go run ./examples/openrouter anthropic/claude-opus-5.5 openai/gpt-6.1-terra
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openrouter"
)

// missingModelID names a model that OpenRouter does not have.
const missingModelID = "ai-sdk-go/no-such-model"

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelIDs := os.Args[1:]
	if len(modelIDs) == 0 {
		return errors.New("name at least one OpenRouter model ID")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	provider := openrouter.New(
		openrouter.WithAppName("ai-sdk-go examples"),
	)

	failures := 0

	for _, modelID := range modelIDs {
		fmt.Printf("== %s\n", modelID)

		failures += runChecks(ctx, provider.Chat(modelID))

		fmt.Println()
	}

	fmt.Printf("== %s\n", missingModelID)

	if err := checkMissingModel(ctx, provider.Chat(missingModelID)); err != nil {
		fmt.Printf("missing model: fail: %v\n", err)
		failures++
	}

	if failures > 0 {
		return fmt.Errorf("%d checks failed", failures)
	}

	return nil
}

// runChecks runs every check against one model, prints each outcome, and
// returns the number of failures.
func runChecks(ctx context.Context, languageModel model.LanguageModel) int {
	failures := 0

	for _, check := range checks {
		report, err := check.run(ctx, languageModel)
		if err != nil {
			fmt.Printf("%s: fail: %v\n", check.name, err)
			failures++

			continue
		}

		fmt.Printf("%s: ok\n%s\n", check.name, report)
	}

	return failures
}

// checkMissingModel asks for a model that does not exist and expects an
// APICallError with the status OpenRouter answered.
func checkMissingModel(ctx context.Context, languageModel model.LanguageModel) error {
	_, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
		Model: languageModel,
		Prompt: ai.Prompt{
			Prompt: "Say hello.",
		},
		ProviderOptions: routingOptions(),
	})
	if err == nil {
		return errors.New("the call succeeded")
	}

	apiError, ok := errors.AsType[*model.APICallError](err)
	if !ok {
		return fmt.Errorf("error is not an APICallError: %w", err)
	}

	fmt.Printf("missing model: ok\n  status %d, retryable %t: %s\n", apiError.StatusCode, apiError.IsRetryable, apiError.Message)

	return nil
}

// routingOptions request usage accounting and route only to providers that
// do not collect data.
func routingOptions() model.ProviderOptions {
	return model.ProviderOptions{
		openrouter.Namespace: routingChatOptions(),
	}
}

func routingChatOptions() openrouter.ChatOptions {
	return openrouter.ChatOptions{
		Usage: &openrouter.UsageOptions{
			Include: true,
		},
		Provider: &openrouter.ProviderRouting{
			DataCollection: openrouter.DataCollectionDeny,
		},
	}
}
