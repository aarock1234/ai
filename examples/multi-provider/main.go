// Use the provider registry for config-driven provider selection.
// The provider and model are chosen at runtime from environment
// variables instead of being hardcoded.
//
// Usage:
//
//	AI_PROVIDER=openai AI_MODEL=gpt-5.4 OPENAI_API_KEY=sk-... go run ./examples/multi-provider
//	AI_PROVIDER=openrouter AI_MODEL=openai/gpt-5.4 OPENROUTER_API_KEY=sk-... go run ./examples/multi-provider
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider"
	"github.com/aarock1234/ai/pkg/provider/openai"
	"github.com/aarock1234/ai/pkg/provider/openrouter"
)

func main() {
	providerName := envOrDefault("AI_PROVIDER", "openai")
	modelID := envOrDefault("AI_MODEL", "gpt-5.4")

	// Register all available providers.
	registry := provider.NewRegistry()
	registry.Register("openai", openai.New())
	registry.Register("openrouter", openrouter.New())

	// Look up the provider by name.
	p, ok := registry.Get(providerName)
	if !ok {
		log.Fatalf("unknown provider: %s", providerName)
	}

	fmt.Printf("Using provider=%s model=%s\n\n", providerName, modelID)

	ctx := context.Background()

	result, err := ai.GenerateText(ctx, p.Chat(modelID),
		ai.WithPrompt("Say hello and tell me which model you are."),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Text)
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}

	return fallback
}
