package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// agentInstructions are the agent's system prompt.
const agentInstructions = `You answer questions about a store's products.
Look up stock and prices with the tools before answering, and answer in one sentence.`

// ProductInput names a product.
type ProductInput struct {
	SKU string `json:"sku" jsonschema:"description=The product SKU, such as A-100"`
}

// Stock is the stock of a product.
type Stock struct {
	SKU      string `json:"sku"`
	Quantity int    `json:"quantity"`
}

// Price is the price of a product.
type Price struct {
	SKU      string  `json:"sku"`
	Amount   float64 `json:"amount"`
	Currency string  `json:"currency"`
}

// inventory is the simulated store.
var inventory = map[string]Stock{
	"A-100": {
		SKU:      "A-100",
		Quantity: 12,
	},
}

// prices is the simulated price list.
var prices = map[string]Price{
	"A-100": {
		SKU:      "A-100",
		Amount:   24.99,
		Currency: "USD",
	},
}

func checkAgent(ctx context.Context, languageModel model.LanguageModel) (string, error) {
	agent := newStoreAgent(languageModel)

	result, err := agent.Generate(ctx, ai.AgentCallOptions{
		Prompt: "Is A-100 in stock, and what does it cost?",
	})
	if err != nil {
		return "", err
	}

	called := calledTools(result.Steps)
	for _, name := range []string{"look_up_stock", "look_up_price"} {
		if !called[name] {
			return "", fmt.Errorf("the agent did not call %s", name)
		}
	}

	return report(result, result.Text), nil
}

// newStoreAgent returns an agent with two tools and a step limit.
func newStoreAgent(languageModel model.LanguageModel) *ai.ToolLoopAgent {
	lookUpStock := ai.NewTool(ai.ToolDefinition[ProductInput, Stock]{
		Name:        "look_up_stock",
		Description: "Look up how many units of a product are in stock.",
		Execute:     findStock,
	})

	lookUpPrice := ai.NewTool(ai.ToolDefinition[ProductInput, Price]{
		Name:        "look_up_price",
		Description: "Look up the price of a product.",
		Execute:     findPrice,
	})

	return ai.NewToolLoopAgent(ai.ToolLoopAgentSettings{
		ID: "store",
		GenerateTextOptions: ai.GenerateTextOptions{
			Model: languageModel,
			Prompt: ai.Prompt{
				System: agentInstructions,
			},
			Tools: []ai.Tool{
				lookUpStock,
				lookUpPrice,
			},
			StopWhen: []ai.StopCondition{
				ai.IsStepCount(5),
			},
			ProviderOptions: routingOptions(),
		},
	})
}

func findStock(_ context.Context, input ProductInput, _ ai.ToolExecutionOptions) (Stock, error) {
	stock, found := inventory[normalizeSKU(input.SKU)]
	if !found {
		return Stock{}, fmt.Errorf("product %q not found", input.SKU)
	}

	return stock, nil
}

func findPrice(_ context.Context, input ProductInput, _ ai.ToolExecutionOptions) (Price, error) {
	price, found := prices[normalizeSKU(input.SKU)]
	if !found {
		return Price{}, fmt.Errorf("product %q not found", input.SKU)
	}

	return price, nil
}

func normalizeSKU(sku string) string {
	return strings.ToUpper(strings.TrimSpace(sku))
}

// calledTools returns the names of the tools the steps called.
func calledTools(steps []ai.StepResult) map[string]bool {
	called := make(map[string]bool)

	for _, step := range steps {
		for _, call := range step.ToolCalls() {
			called[call.ToolName] = true
		}
	}

	return called
}
