// Command generate-object generates structured output with GenerateText and
// ObjectOutput. The JSON Schema sent to the model is derived from the Recipe
// type, and the validated output is decoded into it.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/generate-object
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

// Recipe is the structured output. Its json tags name the schema properties,
// and its jsonschema tags describe them to the model.
type Recipe struct {
	Name        string       `json:"name" jsonschema:"description=The name of the dish"`
	Servings    int          `json:"servings"`
	Ingredients []Ingredient `json:"ingredients"`
	Steps       []string     `json:"steps" jsonschema:"description=Preparation steps in order"`
}

// Ingredient is one ingredient of a Recipe.
type Ingredient struct {
	Name   string `json:"name"`
	Amount string `json:"amount" jsonschema:"description=Quantity with its unit"`
}

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
			Prompt: "Give me a simple recipe for pancakes.",
		},
		Output: ai.ObjectOutput[Recipe](ai.ObjectOutputOptions{
			Name:        "recipe",
			Description: "A cooking recipe",
		}),
	})
	if err != nil {
		return fmt.Errorf("generate recipe: %w", err)
	}

	recipe, err := ai.OutputAs[Recipe](result.Output)
	if err != nil {
		return fmt.Errorf("read recipe: %w", err)
	}

	printRecipe(recipe)

	return nil
}

func printRecipe(recipe Recipe) {
	fmt.Printf("%s (serves %d)\n", recipe.Name, recipe.Servings)

	fmt.Println()
	fmt.Println("ingredients:")

	for _, ingredient := range recipe.Ingredients {
		fmt.Printf("  - %s %s\n", ingredient.Amount, ingredient.Name)
	}

	fmt.Println()
	fmt.Println("steps:")

	for number, step := range recipe.Steps {
		fmt.Printf("  %d. %s\n", number+1, step)
	}
}
