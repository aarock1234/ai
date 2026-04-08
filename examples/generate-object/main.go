// Generate structured output and unmarshal it into a Go type.
// The SDK derives the JSON Schema from the type automatically.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run ./examples/generate-object
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

// Recipe is the structured output type. The JSON tags control the
// schema field names, and jsonschema tags add descriptions.
type Recipe struct {
	Name        string       `json:"name"        jsonschema:"description=Name of the dish"`
	Cuisine     string       `json:"cuisine"      jsonschema:"description=Type of cuisine"`
	PrepTime    string       `json:"prep_time"    jsonschema:"description=Preparation time"`
	Ingredients []Ingredient `json:"ingredients"`
	Steps       []string     `json:"steps"        jsonschema:"description=Step-by-step instructions"`
}

type Ingredient struct {
	Name   string `json:"name"`
	Amount string `json:"amount"`
}

func main() {
	ctx := context.Background()

	result, err := ai.GenerateObject[Recipe](ctx, openai.Chat(""),
		ai.WithPrompt("A recipe for chicken tikka masala."),
	)
	if err != nil {
		log.Fatal(err)
	}

	r := result.Object

	fmt.Printf("%s (%s)\n", r.Name, r.Cuisine)
	fmt.Printf("Prep time: %s\n\n", r.PrepTime)

	fmt.Println("Ingredients:")
	for _, ing := range r.Ingredients {
		fmt.Printf("  - %s %s\n", ing.Amount, ing.Name)
	}

	fmt.Println("\nSteps:")
	for i, step := range r.Steps {
		fmt.Printf("  %d. %s\n", i+1, step)
	}

	fmt.Printf("\nTokens: %d total\n", result.Usage.TotalTokens())
}
