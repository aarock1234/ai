// Use the automatic tool loop to let the model call tools and
// incorporate results into its response.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run ./examples/tool-calling
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

type WeatherInput struct {
	City string `json:"city" jsonschema:"description=City name"`
}

type CalculateInput struct {
	Expression string `json:"expression" jsonschema:"description=A simple math expression like '2+2' or 'sqrt(16)'"`
}

func main() {
	ctx := context.Background()

	weather := ai.Tool{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		InputSchema: ai.SchemaFor[WeatherInput](),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var params WeatherInput
			if err := json.Unmarshal(input, &params); err != nil {
				return "", err
			}

			// Simulated weather data.
			return fmt.Sprintf(`{"city": %q, "temp_f": 72, "condition": "sunny"}`, params.City), nil
		},
	}

	calculate := ai.Tool{
		Name:        "calculate",
		Description: "Evaluate a simple math expression.",
		InputSchema: ai.SchemaFor[CalculateInput](),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var params CalculateInput
			if err := json.Unmarshal(input, &params); err != nil {
				return "", err
			}

			// Tiny evaluator for demonstration.
			if params.Expression == "sqrt(16)" {
				return fmt.Sprintf("%g", math.Sqrt(16)), nil
			}

			return "unsupported expression", nil
		},
	}

	result, err := ai.GenerateText(ctx, openai.Chat(""),
		ai.WithSystem("You are a helpful assistant. Use the provided tools when needed."),
		ai.WithPrompt("What's the weather in Tokyo? Also, what is the square root of 16?"),
		ai.WithTools(weather, calculate),
		ai.WithMaxSteps(3),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Text)

	fmt.Printf("\nSteps: %d\n", len(result.Steps))
	for i, step := range result.Steps {
		fmt.Printf("  Step %d: finish=%s, tools=%d\n", i+1, step.FinishReason, len(step.ToolCalls))
	}

	fmt.Printf("\nTool results:\n")
	for _, tr := range result.ToolResults {
		fmt.Printf("  %s: %s\n", tr.ToolName, tr.Content)
	}
}
