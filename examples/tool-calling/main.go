// Command tool-calling gives the model a typed tool and lets GenerateText run
// the tool-calling loop: the model calls the tool, the SDK executes it and
// sends the result back, and the model answers with the result.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/tool-calling
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

// WeatherInput is the input the model produces. The tool's JSON Schema is
// derived from it.
type WeatherInput struct {
	City string `json:"city" jsonschema:"description=The city to look up"`
}

// WeatherReport is the output sent back to the model as JSON.
type WeatherReport struct {
	City        string `json:"city"`
	Temperature int    `json:"temperatureCelsius"`
	Condition   string `json:"condition"`
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

	weather := ai.NewTool(ai.ToolDefinition[WeatherInput, WeatherReport]{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		Execute:     lookUpWeather,
	})

	result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
		Model: provider.Responses("gpt-5.4"),
		Prompt: ai.Prompt{
			Prompt: "Should I bring an umbrella in Tokyo or in Lisbon today?",
		},
		Tools: []ai.Tool{
			weather,
		},
		// Without a stop condition, the loop ends after the first step.
		StopWhen: []ai.StopCondition{
			ai.IsStepCount(5),
		},
	})
	if err != nil {
		return fmt.Errorf("generate text: %w", err)
	}

	for _, step := range result.Steps {
		for _, call := range step.ToolCalls() {
			fmt.Printf("step %d: %s(%s)\n", step.StepNumber+1, call.ToolName, call.Input)
		}
	}

	fmt.Println()
	fmt.Println(result.Text)

	return nil
}

// lookUpWeather returns simulated weather. A real tool would call a weather
// API with ctx, which is cancelled when the generation is.
func lookUpWeather(_ context.Context, input WeatherInput, _ ai.ToolExecutionOptions) (WeatherReport, error) {
	city := strings.TrimSpace(input.City)
	if city == "" {
		return WeatherReport{}, errors.New("city is required")
	}

	report := WeatherReport{
		City:        city,
		Temperature: 21,
		Condition:   "sunny",
	}

	if strings.EqualFold(city, "Tokyo") {
		report.Temperature = 17
		report.Condition = "light rain"
	}

	return report, nil
}
