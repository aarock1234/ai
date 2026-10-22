// Command agent builds a reusable ToolLoopAgent from a model, instructions
// and tools, then streams its answer to a prompt. The agent calls its tools
// across as many steps as it needs, up to its stop condition.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/agent
package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

const instructions = `You are a support agent for a bookstore.
Look up orders before answering questions about them, and keep answers short.`

// OrderInput identifies an order.
type OrderInput struct {
	OrderID string `json:"orderId" jsonschema:"description=The order number such as A-1001"`
}

// Order is the status of an order.
type Order struct {
	OrderID string `json:"orderId"`
	Title   string `json:"title"`
	Status  string `json:"status"`
}

// orders is the simulated order database.
var orders = map[string]Order{
	"A-1001": {
		OrderID: "A-1001",
		Title:   "The Go Programming Language",
		Status:  "shipped",
	},
	"A-1002": {
		OrderID: "A-1002",
		Title:   "Designing Data-Intensive Applications",
		Status:  "awaiting stock",
	},
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

	agent := newSupportAgent(openai.New())

	stream, err := agent.Stream(ctx, ai.AgentStreamOptions{
		AgentCallOptions: ai.AgentCallOptions{
			Prompt: "Where are my orders A-1001 and A-1002?",
		},
	})
	if err != nil {
		return fmt.Errorf("start agent: %w", err)
	}
	defer closeStream(stream)

	for delta := range stream.TextStream() {
		fmt.Print(delta)
	}

	fmt.Println()

	result, err := stream.Result()
	if err != nil {
		return fmt.Errorf("run agent: %w", err)
	}

	fmt.Println()
	fmt.Printf("steps: %d, tool calls: %d\n", len(result.Steps), countToolCalls(result.Steps))

	return nil
}

// newSupportAgent returns an agent that can look up orders. The same agent can
// serve any number of calls.
func newSupportAgent(provider *openai.Provider) *ai.ToolLoopAgent {
	lookUpOrder := ai.NewTool(ai.ToolDefinition[OrderInput, Order]{
		Name:        "look_up_order",
		Description: "Look up the status of an order by its order number.",
		Execute:     findOrder,
	})

	return ai.NewToolLoopAgent(ai.ToolLoopAgentSettings{
		ID: "bookstore-support",
		GenerateTextOptions: ai.GenerateTextOptions{
			Model: provider.Responses("gpt-5.4"),
			Prompt: ai.Prompt{
				System: instructions,
			},
			Tools: []ai.Tool{
				lookUpOrder,
			},
			StopWhen: []ai.StopCondition{
				ai.IsStepCount(10),
			},
		},
	})
}

func findOrder(_ context.Context, input OrderInput, _ ai.ToolExecutionOptions) (Order, error) {
	order, found := orders[strings.ToUpper(strings.TrimSpace(input.OrderID))]
	if !found {
		return Order{}, fmt.Errorf("order %q not found", input.OrderID)
	}

	return order, nil
}

func countToolCalls(steps []ai.StepResult) int {
	count := 0

	for _, step := range steps {
		count += len(step.ToolCalls())
	}

	return count
}

// closeStream releases the stream and reports a failure to close it.
func closeStream(stream io.Closer) {
	if err := stream.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "error: close stream: %v\n", err)
	}
}
