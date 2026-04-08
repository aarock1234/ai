// Multi-turn conversation using WithMessages to pass conversation
// history back to the model.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run ./examples/conversation
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func main() {
	ctx := context.Background()
	model := openai.Chat("gpt-5.4")

	// Start a conversation.
	history := []ai.Message{
		{Role: ai.RoleUser, Content: "My name is Alex and I'm building a Go SDK."},
	}

	result, err := ai.GenerateText(ctx, model,
		ai.WithSystem("You are a friendly assistant. Remember everything the user tells you."),
		ai.WithMessages(history...),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Assistant: %s\n\n", result.Text)

	// Append the assistant's reply and a follow-up question.
	history = append(history,
		ai.Message{Role: ai.RoleAssistant, Content: result.Text},
		ai.Message{Role: ai.RoleUser, Content: "What's my name and what am I building?"},
	)

	result, err = ai.GenerateText(ctx, model,
		ai.WithSystem("You are a friendly assistant. Remember everything the user tells you."),
		ai.WithMessages(history...),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Assistant: %s\n", result.Text)
}
