// Command conversation holds a multi-turn conversation by appending each
// response's messages to the history before asking the next question.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/conversation
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

const instructions = "You are a friendly assistant. Keep your answers to one or two sentences."

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
	languageModel := provider.Responses("gpt-5.4")

	questions := []string{
		"My name is Alex and I'm building a Go SDK for language models.",
		"What's my name, and what am I building?",
	}

	var history []model.Message

	for _, question := range questions {
		history = append(history, userMessage(question))

		result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
			Model: languageModel,
			Prompt: ai.Prompt{
				System:   instructions,
				Messages: history,
			},
		})
		if err != nil {
			return fmt.Errorf("answer %q: %w", question, err)
		}

		// The response messages carry the assistant's reply, including any
		// tool calls and results, ready to send back on the next turn.
		history = append(history, result.ResponseMessages...)

		fmt.Printf("user: %s\n", question)
		fmt.Printf("assistant: %s\n\n", result.Text)
	}

	return nil
}

// userMessage returns a user message with a single text part.
func userMessage(text string) model.UserMessage {
	return model.UserMessage{
		Content: []model.UserPart{
			model.TextPart{
				Text: text,
			},
		},
	}
}
