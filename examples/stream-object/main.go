// Stream structured output and print the final object.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run ./examples/stream-object
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

type Answer struct {
	Topic   string `json:"topic"`
	Summary string `json:"summary"`
}

func main() {
	ctx := context.Background()

	stream, err := ai.StreamObject[Answer](ctx, openai.Chat("gpt-5.4-mini"),
		ai.WithPrompt("Summarize concurrency in Go as JSON with topic and summary fields."),
	)
	if err != nil {
		log.Fatal(err)
	}

	for delta := range stream.JSON() {
		fmt.Print(delta)
	}

	result, err := stream.Result()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\n\nTopic: %s\nSummary: %s\n", result.Object.Topic, result.Object.Summary)
}
