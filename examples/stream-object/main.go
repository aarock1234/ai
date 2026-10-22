// Command stream-object streams structured output with StreamText and
// ObjectOutput. It reports the progress of the partial object while the JSON
// arrives, then prints the validated final object.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/stream-object
package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

// Itinerary is the structured output.
type Itinerary struct {
	City string `json:"city"`
	Days []Day  `json:"days"`
}

// Day is one day of an Itinerary.
type Day struct {
	Title      string   `json:"title" jsonschema:"description=A short theme for the day"`
	Activities []string `json:"activities"`
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

	stream, err := ai.StreamText(ctx, ai.StreamTextOptions{
		Model: provider.Responses("gpt-5.4"),
		Prompt: ai.Prompt{
			Prompt: "Plan a three-day trip to Kyoto with three activities per day.",
		},
		Output: ai.ObjectOutput[Itinerary](ai.ObjectOutputOptions{
			Name: "itinerary",
		}),
	})
	if err != nil {
		return fmt.Errorf("start stream: %w", err)
	}
	defer closeStream(stream)

	// Partial outputs are Itinerary values decoded from the incomplete JSON.
	// They are not validated, so fields may still be missing.
	plannedDays := 0

	for partial := range stream.PartialOutputStream() {
		itinerary, err := ai.OutputAs[Itinerary](partial)
		if err != nil {
			return fmt.Errorf("read partial itinerary: %w", err)
		}

		if len(itinerary.Days) > plannedDays {
			plannedDays = len(itinerary.Days)
			fmt.Printf("planning day %d...\n", plannedDays)
		}
	}

	output, err := stream.Output()
	if err != nil {
		return fmt.Errorf("stream itinerary: %w", err)
	}

	itinerary, err := ai.OutputAs[Itinerary](output)
	if err != nil {
		return fmt.Errorf("read itinerary: %w", err)
	}

	printItinerary(itinerary)

	return nil
}

func printItinerary(itinerary Itinerary) {
	fmt.Println()
	fmt.Printf("%s\n", itinerary.City)

	for number, day := range itinerary.Days {
		fmt.Println()
		fmt.Printf("day %d: %s\n", number+1, day.Title)

		for _, activity := range day.Activities {
			fmt.Printf("  - %s\n", activity)
		}
	}
}

// closeStream releases the stream and reports a failure to close it.
func closeStream(stream io.Closer) {
	if err := stream.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "error: close stream: %v\n", err)
	}
}
