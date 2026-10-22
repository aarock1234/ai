// Command transcribe converts an audio file to text and prints the transcript
// with its timed segments.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/transcribe path/to/audio.mp3
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) < 2 {
		return errors.New("usage: transcribe path/to/audio.mp3")
	}

	audioPath := os.Args[1]

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	audio, err := os.ReadFile(audioPath)
	if err != nil {
		return fmt.Errorf("read audio: %w", err)
	}

	provider := openai.New()

	result, err := ai.Transcribe(ctx, ai.TranscribeOptions{
		Model: provider.Transcription("gpt-4o-mini-transcribe"),
		Audio: model.FileBytes{
			Data: audio,
		},
	})
	if err != nil {
		return fmt.Errorf("transcribe %s: %w", audioPath, err)
	}

	fmt.Println(result.Text)

	if result.Language != "" {
		fmt.Printf("\nlanguage: %s\n", result.Language)
	}

	if result.Duration > 0 {
		fmt.Printf("duration: %s\n", result.Duration)
	}

	for _, segment := range result.Segments {
		fmt.Printf("[%s - %s] %s\n", segment.Start, segment.End, segment.Text)
	}

	return nil
}
