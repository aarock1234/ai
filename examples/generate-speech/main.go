// Command generate-speech converts text to speech and writes the audio to an
// MP3 file.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/generate-speech [output.mp3]
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

const defaultOutputPath = "speech.mp3"

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	outputPath := defaultOutputPath
	if len(os.Args) > 1 {
		outputPath = os.Args[1]
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	provider := openai.New()

	result, err := ai.GenerateSpeech(ctx, ai.GenerateSpeechOptions{
		Model:        provider.Speech("gpt-4o-mini-tts"),
		Text:         "Hello from Go! This audio was generated with the AI SDK.",
		Voice:        "alloy",
		OutputFormat: "mp3",
		Instructions: "Speak in a warm and cheerful tone.",
	})
	if err != nil {
		return fmt.Errorf("generate speech: %w", err)
	}

	audio := result.Audio

	if err := os.WriteFile(outputPath, audio.Data, 0o644); err != nil {
		return fmt.Errorf("write audio: %w", err)
	}

	fmt.Printf("wrote %s (%s, %d bytes)\n", outputPath, audio.MediaType, len(audio.Data))

	for _, warning := range result.Warnings {
		fmt.Printf("warning: %s\n", warning)
	}

	return nil
}
