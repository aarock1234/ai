// Generate speech from text.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run ./examples/generate-speech
package main

import (
	"context"
	"log"
	"os"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func main() {
	ctx := context.Background()

	result, err := ai.GenerateSpeech(ctx, openai.Speech("gpt-4o-mini-tts"), model.SpeechRequest{
		Input:          "The AI SDK for Go now supports speech synthesis.",
		Voice:          "alloy",
		ResponseFormat: "mp3",
	})
	if err != nil {
		log.Fatal(err)
	}

	if err := os.WriteFile("speech.mp3", result.Audio, 0o644); err != nil {
		log.Fatal(err)
	}

	log.Printf("wrote %d bytes to speech.mp3", len(result.Audio))
}
