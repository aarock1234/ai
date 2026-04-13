// Transcribe an audio file to text.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run ./examples/transcribe path\\to\\audio.mp3
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
	if len(os.Args) < 2 {
		log.Fatal("usage: go run ./examples/transcribe path\\to\\audio.mp3")
	}

	audioData, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	result, err := ai.Transcribe(ctx, openai.Transcription("gpt-4o-mini-transcribe"), model.TranscriptionRequest{
		Audio: model.AudioInput{
			Data:     audioData,
			Filename: os.Args[1],
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	log.Print(result.Text)
}
