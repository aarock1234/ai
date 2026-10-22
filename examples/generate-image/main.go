// Command generate-image generates an image from a prompt and writes it to a
// PNG file. The output format is selected with typed OpenAI provider options.
//
// Usage:
//
//	OPENAI_API_KEY=... go run ./examples/generate-image [output.png]
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

const defaultOutputPath = "image.png"

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

	result, err := ai.GenerateImage(ctx, ai.GenerateImageOptions{
		Model:  provider.Image("gpt-image-1"),
		Prompt: "A watercolor painting of a gopher reading a book under a tree.",
		Size:   "1024x1024",
		ProviderOptions: model.ProviderOptions{
			"openai": openai.ImageOptions{
				OutputFormat: openai.ImageOutputPNG,
			},
		},
	})
	if err != nil {
		return fmt.Errorf("generate image: %w", err)
	}

	if len(result.Images) == 0 {
		return errors.New("generate image: no image returned")
	}

	image := result.Images[0]

	if err := os.WriteFile(outputPath, image.Data, 0o644); err != nil {
		return fmt.Errorf("write image: %w", err)
	}

	fmt.Printf("wrote %s (%s, %d bytes)\n", outputPath, image.MediaType, len(image.Data))

	for _, warning := range result.Warnings {
		fmt.Printf("warning: %s\n", warning)
	}

	return nil
}
