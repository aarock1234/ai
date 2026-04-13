// Generate an image from a text prompt.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run ./examples/generate-image
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func main() {
	ctx := context.Background()

	result, err := ai.GenerateImage(ctx, openai.Image("gpt-image-1"), model.ImageRequest{
		Prompt:       "A watercolor illustration of a lighthouse on a cliff at sunrise.",
		Size:         "1024x1024",
		OutputFormat: "png",
	})
	if err != nil {
		log.Fatal(err)
	}

	if len(result.Images) == 0 {
		log.Fatal("no images returned")
	}

	fmt.Printf("Generated %d image(s)\n", len(result.Images))
	fmt.Printf("First image bytes: %d\n", len(result.Images[0].Data))
}
