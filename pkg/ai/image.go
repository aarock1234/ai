package ai

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// GenerateImage creates images from the given prompt.
func GenerateImage(ctx context.Context, m model.ImageModel, req model.ImageRequest) (*model.ImageResponse, error) {
	resp, err := m.DoGenerateImage(ctx, &req)
	if err != nil {
		return nil, fmt.Errorf("generate image: %w", err)
	}

	return resp, nil
}
