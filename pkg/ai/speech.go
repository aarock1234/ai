package ai

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// GenerateSpeech converts text into audio.
func GenerateSpeech(ctx context.Context, m model.SpeechModel, req model.SpeechRequest) (*model.SpeechResponse, error) {
	resp, err := m.DoGenerateSpeech(ctx, &req)
	if err != nil {
		return nil, fmt.Errorf("generate speech: %w", err)
	}

	return resp, nil
}
