package ai

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Transcribe converts audio into text.
func Transcribe(ctx context.Context, m model.TranscriptionModel, req model.TranscriptionRequest) (*model.TranscriptionResponse, error) {
	resp, err := m.DoTranscribe(ctx, &req)
	if err != nil {
		return nil, fmt.Errorf("transcribe: %w", err)
	}

	return resp, nil
}
