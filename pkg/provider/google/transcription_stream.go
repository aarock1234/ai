package google

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/aarock1234/ai/internal/geminilive"
	"github.com/aarock1234/ai/internal/geministream"
	"github.com/aarock1234/ai/pkg/model"
)

// TranscriptionStreamMetadata is the Google metadata of a streaming
// transcription, stored under ProviderMetadata["google"] when usage is
// reported.
type TranscriptionStreamMetadata struct {
	UsageMetadata json.RawMessage `json:"usageMetadata"`
}

var _ model.StreamingTranscriptionModel = (*TranscriptionModel)(nil)

// StreamTranscribe transcribes live audio with a live model, such as
// gemini-3.5-transcribe-live, over the Live API WebSocket. Other models do
// not stream. The input audio must be 16 kHz 16-bit PCM, and the API key is
// required: it authenticates the WebSocket as a query parameter. Once Audio
// reaches io.EOF, the stream finishes when the server reports that all
// processing is complete, or after a quiet window.
func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, options model.TranscriptionStreamOptions) (*model.TranscriptionStreamResult, error) {
	if !geminilive.IsLiveModel(m.modelID) {
		return nil, fmt.Errorf(
			"%w: model %q does not support streaming transcription; use a live model such as gemini-3.5-transcribe-live",
			model.ErrInvalidArgument,
			m.modelID,
		)
	}

	if options.Audio == nil {
		return nil, fmt.Errorf("%w: streaming transcription requires audio", model.ErrInvalidArgument)
	}

	timestamp := time.Now()

	googleOptions, _, err := model.ProviderValue[TranscriptionOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := validateEnum("mode", googleOptions.Mode, transcriptionModes); err != nil {
		return nil, err
	}

	if err := geminilive.ValidateInputAudioFormat(options.InputAudioFormat); err != nil {
		return nil, err
	}

	setup, err := geminilive.EncodeSetup(geminilive.Setup[TranscriptionOptions]{
		Model:                   modelPath(m.modelID),
		InputAudioTranscription: googleOptions,
	})
	if err != nil {
		return nil, err
	}

	dial, err := m.provider.liveDialOptions(ctx, options.Headers, "streaming transcription")
	if err != nil {
		return nil, err
	}

	return geministream.Start(ctx, geministream.Transcription{
		ModelID:             m.modelID,
		Dial:                dial,
		Setup:               setup,
		Options:             options,
		Timestamp:           timestamp,
		DefaultErrorMessage: "Google Live API error",
		Name:                "google live transcription",
		Metadata:            liveTranscriptionMetadata,
	})
}

func liveTranscriptionMetadata(usageMetadata json.RawMessage) model.ProviderMetadata {
	return model.ProviderMetadata{
		Namespace: TranscriptionStreamMetadata{
			UsageMetadata: usageMetadata,
		},
	}
}
