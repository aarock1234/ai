package googlevertex

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/geminilive"
	"github.com/aarock1234/ai/internal/geministream"
	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/google"
	"github.com/aarock1234/ai/pkg/provider/googlevertex/internal/vertex"
)

// liveWebSocketPath is the Live API service of Vertex AI.
const liveWebSocketPath = "google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

// GeminiTranscriptionStreamMetadata is the metadata of a streaming Gemini
// transcription, stored under ProviderMetadata["google"] when usage is
// reported.
type GeminiTranscriptionStreamMetadata = google.TranscriptionStreamMetadata

var _ model.StreamingTranscriptionModel = (*TranscriptionModel)(nil)

// StreamTranscribe transcribes live audio with a live Gemini model, such as
// gemini-3.5-transcribe-live, over the Vertex AI Live API WebSocket, which
// authenticates with a bearer token from the token source. Other models do
// not stream. The input audio must be 16 kHz 16-bit PCM. Once Audio reaches
// io.EOF, the stream finishes when the server reports that all processing is
// complete, or after a quiet window.
func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, options model.TranscriptionStreamOptions) (*model.TranscriptionStreamResult, error) {
	if err := m.validateLiveRequest(options); err != nil {
		return nil, err
	}

	timestamp := time.Now()

	geminiOptions, _, err := readOptions[GeminiTranscriptionOptions](options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	if err := geminiOptions.validate(); err != nil {
		return nil, err
	}

	if err := geminilive.ValidateInputAudioFormat(options.InputAudioFormat); err != nil {
		return nil, err
	}

	location, err := vertex.LoadLocation(m.settings.location, "")
	if err != nil {
		return nil, err
	}

	setup, err := m.liveSetup(location, geminiOptions)
	if err != nil {
		return nil, err
	}

	dial, err := m.liveDialOptions(ctx, location, options)
	if err != nil {
		return nil, err
	}

	return geministream.Start(ctx, geministream.Transcription{
		ModelID:             m.modelID,
		Dial:                dial,
		Setup:               setup,
		Options:             options,
		Timestamp:           timestamp,
		DefaultErrorMessage: "Vertex Live API error",
		Name:                "google vertex live transcription",
		Metadata:            liveTranscriptionMetadata,
	})
}

// validateLiveRequest checks that the model streams and that the call has
// audio and standard credentials.
func (m *TranscriptionModel) validateLiveRequest(options model.TranscriptionStreamOptions) error {
	if err := m.settings.requireTokenSource("transcription models"); err != nil {
		return err
	}

	if !strings.HasPrefix(m.modelID, "gemini") {
		return fmt.Errorf("%w: streaming transcription with %s", model.ErrUnsupported, m.modelID)
	}

	if !geminilive.IsLiveModel(m.modelID) {
		return fmt.Errorf(
			"%w: model %q does not support streaming transcription; use a live model such as gemini-3.5-transcribe-live",
			model.ErrInvalidArgument,
			m.modelID,
		)
	}

	if options.Audio == nil {
		return fmt.Errorf("%w: streaming transcription requires audio", model.ErrInvalidArgument)
	}

	return nil
}

// liveSetup encodes the session setup, which addresses the model by its
// resource name in the project and location.
func (m *TranscriptionModel) liveSetup(location string, geminiOptions GeminiTranscriptionOptions) (geminilive.EncodedSetup, error) {
	project, err := vertex.LoadProject(m.settings.project)
	if err != nil {
		return geminilive.EncodedSetup{}, err
	}

	return geminilive.EncodeSetup(geminilive.Setup[GeminiTranscriptionOptions]{
		Model:                   vertex.ModelResource(project, location, "google", m.modelID),
		InputAudioTranscription: geminiOptions,
	})
}

// liveDialOptions describes the Live API WebSocket of the location with the
// bearer token and the provider and call headers.
func (m *TranscriptionModel) liveDialOptions(ctx context.Context, location string, options model.TranscriptionStreamOptions) (livetranscription.DialOptions, error) {
	liveURL := "wss://" + vertex.Host(location) + "/ws/" + liveWebSocketPath

	request, err := m.settings.request(ctx, liveURL, options.Headers)
	if err != nil {
		return livetranscription.DialOptions{}, err
	}

	return livetranscription.DialOptions{
		URL:        request.URL,
		Client:     request.Client,
		Header:     request.Headers,
		ParseError: request.ParseError,
	}, nil
}

func liveTranscriptionMetadata(usageMetadata json.RawMessage) model.ProviderMetadata {
	return model.ProviderMetadata{
		google.Namespace: GeminiTranscriptionStreamMetadata{
			UsageMetadata: usageMetadata,
		},
	}
}
