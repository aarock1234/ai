// Package geminilive implements live transcription over the Gemini Live API
// (BidiGenerateContent), which the Gemini API and Vertex AI share. Providers
// own the URL, authentication and metadata namespace.
package geminilive

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// InputSampleRate is the only rate of Live transcription input audio.
const InputSampleRate = 16000

// audioMimeType is the media type of the input audio chunks.
const audioMimeType = "audio/pcm;rate=16000"

// DefaultFinishGrace is how long the stream waits for a terminal signal
// after the input audio ended. Trailing transcripts restart the wait.
const DefaultFinishGrace = 3 * time.Second

// IsLiveModel reports whether a model ID names a live variant, the only
// models that stream.
func IsLiveModel(modelID string) bool {
	return strings.Contains(modelID, "-live")
}

// ValidateInputAudioFormat accepts 16 kHz 16-bit PCM only.
func ValidateInputAudioFormat(format model.AudioFormat) error {
	hasSupportedRate := format.Rate == 0 || format.Rate == InputSampleRate
	if format.Type == "audio/pcm" && hasSupportedRate {
		return nil
	}

	return fmt.Errorf(
		"%w: the Gemini Live transcription API only supports 16kHz 16-bit PCM input audio",
		model.ErrInvalidArgument,
	)
}

// Setup is the setup of a Live transcription session: the model resource
// and Google's AudioTranscriptionConfig, sent as {} when no option is set.
type Setup[Config any] struct {
	Model                   string `json:"model"`
	InputAudioTranscription Config `json:"inputAudioTranscription"`
}

// setupMessage is the first client message.
type setupMessage struct {
	Setup json.RawMessage `json:"setup"`
}

// EncodedSetup is a setup as the request body and as the setup message.
type EncodedSetup struct {
	// Body is the setup, recorded as the request body.
	Body json.RawMessage

	// Message wraps the setup as the first client message.
	Message []byte
}

// EncodeSetup encodes the setup of a session.
func EncodeSetup[Config any](setup Setup[Config]) (EncodedSetup, error) {
	body, err := json.Marshal(setup)
	if err != nil {
		return EncodedSetup{}, fmt.Errorf("encode gemini live setup: %w", err)
	}

	message, err := json.Marshal(setupMessage{
		Setup: body,
	})
	if err != nil {
		return EncodedSetup{}, fmt.Errorf("encode gemini live setup: %w", err)
	}

	return EncodedSetup{
		Body:    body,
		Message: message,
	}, nil
}
