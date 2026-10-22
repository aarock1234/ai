package ai

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// TranscriptionStartEvent describes a Transcribe or StreamTranscribe call
// before the model is called. The audio itself is left out.
type TranscriptionStartEvent struct {
	CallID string

	// OperationID is "ai.transcribe" or "ai.streamTranscribe".
	OperationID string

	Provider string
	ModelID  string

	// AudioByteLength is the size of the audio. It is nil for streams, whose
	// size is reported by the end event, and when the audio could not be
	// loaded.
	AudioByteLength *int

	// AudioMediaType is empty when the audio could not be loaded.
	AudioMediaType string

	// InputAudioFormat is the raw audio format of a stream, and nil for
	// Transcribe.
	InputAudioFormat *model.AudioFormat

	// MaxRetries is nil for streams, which are not retried.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// TranscriptionEndEvent describes a successful Transcribe or
// StreamTranscribe call. The audio itself is left out.
type TranscriptionEndEvent struct {
	CallID      string
	OperationID string
	Provider    string
	ModelID     string

	AudioByteLength int
	AudioMediaType  string

	Text     string
	Segments []model.TranscriptionSegment

	// Language is the detected language, if reported.
	Language string

	// Duration is the length of the audio, or zero when unknown.
	Duration time.Duration

	// Usage is the provider's native usage object, if any.
	Usage json.RawMessage

	Warnings         []model.Warning
	ProviderMetadata model.ProviderMetadata
	Response         model.ResponseInfo
}
