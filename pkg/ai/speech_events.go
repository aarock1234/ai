package ai

import (
	"encoding/json"
	"net/http"

	"github.com/aarock1234/ai/pkg/model"
)

// GenerateSpeechStartEvent describes a GenerateSpeech call before the model
// is called.
type GenerateSpeechStartEvent struct {
	CallID string

	// OperationID is "ai.generateSpeech".
	OperationID string

	Provider        string
	ModelID         string
	Text            string
	Voice           string
	OutputFormat    string
	Instructions    string
	Speed           *float64
	Language        string
	MaxRetries      int
	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// GenerateSpeechEndEvent describes a successful GenerateSpeech call. The
// audio itself is left out; only its size, media type and format are
// reported.
type GenerateSpeechEndEvent struct {
	CallID      string
	OperationID string
	Provider    string
	ModelID     string
	Text        string

	AudioByteLength int
	AudioMediaType  string
	AudioFormat     string

	// Usage is the provider's native usage object, if any.
	Usage json.RawMessage

	Warnings         []model.Warning
	ProviderMetadata model.ProviderMetadata

	// Response is the response metadata and headers, without the body.
	Response model.ResponseInfo
}
