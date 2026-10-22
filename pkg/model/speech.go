package model

import (
	"context"
	"encoding/json"
	"net/http"
)

// SpeechModel converts text into audio.
type SpeechModel interface {
	Provider() string
	ModelID() string

	Generate(ctx context.Context, options SpeechOptions) (*SpeechResult, error)
}

// SpeechOptions are the inputs of one speech call. Empty values leave the
// provider default.
type SpeechOptions struct {
	Text string

	Voice        string
	OutputFormat string
	Instructions string
	Speed        *float64
	Language     string

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// SpeechResult is the outcome of one speech call.
type SpeechResult struct {
	Audio []byte

	// Usage is the provider's native usage object, if any.
	Usage json.RawMessage

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
	Request          RequestInfo
	Response         ResponseInfo
}
