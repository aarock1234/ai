package model

import (
	"context"
	"net/http"
)

// TranscriptionModel is the interface for speech-to-text providers.
type TranscriptionModel interface {
	// Provider returns the provider name (e.g. "openai").
	Provider() string

	// ModelID returns the model identifier.
	ModelID() string

	// DoTranscribe performs a transcription request.
	DoTranscribe(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error)
}

// SpeechModel is the interface for text-to-speech providers.
type SpeechModel interface {
	// Provider returns the provider name (e.g. "openai").
	Provider() string

	// ModelID returns the model identifier.
	ModelID() string

	// DoGenerateSpeech performs a speech generation request.
	DoGenerateSpeech(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error)
}

// AudioInput is an uploaded audio file.
type AudioInput struct {
	Data     []byte
	MIMEType string
	Filename string
}

// TranscriptionRequest is the normalized request shape for speech-to-text.
type TranscriptionRequest struct {
	Audio           AudioInput
	Language        string
	Prompt          string
	Temperature     *float64
	ResponseFormat  string
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// TranscriptionSegment is a time-aligned transcription segment.
type TranscriptionSegment struct {
	ID      string
	Start   float64
	End     float64
	Text    string
	Speaker string
}

// TranscriptionResponse is the normalized response from a transcription request.
type TranscriptionResponse struct {
	Text            string
	Language        string
	DurationSeconds float64
	Segments        []TranscriptionSegment
	Metadata        ProviderMetadata
}

// SpeechRequest is the normalized request shape for text-to-speech.
type SpeechRequest struct {
	Input           string
	Voice           string
	Instructions    string
	ResponseFormat  string
	Speed           *float64
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// SpeechResponse is the normalized response from a text-to-speech request.
type SpeechResponse struct {
	Audio    []byte
	MIMEType string
	Metadata ProviderMetadata
}
