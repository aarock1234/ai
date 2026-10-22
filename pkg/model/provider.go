package model

import "fmt"

// Kind names a kind of model, for errors and registries.
type Kind string

// Model kinds.
const (
	LanguageModelKind      Kind = "language model"
	EmbeddingModelKind     Kind = "embedding model"
	ImageModelKind         Kind = "image model"
	TranscriptionModelKind Kind = "transcription model"
	SpeechModelKind        Kind = "speech model"
	RerankingModelKind     Kind = "reranking model"
)

// Provider creates models by ID. Methods for model types the provider does
// not offer return an error wrapping ErrNoSuchModel.
type Provider interface {
	LanguageModel(modelID string) (LanguageModel, error)
	EmbeddingModel(modelID string) (EmbeddingModel, error)
	ImageModel(modelID string) (ImageModel, error)
	TranscriptionModel(modelID string) (TranscriptionModel, error)
	SpeechModel(modelID string) (SpeechModel, error)
	RerankingModel(modelID string) (RerankingModel, error)
}

// NoSuchModelError returns an error wrapping ErrNoSuchModel.
func NoSuchModelError(provider string, kind Kind, modelID string) error {
	return fmt.Errorf("%w: %s %s %q", ErrNoSuchModel, provider, kind, modelID)
}
