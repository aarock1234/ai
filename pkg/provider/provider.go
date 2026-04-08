// Package provider defines the factory interface for dynamic provider
// selection. Consumers who need to choose a provider at runtime (e.g.
// from configuration) use Provider and Registry. Most consumers will
// use provider-specific constructors (openai.Chat, openrouter.Chat)
// directly instead.
package provider

import "github.com/aarock1234/ai/pkg/model"

// Provider is a factory that creates model instances for a given
// provider. Implementations return concrete model.Model and
// model.EmbeddingModel values configured with the provider's
// credentials and settings.
type Provider interface {
	// Chat creates a language model for the given model ID.
	Chat(modelID string) model.Model

	// Embedding creates an embedding model for the given model ID.
	// Returns nil if the provider does not support embeddings.
	Embedding(modelID string) model.EmbeddingModel
}
