package openai

import (
	"net/http"
	"os"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider"
)

// Compile-time check that Provider implements provider.Provider.
var _ provider.Provider = (*Provider)(nil)

// Provider is the OpenAI provider factory for dynamic provider
// selection via a Registry. It captures provider-level options once
// and creates models from them.
type Provider struct {
	clientConfig *ClientConfig
}

// New creates an OpenAI provider with the given options. If no API
// key is provided via WithAPIKey, the OPENAI_API_KEY environment
// variable is used.
func New(opts ...Option) *Provider {
	cfg := config{
		baseURL: defaultBaseURL,
		apiKey:  os.Getenv("OPENAI_API_KEY"),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &Provider{
		clientConfig: buildClientConfig(&cfg),
	}
}

// Chat creates a language model for the given model ID.
func (p *Provider) Chat(modelID string) model.Model {
	return &ChatModel{
		modelID:      modelID,
		clientConfig: p.clientConfig,
	}
}

// Embedding creates an embedding model for the given model ID.
func (p *Provider) Embedding(modelID string) model.EmbeddingModel {
	return &EmbeddingModel{
		modelID:      modelID,
		clientConfig: p.clientConfig,
	}
}

// Image creates an image model for the given model ID.
func (p *Provider) Image(modelID string) model.ImageModel {
	return &ImageModel{
		modelID:      modelID,
		clientConfig: p.clientConfig,
	}
}

// Transcription creates a transcription model for the given model ID.
func (p *Provider) Transcription(modelID string) model.TranscriptionModel {
	return &TranscriptionModel{
		modelID:      modelID,
		clientConfig: p.clientConfig,
	}
}

// Speech creates a speech model for the given model ID.
func (p *Provider) Speech(modelID string) model.SpeechModel {
	return &SpeechModel{
		modelID:      modelID,
		clientConfig: p.clientConfig,
	}
}

// Chat creates a language model for OpenAI chat completions. The
// model ID should be a valid OpenAI model name (e.g. "gpt-5.4",
// ""). If no API key is provided via WithAPIKey, the
// OPENAI_API_KEY environment variable is used.
func Chat(modelID string, opts ...Option) model.Model {
	cfg := config{
		baseURL: defaultBaseURL,
		apiKey:  os.Getenv("OPENAI_API_KEY"),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &ChatModel{
		modelID:      modelID,
		clientConfig: buildClientConfig(&cfg),
	}
}

// Embedding creates an embedding model for OpenAI embeddings. The
// model ID should be a valid OpenAI embedding model (e.g.
// "text-embedding-3-small"). If no API key is provided via WithAPIKey,
// the OPENAI_API_KEY environment variable is used.
func Embedding(modelID string, opts ...Option) model.EmbeddingModel {
	cfg := config{
		baseURL: defaultBaseURL,
		apiKey:  os.Getenv("OPENAI_API_KEY"),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &EmbeddingModel{
		modelID:      modelID,
		clientConfig: buildClientConfig(&cfg),
	}
}

// Image creates an image model for OpenAI image generation.
func Image(modelID string, opts ...Option) model.ImageModel {
	cfg := config{
		baseURL: defaultBaseURL,
		apiKey:  os.Getenv("OPENAI_API_KEY"),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &ImageModel{
		modelID:      modelID,
		clientConfig: buildClientConfig(&cfg),
	}
}

// Transcription creates a speech-to-text model for OpenAI audio APIs.
func Transcription(modelID string, opts ...Option) model.TranscriptionModel {
	cfg := config{
		baseURL: defaultBaseURL,
		apiKey:  os.Getenv("OPENAI_API_KEY"),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &TranscriptionModel{
		modelID:      modelID,
		clientConfig: buildClientConfig(&cfg),
	}
}

// Speech creates a text-to-speech model for OpenAI audio APIs.
func Speech(modelID string, opts ...Option) model.SpeechModel {
	cfg := config{
		baseURL: defaultBaseURL,
		apiKey:  os.Getenv("OPENAI_API_KEY"),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &SpeechModel{
		modelID:      modelID,
		clientConfig: buildClientConfig(&cfg),
	}
}

// buildClientConfig translates provider config into the shared
// OpenAI-compatible client config.
func buildClientConfig(cfg *config) *ClientConfig {
	cc := &ClientConfig{
		BaseURL:    cfg.baseURL,
		APIKey:     cfg.apiKey,
		HTTPClient: cfg.httpClient,
	}

	if cfg.organization != "" {
		cc.ExtraHeaders = http.Header{
			"OpenAI-Organization": {cfg.organization},
		}
	}

	return cc
}
