// Package anthropic provides an Anthropic Messages API provider for the AI SDK.
package anthropic

import (
	"os"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider"
)

// Compile-time check that Provider implements provider.Provider.
var _ provider.Provider = (*Provider)(nil)

// Provider is the Anthropic provider factory for dynamic provider
// selection via a Registry. It captures provider-level options once
// and creates models from them.
type Provider struct {
	clientConfig *ClientConfig
}

// New creates an Anthropic provider with the given options. If no API
// key is provided via WithAPIKey, the ANTHROPIC_API_KEY environment
// variable is used.
func New(opts ...Option) *Provider {
	cfg := config{
		baseURL:    defaultBaseURL,
		apiKey:     os.Getenv("ANTHROPIC_API_KEY"),
		apiVersion: defaultVersion,
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

// Embedding returns nil because Anthropic does not provide native
// embedding models through this provider.
func (p *Provider) Embedding(_ string) model.EmbeddingModel {
	return nil
}

// Chat creates a language model for the Anthropic Messages API. If no
// API key is provided via WithAPIKey, the ANTHROPIC_API_KEY
// environment variable is used.
func Chat(modelID string, opts ...Option) model.Model {
	cfg := config{
		baseURL:    defaultBaseURL,
		apiKey:     os.Getenv("ANTHROPIC_API_KEY"),
		apiVersion: defaultVersion,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &ChatModel{
		modelID:      modelID,
		clientConfig: buildClientConfig(&cfg),
	}
}

func buildClientConfig(cfg *config) *ClientConfig {
	return &ClientConfig{
		BaseURL:    cfg.baseURL,
		APIKey:     cfg.apiKey,
		APIVersion: cfg.apiVersion,
		HTTPClient: cfg.httpClient,
	}
}
