// Package openrouter provides an OpenRouter language model provider
// for the AI SDK. OpenRouter proxies requests to many LLM providers
// using the OpenAI-compatible API format.
package openrouter

import (
	"net/http"
	"os"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider"
	oai "github.com/aarock1234/ai/pkg/provider/openai"
)

const defaultBaseURL = "https://openrouter.ai/api/v1"

// Compile-time check that Provider implements provider.Provider.
var _ provider.Provider = (*Provider)(nil)

// Provider is the OpenRouter provider factory for dynamic provider
// selection via a Registry. It captures provider-level options once
// and creates models from them.
type Provider struct {
	clientConfig *oai.ClientConfig
}

// New creates an OpenRouter provider with the given options. If no
// API key is provided via WithAPIKey, the OPENROUTER_API_KEY
// environment variable is used.
func New(opts ...Option) *Provider {
	cfg := config{
		baseURL: defaultBaseURL,
		apiKey:  os.Getenv("OPENROUTER_API_KEY"),
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

// Embedding returns nil because OpenRouter does not support embeddings.
func (p *Provider) Embedding(_ string) model.EmbeddingModel {
	return nil
}

// Chat creates a language model for OpenRouter chat completions.
// The model ID uses OpenRouter's provider-prefixed format (e.g.
// "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"). If no API key is
// provided via WithAPIKey, the OPENROUTER_API_KEY environment
// variable is used.
func Chat(modelID string, opts ...Option) model.Model {
	cfg := config{
		baseURL: defaultBaseURL,
		apiKey:  os.Getenv("OPENROUTER_API_KEY"),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &ChatModel{
		modelID:      modelID,
		clientConfig: buildClientConfig(&cfg),
	}
}

// buildClientConfig translates provider config into the shared
// OpenAI-compatible client config, including OpenRouter-specific headers.
func buildClientConfig(cfg *config) *oai.ClientConfig {
	cc := &oai.ClientConfig{
		BaseURL:    cfg.baseURL,
		APIKey:     cfg.apiKey,
		HTTPClient: cfg.httpClient,
	}

	headers := http.Header{}
	if cfg.httpReferer != "" {
		headers.Set("HTTP-Referer", cfg.httpReferer)
	}
	if cfg.siteURL != "" {
		headers.Set("X-Title", cfg.siteURL)
	}
	if len(headers) > 0 {
		cc.ExtraHeaders = headers
	}

	return cc
}
