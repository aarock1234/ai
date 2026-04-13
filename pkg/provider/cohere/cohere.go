// Package cohere provides Cohere reranking models for the AI SDK.
package cohere

import (
	"net/http"
	"os"

	"github.com/aarock1234/ai/pkg/model"
)

// Provider captures Cohere provider settings.
type Provider struct {
	clientConfig *clientConfig
}

// New creates a Cohere provider with the given options.
func New(opts ...Option) *Provider {
	cfg := config{
		baseURL: defaultBaseURL,
		apiKey:  os.Getenv("COHERE_API_KEY"),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &Provider{
		clientConfig: buildClientConfig(&cfg),
	}
}

// Rerank creates a reranking model for the given model ID.
func (p *Provider) Rerank(modelID string) model.RerankingModel {
	return &RerankModel{
		modelID:      modelID,
		clientConfig: p.clientConfig,
	}
}

// Rerank creates a reranking model for the given model ID.
func Rerank(modelID string, opts ...Option) model.RerankingModel {
	cfg := config{
		baseURL: defaultBaseURL,
		apiKey:  os.Getenv("COHERE_API_KEY"),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &RerankModel{
		modelID:      modelID,
		clientConfig: buildClientConfig(&cfg),
	}
}

type clientConfig struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

func buildClientConfig(cfg *config) *clientConfig {
	return &clientConfig{
		baseURL:    cfg.baseURL,
		apiKey:     cfg.apiKey,
		httpClient: cfg.httpClient,
	}
}

func (c *clientConfig) client() *http.Client {
	if c.httpClient != nil {
		return c.httpClient
	}

	return http.DefaultClient
}
