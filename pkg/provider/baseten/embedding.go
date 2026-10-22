package baseten

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

const (
	// defaultEmbeddingModelID is the model ID of dedicated deployments,
	// which ignore it.
	defaultEmbeddingModelID = "embeddings"

	// maxEmbeddingsPerCall is Baseten's per-request input limit. Larger
	// batches are rejected.
	maxEmbeddingsPerCall = 128
)

// PerformanceClient embeds values with Baseten's performance client, which
// batches and hedges requests itself.
type PerformanceClient interface {
	Embed(ctx context.Context, values []string, modelID string) (*PerformanceEmbedResponse, error)
}

// PerformanceClientFactory creates a performance client for a deployment's
// "/sync" URL and API key.
type PerformanceClientFactory func(baseURL string, apiKey string) (PerformanceClient, error)

// PerformanceEmbedResponse is the embedding response of a performance
// client.
type PerformanceEmbedResponse struct {
	Data  []PerformanceEmbedding `json:"data"`
	Usage *PerformanceUsage      `json:"usage,omitempty"`
}

// PerformanceEmbedding is one embedding of a performance client response.
type PerformanceEmbedding struct {
	Embedding []float64 `json:"embedding"`
}

// PerformanceUsage is the usage of a performance client response.
type PerformanceUsage struct {
	TotalTokens *int `json:"total_tokens,omitempty"`
}

// EmbeddingModel is a Baseten embedding model served by a dedicated
// deployment.
type EmbeddingModel struct {
	modelID  string
	provider string

	// embedding sends requests over HTTP when performance is nil.
	embedding   *openaicompatible.EmbeddingModel
	performance PerformanceClient
}

// Embedding returns an embedding model of the deployment set with
// WithModelURL, which must be a "/sync" or "/sync/v1" URL. An empty modelID
// uses a default.
func (p *Provider) Embedding(modelID string) (*EmbeddingModel, error) {
	if p.modelURL == "" {
		return nil, fmt.Errorf("%w: no model URL provided for embeddings, use WithModelURL", model.ErrInvalidArgument)
	}

	if !strings.Contains(p.modelURL, "/sync") {
		return nil, fmt.Errorf("%w: you must use a /sync or /sync/v1 endpoint for embeddings", model.ErrUnsupported)
	}

	if modelID == "" {
		modelID = defaultEmbeddingModelID
	}

	embedding := openaicompatible.NewEmbeddingModel(modelID, openaicompatible.EmbeddingConfig{
		Provider:             Namespace + ".embedding",
		BuildURL:             openaicompatible.InfallibleURL(p.embeddingURL),
		Headers:              p.requestHeaders,
		HTTPClient:           p.httpClient,
		ParseError:           parseError,
		MaxEmbeddingsPerCall: maxEmbeddingsPerCall,
	})

	var performance PerformanceClient
	if p.performanceClient != nil {
		client, err := p.newPerformanceClient()
		if err != nil {
			return nil, err
		}

		performance = client
	}

	return &EmbeddingModel{
		modelID:     modelID,
		provider:    p.name + ".embedding",
		embedding:   embedding,
		performance: performance,
	}, nil
}

// embeddingURL adds "/v1" to "/sync" model URLs.
func (p *Provider) embeddingURL(path string) string {
	if !strings.Contains(p.modelURL, "/sync/v1") {
		return p.modelURL + "/v1" + path
	}

	return p.modelURL + path
}

// newPerformanceClient creates the configured performance client. The client
// appends "/v1" itself, so it receives the "/sync" form of the model URL.
func (p *Provider) newPerformanceClient() (PerformanceClient, error) {
	apiKey, err := p.loadAPIKey()
	if err != nil {
		return nil, err
	}

	client, err := p.performanceClient(strings.Replace(p.modelURL, "/sync/v1", "/sync", 1), apiKey)
	if err != nil {
		return nil, fmt.Errorf("create baseten performance client: %w", err)
	}

	return client, nil
}

// Provider returns the provider identifier, "baseten.embedding" by default.
func (m *EmbeddingModel) Provider() string {
	return m.provider
}

// ModelID returns the model identifier.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns 128 over HTTP, and zero, meaning no limit,
// with a performance client.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	if m.performance != nil {
		return 0
	}

	return m.embedding.MaxEmbeddingsPerCall()
}

// SupportsParallelCalls reports that batches may be sent concurrently.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return m.embedding.SupportsParallelCalls()
}

// Embed embeds values over HTTP or with the performance client.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	if m.performance == nil {
		return m.embedding.Embed(ctx, options)
	}

	response, err := m.performance.Embed(ctx, options.Values, m.modelID)
	if err != nil {
		return nil, err
	}

	if response == nil {
		return nil, fmt.Errorf("%w: performance client returned no response", model.ErrInvalidResponse)
	}

	body, err := json.Marshal(response)
	if err != nil {
		return nil, fmt.Errorf("encode performance client response: %w", err)
	}

	embeddings := make([][]float64, 0, len(response.Data))
	for _, item := range response.Data {
		embeddings = append(embeddings, item.Embedding)
	}

	result := &model.EmbedResult{
		Embeddings: embeddings,
		Response: model.ResponseInfo{
			Body: body,
		},
	}

	if response.Usage != nil && response.Usage.TotalTokens != nil {
		result.Usage = &model.EmbeddingUsage{
			Tokens: *response.Usage.TotalTokens,
		}
	}

	return result, nil
}

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)
