package gateway

import (
	"context"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// embeddingModelPath is the embedding model endpoint.
	embeddingModelPath = "/embedding-model"

	// maxEmbeddingsPerCall is the gateway's batch limit.
	maxEmbeddingsPerCall = 2048
)

// EmbeddingModel is an AI Gateway embedding model. It is safe for concurrent
// use.
type EmbeddingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "gateway" by default.
func (m *EmbeddingModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model identifier.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns the gateway's batch limit of 2048 values.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	return maxEmbeddingsPerCall
}

// SupportsParallelCalls reports that batches may be sent concurrently.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return true
}

type embeddingRequest struct {
	Values          []string              `json:"values"`
	Dimensions      *int                  `json:"dimensions,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type embeddingResponse struct {
	Embeddings       [][]float64     `json:"embeddings"`
	Usage            *embeddingUsage `json:"usage"`
	Warnings         []wireWarning   `json:"warnings"`
	ProviderMetadata wireMetadata    `json:"providerMetadata"`
}

type embeddingUsage struct {
	Tokens int `json:"tokens"`
}

// Embed embeds values in one request.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	request, method, err := m.provider.modelRequest(embeddingModelPath, options.Headers, modelConfigHeaders("embedding", m.modelID))
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, embeddingRequest{
		Values:          options.Values,
		Dimensions:      options.Dimensions,
		ProviderOptions: options.ProviderOptions,
	})
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var decoded embeddingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	var usage *model.EmbeddingUsage
	if decoded.Usage != nil {
		usage = &model.EmbeddingUsage{
			Tokens: decoded.Usage.Tokens,
		}
	}

	return &model.EmbedResult{
		Embeddings:       decoded.Embeddings,
		Usage:            usage,
		ProviderMetadata: decoded.ProviderMetadata.providerMetadata(),
		Warnings:         decodeWarnings(decoded.Warnings),
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// modelConfigHeaders describe a call of the given model kind, such as
// "embedding", to the gateway.
func modelConfigHeaders(kind string, modelID string) http.Header {
	return http.Header{
		http.CanonicalHeaderKey("ai-" + kind + "-model-specification-version"): {
			"4",
		},
		"Ai-Model-Id": {
			modelID,
		},
	}
}

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)
