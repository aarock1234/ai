package perplexity

import (
	"context"
	"encoding/base64"
	"fmt"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// embeddingPath is the embeddings endpoint.
const embeddingPath = "/v1/embeddings"

// maxEmbeddingsPerCall is the batch limit of the embeddings API.
const maxEmbeddingsPerCall = 512

// EmbeddingModel is a Perplexity embedding model. Perplexity returns
// quantized embeddings, which are decoded into numbers: signed int8 values,
// or packed bits as bytes from 0 to 255.
type EmbeddingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "perplexity.embedding" by
// default.
func (m *EmbeddingModel) Provider() string {
	return m.provider.name + ".embedding"
}

// ModelID returns the model identifier.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns 512.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	return maxEmbeddingsPerCall
}

// SupportsParallelCalls reports that batches may be sent concurrently.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return true
}

type embeddingRequest struct {
	Model          string         `json:"model"`
	Input          []string       `json:"input"`
	Dimensions     *int           `json:"dimensions,omitempty"`
	EncodingFormat EncodingFormat `json:"encoding_format"`
}

type embeddingResponse struct {
	Data  []embeddingData `json:"data"`
	Usage *embeddingUsage `json:"usage"`
}

type embeddingData struct {
	Embedding string `json:"embedding"`
}

type embeddingUsage struct {
	PromptTokens int            `json:"prompt_tokens"`
	Cost         *embeddingCost `json:"cost"`
}

type embeddingCost struct {
	InputCost *float64 `json:"input_cost"`
	TotalCost *float64 `json:"total_cost"`
	Currency  *string  `json:"currency"`
}

// Embed converts the values into vectors.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	if len(options.Values) > maxEmbeddingsPerCall {
		return nil, fmt.Errorf(
			"%w: too many values for a single embedding call: %s model %q accepts %d values, got %d",
			model.ErrInvalidArgument,
			m.Provider(),
			m.modelID,
			maxEmbeddingsPerCall,
			len(options.Values),
		)
	}

	embeddingOptions, err := readEmbeddingOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	encodingFormat := embeddingOptions.EncodingFormat
	if encodingFormat == "" {
		encodingFormat = EncodingBase64Int8
	}

	dimensions := embeddingOptions.Dimensions
	if dimensions == nil {
		dimensions = options.Dimensions
	}

	request, err := m.provider.apiRequest(embeddingPath, options.Headers)
	if err != nil {
		return nil, err
	}

	body := embeddingRequest{
		Model:          m.modelID,
		Input:          options.Values,
		Dimensions:     dimensions,
		EncodingFormat: encodingFormat,
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded embeddingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	embeddings := make([][]float64, 0, len(decoded.Data))
	for _, item := range decoded.Data {
		embedding, err := decodeEmbedding(item.Embedding, encodingFormat)
		if err != nil {
			return nil, err
		}

		embeddings = append(embeddings, embedding)
	}

	result := &model.EmbedResult{
		Embeddings: embeddings,
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.Usage != nil {
		result.Usage = &model.EmbeddingUsage{
			Tokens: decoded.Usage.PromptTokens,
		}

		result.ProviderMetadata = embeddingMetadata(decoded.Usage.Cost)
	}

	return result, nil
}

// decodeEmbedding decodes a base64 embedding: signed int8 values, or the
// raw bytes of packed bits.
func decodeEmbedding(encoded string, encodingFormat EncodingFormat) ([]float64, error) {
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("%w: decode embedding: %w", model.ErrInvalidResponse, err)
	}

	embedding := make([]float64, len(raw))
	for index, value := range raw {
		if encodingFormat == EncodingBase64Int8 {
			embedding[index] = float64(int8(value))
		} else {
			embedding[index] = float64(value)
		}
	}

	return embedding, nil
}

func embeddingMetadata(cost *embeddingCost) model.ProviderMetadata {
	if cost == nil {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: EmbeddingMetadata{
			Cost: EmbeddingCost{
				InputCost: cost.InputCost,
				TotalCost: cost.TotalCost,
				Currency:  cost.Currency,
			},
		},
	}
}

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)
