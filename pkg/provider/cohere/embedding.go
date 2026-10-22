package cohere

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// embeddingPath is the embed endpoint.
	embeddingPath = "/embed"

	// maxEmbeddingsPerCall is Cohere's batch limit.
	maxEmbeddingsPerCall = 96
)

// EmbeddingModel is a Cohere embedding model.
type EmbeddingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "cohere.textEmbedding" by default.
func (m *EmbeddingModel) Provider() string {
	return m.provider.name + ".textEmbedding"
}

// ModelID returns the model identifier.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns Cohere's batch limit of 96 values.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	return maxEmbeddingsPerCall
}

// SupportsParallelCalls reports that batches may be sent concurrently.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return true
}

type embeddingRequest struct {
	Model           string          `json:"model"`
	EmbeddingTypes  []EmbeddingType `json:"embedding_types"`
	Texts           []string        `json:"texts"`
	InputType       InputType       `json:"input_type"`
	Truncate        Truncate        `json:"truncate,omitempty"`
	OutputDimension *int            `json:"output_dimension,omitempty"`
}

type embeddingResponse struct {
	// Embeddings holds the vectors of each requested type.
	Embeddings map[EmbeddingType][][]float64 `json:"embeddings"`

	Meta embeddingMeta `json:"meta"`
}

type embeddingMeta struct {
	BilledUnits billedUnits `json:"billed_units"`
}

type billedUnits struct {
	InputTokens int `json:"input_tokens"`
}

// Embed embeds values in one request.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	embeddingOptions, err := readEmbeddingOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	if len(options.Values) > maxEmbeddingsPerCall {
		return nil, fmt.Errorf(
			"%w: too many values for a single embedding call: %s %s accepts %d, got %d",
			model.ErrInvalidArgument,
			m.Provider(),
			m.modelID,
			maxEmbeddingsPerCall,
			len(options.Values),
		)
	}

	embeddingType := embeddingOptions.EmbeddingType
	if embeddingType == "" {
		embeddingType = EmbeddingFloat
	}

	inputType := embeddingOptions.InputType
	if inputType == "" {
		inputType = InputSearchQuery
	}

	dimension := options.Dimensions
	if embeddingOptions.OutputDimension != nil {
		dimension = embeddingOptions.OutputDimension
	}

	request, err := m.provider.apiRequest(embeddingPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, embeddingRequest{
		Model: m.modelID,
		EmbeddingTypes: []EmbeddingType{
			embeddingType,
		},
		Texts:           options.Values,
		InputType:       inputType,
		Truncate:        embeddingOptions.Truncate,
		OutputDimension: dimension,
	})
	if err != nil {
		return nil, err
	}

	var decoded embeddingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	embeddings, ok := decoded.Embeddings[embeddingType]
	if !ok {
		return nil, fmt.Errorf("%w: response has no %s embeddings", model.ErrInvalidResponse, embeddingType)
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Usage: &model.EmbeddingUsage{
			Tokens: decoded.Meta.BilledUnits.InputTokens,
		},
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)
