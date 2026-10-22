package voyage

import (
	"cmp"
	"context"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// embeddingPath is the embeddings endpoint.
	embeddingPath = "/embeddings"

	// maxEmbeddingsPerCall is Voyage's batch limit.
	maxEmbeddingsPerCall = 128
)

// EmbeddingModel is a Voyage embedding model.
type EmbeddingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "voyage.embedding" by default.
func (m *EmbeddingModel) Provider() string {
	return m.provider.name + ".embedding"
}

// ModelID returns the model identifier.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns Voyage's batch limit of 128 values.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	return maxEmbeddingsPerCall
}

// SupportsParallelCalls reports that batches may be sent concurrently.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return true
}

type embeddingRequest struct {
	Input           []string          `json:"input"`
	Model           string            `json:"model"`
	InputType       InputType         `json:"input_type,omitempty"`
	Truncation      *bool             `json:"truncation,omitempty"`
	OutputDimension *int              `json:"output_dimension,omitempty"`
	OutputDtype     EmbeddingDataType `json:"output_dtype,omitempty"`
}

type embeddingResponse struct {
	Data  []embeddingData `json:"data"`
	Model string          `json:"model"`
	Usage *embeddingUsage `json:"usage"`
}

type embeddingData struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type embeddingUsage struct {
	TotalTokens int `json:"total_tokens"`
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

	dimension := options.Dimensions
	if embeddingOptions.OutputDimension != nil {
		dimension = embeddingOptions.OutputDimension
	}

	request, err := m.provider.apiRequest(embeddingPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, embeddingRequest{
		Input:           options.Values,
		Model:           m.modelID,
		InputType:       embeddingOptions.InputType,
		Truncation:      embeddingOptions.Truncation,
		OutputDimension: dimension,
		OutputDtype:     embeddingOptions.OutputDtype,
	})
	if err != nil {
		return nil, err
	}

	var decoded embeddingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	usage := &model.EmbeddingUsage{}
	if decoded.Usage != nil {
		usage.Tokens = decoded.Usage.TotalTokens
	}

	return &model.EmbedResult{
		Embeddings: orderedEmbeddings(decoded.Data),
		Usage:      usage,
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// orderedEmbeddings returns the vectors in input order.
func orderedEmbeddings(data []embeddingData) [][]float64 {
	sorted := slices.SortedStableFunc(slices.Values(data), func(a, b embeddingData) int {
		return cmp.Compare(a.Index, b.Index)
	})

	embeddings := make([][]float64, 0, len(sorted))
	for _, item := range sorted {
		embeddings = append(embeddings, item.Embedding)
	}

	return embeddings
}

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)
