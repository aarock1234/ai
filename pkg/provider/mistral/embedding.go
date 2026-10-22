package mistral

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// embeddingPath is the embeddings endpoint.
	embeddingPath = "/embeddings"

	// maxEmbeddingsPerCall is Mistral's batch limit.
	maxEmbeddingsPerCall = 32
)

// EmbeddingModel is a Mistral embedding model.
type EmbeddingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "mistral.embedding" by default.
func (m *EmbeddingModel) Provider() string {
	return m.provider.name + ".embedding"
}

// ModelID returns the model identifier.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns Mistral's batch limit of 32 values.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	return maxEmbeddingsPerCall
}

// SupportsParallelCalls reports that batches must be sent one at a time.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return false
}

type embeddingRequest struct {
	Model           string                     `json:"model"`
	Input           []string                   `json:"input"`
	Metadata        map[string]json.RawMessage `json:"metadata,omitempty"`
	OutputDimension *int                       `json:"output_dimension,omitempty"`
	OutputDtype     EmbeddingDataType          `json:"output_dtype,omitempty"`
	EncodingFormat  string                     `json:"encoding_format"`
}

type embeddingResponse struct {
	Data  []embeddingData `json:"data"`
	Usage *embeddingUsage `json:"usage"`
}

type embeddingData struct {
	Embedding []float64 `json:"embedding"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
}

// Embed embeds values in one request.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
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

	embeddingOptions, err := readEmbeddingOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
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
		Model:           m.modelID,
		Input:           options.Values,
		Metadata:        embeddingOptions.Metadata,
		OutputDimension: dimension,
		OutputDtype:     embeddingOptions.OutputDtype,
		EncodingFormat:  "float",
	})
	if err != nil {
		return nil, err
	}

	var decoded embeddingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	embeddings := make([][]float64, 0, len(decoded.Data))
	for _, item := range decoded.Data {
		embeddings = append(embeddings, item.Embedding)
	}

	var usage *model.EmbeddingUsage
	if decoded.Usage != nil {
		usage = &model.EmbeddingUsage{
			Tokens: decoded.Usage.PromptTokens,
		}
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Usage:      usage,
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)
