package alibaba

import (
	"cmp"
	"context"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// embeddingPath is the native text embedding endpoint.
const embeddingPath = "/services/embeddings/text-embedding/text-embedding"

// maxEmbeddingsPerCall is Alibaba's batch limit.
const maxEmbeddingsPerCall = 10

// EmbeddingModel is an Alibaba text embedding model.
type EmbeddingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "alibaba.embedding" by default.
func (m *EmbeddingModel) Provider() string {
	return m.provider.name + ".embedding"
}

// ModelID returns the model identifier.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns 10.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	return maxEmbeddingsPerCall
}

// SupportsParallelCalls reports that batches are sent one at a time.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return false
}

// EmbeddingMetadata is the provider metadata of an embedding call with
// sparse output, stored under the "alibaba" namespace.
type EmbeddingMetadata struct {
	SparseEmbeddings []SparseEmbedding `json:"sparseEmbeddings"`
}

// SparseEmbedding is the sparse vector of one input text.
type SparseEmbedding struct {
	// TextIndex is the position of the text in the input.
	TextIndex int `json:"textIndex"`

	SparseEmbedding []SparseValue `json:"sparseEmbedding"`
}

// SparseValue is one non-zero dimension of a sparse vector.
type SparseValue struct {
	Index int     `json:"index"`
	Value float64 `json:"value"`

	// Token is the token of the dimension, when reported.
	Token *string `json:"token,omitempty"`
}

type embeddingRequest struct {
	Model      string              `json:"model"`
	Input      embeddingInput      `json:"input"`
	Parameters embeddingParameters `json:"parameters"`
}

type embeddingInput struct {
	Texts []string `json:"texts"`
}

type embeddingParameters struct {
	TextType   TextType            `json:"text_type,omitempty"`
	Dimension  *int                `json:"dimension,omitempty"`
	OutputType EmbeddingOutputType `json:"output_type,omitempty"`
}

type embeddingResponse struct {
	Output embeddingOutput `json:"output"`
	Usage  *embeddingUsage `json:"usage"`
}

type embeddingOutput struct {
	Embeddings []embeddingItem `json:"embeddings"`
}

type embeddingItem struct {
	Embedding       []float64     `json:"embedding"`
	TextIndex       int           `json:"text_index"`
	SparseEmbedding []SparseValue `json:"sparse_embedding"`
}

type embeddingUsage struct {
	TotalTokens int `json:"total_tokens"`
}

// Embed embeds up to 10 values in one request.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	if len(options.Values) > maxEmbeddingsPerCall {
		return nil, fmt.Errorf(
			"%w: too many values for a single embedding call. the %s model %q can only embed up to %d values per call, but %d values were provided",
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

	dimension := embeddingOptions.Dimension
	if dimension == nil {
		dimension = options.Dimensions
	}

	request, err := m.provider.apiRequest(m.provider.embeddingBaseURL+embeddingPath, parseNativeError, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, embeddingRequest{
		Model: m.modelID,
		Input: embeddingInput{
			Texts: options.Values,
		},
		Parameters: embeddingParameters{
			TextType:   embeddingOptions.TextType,
			Dimension:  dimension,
			OutputType: embeddingOptions.OutputType,
		},
	})
	if err != nil {
		return nil, err
	}

	var decoded embeddingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	items := slices.Clone(decoded.Output.Embeddings)
	slices.SortStableFunc(items, func(a, b embeddingItem) int {
		return cmp.Compare(a.TextIndex, b.TextIndex)
	})

	result := &model.EmbedResult{
		Embeddings:       denseEmbeddings(items),
		ProviderMetadata: sparseMetadata(items),
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.Usage != nil {
		result.Usage = &model.EmbeddingUsage{
			Tokens: decoded.Usage.TotalTokens,
		}
	}

	return result, nil
}

func denseEmbeddings(items []embeddingItem) [][]float64 {
	embeddings := make([][]float64, 0, len(items))
	for _, item := range items {
		embeddings = append(embeddings, item.Embedding)
	}

	return embeddings
}

// sparseMetadata reports the sparse vectors, or nil when there are none.
func sparseMetadata(items []embeddingItem) model.ProviderMetadata {
	var sparse []SparseEmbedding
	for _, item := range items {
		if item.SparseEmbedding == nil {
			continue
		}

		sparse = append(sparse, SparseEmbedding{
			TextIndex:       item.TextIndex,
			SparseEmbedding: item.SparseEmbedding,
		})
	}

	if len(sparse) == 0 {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: EmbeddingMetadata{
			SparseEmbeddings: sparse,
		},
	}
}

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)
