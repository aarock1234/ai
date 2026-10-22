package openai

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// embeddingMaxValuesPerCall is the most inputs one embeddings request accepts.
const embeddingMaxValuesPerCall = 2048

// EmbeddingOptions are the embedding settings read from the "openai"
// provider options namespace.
type EmbeddingOptions struct {
	// Dimensions sets the size of the output vectors. It overrides
	// EmbedOptions.Dimensions and needs text-embedding-3 or a later model.
	Dimensions *int `json:"dimensions,omitempty"`

	// User identifies the end user for abuse monitoring.
	User string `json:"user,omitempty"`
}

// EmbeddingModel is an OpenAI text embedding model.
type EmbeddingModel struct {
	modelID string
	config  config
}

func newEmbeddingModel(modelID string, c config) *EmbeddingModel {
	return &EmbeddingModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as "openai.embedding".
func (m *EmbeddingModel) Provider() string {
	return m.config.modelProvider("embedding")
}

// ModelID returns the model identifier.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns 2048.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	return embeddingMaxValuesPerCall
}

// SupportsParallelCalls reports that batches may be sent concurrently.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return true
}

// embeddingRequest is the body of an embeddings request.
type embeddingRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format"`
	Dimensions     *int     `json:"dimensions,omitempty"`
	User           string   `json:"user,omitempty"`
}

// embeddingResponse is an embeddings response.
type embeddingResponse struct {
	Data  []embeddingData `json:"data"`
	Usage *embeddingUsage `json:"usage"`
}

// embeddingData is one embedding vector.
type embeddingData struct {
	Embedding []float64 `json:"embedding"`
}

// embeddingUsage is the token usage of an embeddings call.
type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
}

// Embed converts values into vectors in one request.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	if len(options.Values) > embeddingMaxValuesPerCall {
		return nil, fmt.Errorf(
			"%w: %s model %q accepts at most %d values per call, got %d",
			model.ErrInvalidArgument,
			m.Provider(),
			m.modelID,
			embeddingMaxValuesPerCall,
			len(options.Values),
		)
	}

	openaiOptions, _, err := model.ProviderValue[EmbeddingOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	dimensions := options.Dimensions
	if openaiOptions.Dimensions != nil {
		dimensions = openaiOptions.Dimensions
	}

	request, err := m.config.request(ctx, "/embeddings", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, embeddingRequest{
		Model:          m.modelID,
		Input:          options.Values,
		EncodingFormat: "float",
		Dimensions:     dimensions,
		User:           openaiOptions.User,
	})
	if err != nil {
		return nil, err
	}

	var decoded embeddingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if err := decoded.validate(); err != nil {
		return nil, err
	}

	return &model.EmbedResult{
		Embeddings: embeddingVectors(decoded.Data),
		Usage:      embeddingResultUsage(decoded.Usage),
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// validate checks that the response has a vector for every data item.
func (r embeddingResponse) validate() error {
	if r.Data == nil {
		return fmt.Errorf("%w: response has no data", model.ErrInvalidResponse)
	}

	for index, item := range r.Data {
		if item.Embedding == nil {
			return fmt.Errorf("%w: data item %d has no embedding", model.ErrInvalidResponse, index)
		}
	}

	return nil
}

func embeddingVectors(data []embeddingData) [][]float64 {
	vectors := make([][]float64, 0, len(data))
	for _, item := range data {
		vectors = append(vectors, item.Embedding)
	}

	return vectors
}

func embeddingResultUsage(usage *embeddingUsage) *model.EmbeddingUsage {
	if usage == nil {
		return nil
	}

	return &model.EmbeddingUsage{
		Tokens: usage.PromptTokens,
	}
}
