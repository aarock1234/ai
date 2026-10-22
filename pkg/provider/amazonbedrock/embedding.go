package amazonbedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
)

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)

const (
	// cohereMaxEmbeddingsPerCall is Cohere's batch limit; Titan and Nova
	// embed one value per call.
	cohereMaxEmbeddingsPerCall = 96

	// inputTokenCountHeader reports the input tokens of Cohere embeddings.
	inputTokenCountHeader = "X-Amzn-Bedrock-Input-Token-Count"
)

// EmbeddingModel is an Amazon Bedrock embedding model: Titan, Cohere or
// Nova.
type EmbeddingModel struct {
	modelID  string
	provider *Provider
	family   ModelFamily
}

// EmbeddingModelOption configures an EmbeddingModel.
type EmbeddingModelOption func(*EmbeddingModel)

// WithEmbeddingModelFamily sets the family of a model whose ID does not
// reveal it, such as an application inference profile ARN:
// ModelFamilyTitan, ModelFamilyCohere or ModelFamilyNova.
func WithEmbeddingModelFamily(family ModelFamily) EmbeddingModelOption {
	return func(m *EmbeddingModel) {
		m.family = family
	}
}

// Provider returns the provider identifier, "amazon-bedrock" by default.
func (m *EmbeddingModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model ID.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns 96 for Cohere models and 1 for others.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	if m.modelFamily() == ModelFamilyCohere {
		return cohereMaxEmbeddingsPerCall
	}

	return 1
}

// SupportsParallelCalls reports that calls may run in parallel.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return true
}

// modelFamily returns the configured family, or the family detected from
// the model ID, including cross-region inference profile IDs.
func (m *EmbeddingModel) modelFamily() ModelFamily {
	if m.family != "" {
		return m.family
	}

	switch {
	case strings.Contains(m.modelID, "amazon.nova-") && strings.Contains(m.modelID, "embed"):
		return ModelFamilyNova
	case strings.Contains(m.modelID, "cohere.embed-"):
		return ModelFamilyCohere
	default:
		return ModelFamilyTitan
	}
}

type titanEmbeddingRequest struct {
	InputText  string `json:"inputText"`
	Dimensions *int   `json:"dimensions,omitempty"`
	Normalize  *bool  `json:"normalize,omitempty"`
}

type cohereEmbeddingRequest struct {
	InputType       EmbeddingInputType `json:"input_type"`
	Texts           []string           `json:"texts"`
	Truncate        *Truncation        `json:"truncate,omitempty"`
	OutputDimension *int               `json:"output_dimension,omitempty"`
}

type novaEmbeddingRequest struct {
	TaskType              string              `json:"taskType"`
	SingleEmbeddingParams novaEmbeddingParams `json:"singleEmbeddingParams"`
}

type novaEmbeddingParams struct {
	EmbeddingPurpose   EmbeddingPurpose  `json:"embeddingPurpose"`
	EmbeddingDimension int               `json:"embeddingDimension"`
	Text               novaEmbeddingText `json:"text"`
}

type novaEmbeddingText struct {
	TruncationMode Truncation `json:"truncationMode"`
	Value          string     `json:"value"`
}

// embeddingResponse covers the Titan, Nova and Cohere response shapes.
type embeddingResponse struct {
	Embedding           []float64       `json:"embedding"`
	InputTextTokenCount *int            `json:"inputTextTokenCount"`
	Embeddings          json.RawMessage `json:"embeddings"`
	InputTokenCount     *int            `json:"inputTokenCount"`
}

type novaEmbedding struct {
	EmbeddingType string    `json:"embeddingType"`
	Embedding     []float64 `json:"embedding"`
}

type cohereFloatEmbeddings struct {
	Float [][]float64 `json:"float"`
}

// Embed embeds values with one InvokeModel call.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	if len(options.Values) > m.MaxEmbeddingsPerCall() {
		return nil, fmt.Errorf(
			"%w: too many values for a single embedding call: %s %s accepts %d, got %d",
			model.ErrInvalidArgument,
			m.Provider(),
			m.modelID,
			m.MaxEmbeddingsPerCall(),
			len(options.Values),
		)
	}

	embeddingOptions, err := readEmbeddingOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(ctx, bedrock.Runtime, bedrock.ModelPath(m.modelID, "invoke"), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, m.requestBody(options, embeddingOptions))
	if err != nil {
		return nil, err
	}

	var decoded embeddingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	embeddings, err := decoded.vectors()
	if err != nil {
		return nil, err
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Usage:      decoded.usage(response.Headers),
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// requestBody returns the payload of the model family.
func (m *EmbeddingModel) requestBody(options model.EmbedOptions, embeddingOptions EmbeddingOptions) any {
	first := ""
	if len(options.Values) > 0 {
		first = options.Values[0]
	}

	switch m.modelFamily() {
	case ModelFamilyNova:
		dimension := embeddingOptions.EmbeddingDimension
		if dimension == nil {
			dimension = options.Dimensions
		}

		return novaEmbeddingRequest{
			TaskType: "SINGLE_EMBEDDING",
			SingleEmbeddingParams: novaEmbeddingParams{
				EmbeddingPurpose:   optional.Or(embeddingOptions.EmbeddingPurpose, EmbeddingGenericIndex),
				EmbeddingDimension: optional.Or(dimension, 1024),
				Text: novaEmbeddingText{
					TruncationMode: optional.Or(embeddingOptions.Truncate, TruncateEnd),
					Value:          first,
				},
			},
		}

	case ModelFamilyCohere:
		dimension := embeddingOptions.OutputDimension
		if dimension == nil {
			dimension = options.Dimensions
		}

		// Cohere models on Bedrock require an input type.
		return cohereEmbeddingRequest{
			InputType:       optional.Or(embeddingOptions.InputType, EmbeddingSearchQuery),
			Texts:           options.Values,
			Truncate:        embeddingOptions.Truncate,
			OutputDimension: dimension,
		}

	default:
		dimension := embeddingOptions.Dimensions
		if dimension == nil {
			dimension = options.Dimensions
		}

		return titanEmbeddingRequest{
			InputText:  first,
			Dimensions: dimension,
			Normalize:  embeddingOptions.Normalize,
		}
	}
}

// vectors returns the embeddings of a Titan, Nova, Cohere v3 or Cohere v4
// response.
func (r embeddingResponse) vectors() ([][]float64, error) {
	if r.Embedding != nil {
		return [][]float64{
			r.Embedding,
		}, nil
	}

	trimmed := bytes.TrimSpace(r.Embeddings)
	if len(trimmed) == 0 {
		return nil, errMissingField("embeddings")
	}

	if trimmed[0] != '[' {
		var cohere cohereFloatEmbeddings
		if err := json.Unmarshal(trimmed, &cohere); err != nil || cohere.Float == nil {
			return nil, fmt.Errorf("%w: unrecognized embeddings", model.ErrInvalidResponse)
		}

		return cohere.Float, nil
	}

	var nova []novaEmbedding
	if err := json.Unmarshal(trimmed, &nova); err == nil && len(nova) > 0 && nova[0].EmbeddingType != "" {
		return [][]float64{
			nova[0].Embedding,
		}, nil
	}

	var vectors [][]float64
	if err := json.Unmarshal(trimmed, &vectors); err != nil {
		return nil, fmt.Errorf("%w: unrecognized embeddings: %w", model.ErrInvalidResponse, err)
	}

	return vectors, nil
}

// usage returns the Titan or Nova token count, or the count Cohere reports
// in a header.
func (r embeddingResponse) usage(headers http.Header) *model.EmbeddingUsage {
	tokens := r.InputTextTokenCount
	if r.Embedding == nil {
		tokens = r.InputTokenCount
	}

	if tokens == nil {
		count, err := strconv.Atoi(headers.Get(inputTokenCountHeader))
		if err != nil {
			return nil
		}

		tokens = &count
	}

	return &model.EmbeddingUsage{
		Tokens: *tokens,
	}
}
