package googlevertex

import (
	"context"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/google"
)

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)

const (
	// predictMaxEmbeddingsPerCall is the batch limit of the predict API.
	predictMaxEmbeddingsPerCall = 250

	// embedContentMaxEmbeddingsPerCall is the limit of models that only
	// support the embedContent API.
	embedContentMaxEmbeddingsPerCall = 1
)

// embedContentModels only support the embedContent API, not predict.
var embedContentModels = []string{
	"gemini-embedding-2",
	"gemini-embedding-2-preview",
}

var embeddingTaskTypes = []google.EmbeddingTaskType{
	google.TaskSemanticSimilarity,
	google.TaskClassification,
	google.TaskClustering,
	google.TaskRetrievalDocument,
	google.TaskRetrievalQuery,
	google.TaskQuestionAnswering,
	google.TaskFactVerification,
	google.TaskCodeRetrievalQuery,
}

// EmbeddingOptions are the options of an embedding call, read from
// ProviderOptions["googleVertex"], ["vertex"] or ["google"].
type EmbeddingOptions struct {
	// OutputDimensionality truncates the embeddings to this size. It takes
	// precedence over EmbedOptions.Dimensions.
	OutputDimensionality *int `json:"outputDimensionality,omitempty"`

	// TaskType optimizes the embeddings for a task.
	TaskType google.EmbeddingTaskType `json:"taskType,omitempty"`

	// Title is the title of the embedded document. It is only valid with
	// the RETRIEVAL_DOCUMENT task type.
	Title string `json:"title,omitempty"`

	// AutoTruncate truncates input that exceeds the model's limit instead
	// of failing. The API default is true.
	AutoTruncate *bool `json:"autoTruncate,omitempty"`
}

func (o EmbeddingOptions) validate() error {
	if o.TaskType != "" && !slices.Contains(embeddingTaskTypes, o.TaskType) {
		return fmt.Errorf("%w: unsupported embedding task type %q", model.ErrInvalidArgument, o.TaskType)
	}

	return nil
}

// EmbeddingModel is a Vertex AI text embedding model. It is safe for
// concurrent use.
type EmbeddingModel struct {
	modelID  string
	settings *settings
}

// Provider returns "google.vertex.embedding".
func (m *EmbeddingModel) Provider() string {
	return m.settings.name + ".embedding"
}

// ModelID returns the model ID.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns 250, or 1 for gemini-embedding-2 models,
// which only support the embedContent API.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	if m.usesEmbedContent() {
		return embedContentMaxEmbeddingsPerCall
	}

	return predictMaxEmbeddingsPerCall
}

// SupportsParallelCalls returns true.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return true
}

func (m *EmbeddingModel) usesEmbedContent() bool {
	return slices.Contains(embedContentModels, m.modelID)
}

// predictRequest is the body of POST models/{model}:predict.
type predictRequest struct {
	Instances  []predictInstance `json:"instances"`
	Parameters predictParameters `json:"parameters"`
}

type predictInstance struct {
	Content  string                   `json:"content"`
	TaskType google.EmbeddingTaskType `json:"task_type,omitempty"`
	Title    string                   `json:"title,omitempty"`
}

type predictParameters struct {
	OutputDimensionality *int  `json:"outputDimensionality,omitempty"`
	AutoTruncate         *bool `json:"autoTruncate,omitempty"`
}

type predictResponse struct {
	Predictions []struct {
		Embeddings struct {
			Values     []float64 `json:"values"`
			Statistics struct {
				TokenCount int `json:"token_count"`
			} `json:"statistics"`
		} `json:"embeddings"`
	} `json:"predictions"`
}

// embedContentRequest is the body of POST models/{model}:embedContent.
type embedContentRequest struct {
	Content            embedContent       `json:"content"`
	EmbedContentConfig embedContentConfig `json:"embedContentConfig"`
}

type embedContent struct {
	Parts []embedContentPart `json:"parts"`
}

type embedContentPart struct {
	Text string `json:"text"`
}

type embedContentConfig struct {
	OutputDimensionality *int                     `json:"outputDimensionality,omitempty"`
	TaskType             google.EmbeddingTaskType `json:"taskType,omitempty"`
	Title                string                   `json:"title,omitempty"`
	AutoTruncate         *bool                    `json:"autoTruncate,omitempty"`
}

type embedContentResponse struct {
	Embedding struct {
		Values []float64 `json:"values"`
	} `json:"embedding"`
	UsageMetadata *struct {
		PromptTokenCount *int `json:"promptTokenCount"`
	} `json:"usageMetadata"`
}

// Embed converts the values into vectors with the predict API, or with the
// embedContent API for gemini-embedding-2 models.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	vertexOptions, _, err := readOptions[EmbeddingOptions](options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	if err := vertexOptions.validate(); err != nil {
		return nil, err
	}

	if len(options.Values) > m.MaxEmbeddingsPerCall() {
		return nil, fmt.Errorf(
			"%w: too many values for a single embedding call: %s %s supports at most %d values per call, got %d",
			model.ErrInvalidArgument,
			m.Provider(),
			m.modelID,
			m.MaxEmbeddingsPerCall(),
			len(options.Values),
		)
	}

	dimensions := vertexOptions.OutputDimensionality
	if dimensions == nil {
		dimensions = options.Dimensions
	}

	if m.usesEmbedContent() {
		return m.embedContent(ctx, options, vertexOptions, dimensions)
	}

	return m.predict(ctx, options, vertexOptions, dimensions)
}

func (m *EmbeddingModel) predict(ctx context.Context, options model.EmbedOptions, vertexOptions EmbeddingOptions, dimensions *int) (*model.EmbedResult, error) {
	instances := make([]predictInstance, 0, len(options.Values))
	for _, value := range options.Values {
		instances = append(instances, predictInstance{
			Content:  value,
			TaskType: vertexOptions.TaskType,
			Title:    vertexOptions.Title,
		})
	}

	body := predictRequest{
		Instances: instances,
		Parameters: predictParameters{
			OutputDimensionality: dimensions,
			AutoTruncate:         vertexOptions.AutoTruncate,
		},
	}

	response, err := m.post(ctx, ":predict", options, body)
	if err != nil {
		return nil, err
	}

	var decoded predictResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	embeddings := make([][]float64, 0, len(decoded.Predictions))
	tokens := 0

	for _, prediction := range decoded.Predictions {
		embeddings = append(embeddings, prediction.Embeddings.Values)
		tokens += prediction.Embeddings.Statistics.TokenCount
	}

	return &model.EmbedResult{
		Embeddings: embeddings,
		Usage: &model.EmbeddingUsage{
			Tokens: tokens,
		},
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

func (m *EmbeddingModel) embedContent(ctx context.Context, options model.EmbedOptions, vertexOptions EmbeddingOptions, dimensions *int) (*model.EmbedResult, error) {
	text := ""
	if len(options.Values) > 0 {
		text = options.Values[0]
	}

	body := embedContentRequest{
		Content: embedContent{
			Parts: []embedContentPart{
				{
					Text: text,
				},
			},
		},
		EmbedContentConfig: embedContentConfig{
			OutputDimensionality: dimensions,
			TaskType:             vertexOptions.TaskType,
			Title:                vertexOptions.Title,
			AutoTruncate:         vertexOptions.AutoTruncate,
		},
	}

	response, err := m.post(ctx, ":embedContent", options, body)
	if err != nil {
		return nil, err
	}

	var decoded embedContentResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	result := &model.EmbedResult{
		Embeddings: [][]float64{
			decoded.Embedding.Values,
		},
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.UsageMetadata != nil && decoded.UsageMetadata.PromptTokenCount != nil {
		result.Usage = &model.EmbeddingUsage{
			Tokens: *decoded.UsageMetadata.PromptTokenCount,
		}
	}

	return result, nil
}

// post sends body to the model's method, such as ":predict".
func (m *EmbeddingModel) post(ctx context.Context, method string, options model.EmbedOptions, body any) (*apicall.Response, error) {
	baseURL, err := m.settings.resolveBaseURL(false)
	if err != nil {
		return nil, err
	}

	request, err := m.settings.request(ctx, baseURL+"/models/"+m.modelID+method, options.Headers)
	if err != nil {
		return nil, err
	}

	return apicall.PostJSON(ctx, request, body)
}
