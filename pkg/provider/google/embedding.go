package google

import (
	"context"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// maxEmbeddingsPerCall is the batch limit of batchEmbedContents.
const maxEmbeddingsPerCall = 100

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)

// EmbeddingOptions are the Google provider options of an embedding
// call, read from ProviderOptions["google"].
type EmbeddingOptions struct {
	// OutputDimensionality truncates the embeddings to this size. It takes
	// precedence over EmbedOptions.Dimensions.
	OutputDimensionality *int `json:"outputDimensionality,omitempty"`

	TaskType EmbeddingTaskType `json:"taskType,omitempty"`

	// Content holds multimodal parts per value, such as images, video, PDF
	// or audio, in value order. Its length must match the number of values;
	// a nil entry embeds the value as text only.
	Content [][]EmbeddingContentPart `json:"content,omitempty"`
}

// EmbeddingTaskType optimizes embeddings for a task.
type EmbeddingTaskType string

// Embedding task types.
const (
	TaskSemanticSimilarity EmbeddingTaskType = "SEMANTIC_SIMILARITY"
	TaskClassification     EmbeddingTaskType = "CLASSIFICATION"
	TaskClustering         EmbeddingTaskType = "CLUSTERING"
	TaskRetrievalDocument  EmbeddingTaskType = "RETRIEVAL_DOCUMENT"
	TaskRetrievalQuery     EmbeddingTaskType = "RETRIEVAL_QUERY"
	TaskQuestionAnswering  EmbeddingTaskType = "QUESTION_ANSWERING"
	TaskFactVerification   EmbeddingTaskType = "FACT_VERIFICATION"
	TaskCodeRetrievalQuery EmbeddingTaskType = "CODE_RETRIEVAL_QUERY"
)

var embeddingTaskTypes = []EmbeddingTaskType{
	TaskSemanticSimilarity,
	TaskClassification,
	TaskClustering,
	TaskRetrievalDocument,
	TaskRetrievalQuery,
	TaskQuestionAnswering,
	TaskFactVerification,
	TaskCodeRetrievalQuery,
}

// EmbeddingContentPart is one part of embedded content. Exactly one field is
// set.
type EmbeddingContentPart struct {
	Text       *string     `json:"text,omitempty"`
	InlineData *InlineData `json:"inlineData,omitempty"`
	FileData   *FileData   `json:"fileData,omitempty"`
}

// EmbeddingModel embeds text and multimodal content. It is safe for
// concurrent use.
type EmbeddingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider name, "google.generative-ai" by default.
func (m *EmbeddingModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model ID.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns 100.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	return maxEmbeddingsPerCall
}

// SupportsParallelCalls returns true.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return true
}

type embedContentRequest struct {
	Model                string            `json:"model"`
	Content              embeddingContent  `json:"content"`
	OutputDimensionality *int              `json:"outputDimensionality,omitempty"`
	TaskType             EmbeddingTaskType `json:"taskType,omitempty"`
}

type embeddingContent struct {
	Role  contentRole            `json:"role,omitempty"`
	Parts []EmbeddingContentPart `json:"parts"`
}

type batchEmbedContentsRequest struct {
	Requests []embedContentRequest `json:"requests"`
}

type embeddingValues struct {
	Values []float64 `json:"values"`
}

type embedContentResponse struct {
	Embedding *embeddingValues `json:"embedding"`
}

type batchEmbedContentsResponse struct {
	Embeddings []embeddingValues `json:"embeddings"`
}

// Embed embeds the values. A single value uses embedContent; several values
// use batchEmbedContents.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	googleOptions, err := embeddingOptions(options.ProviderOptions, len(options.Values))
	if err != nil {
		return nil, err
	}

	if len(options.Values) > maxEmbeddingsPerCall {
		return nil, fmt.Errorf(
			"%w: %s accepts at most %d values per embedding call, got %d",
			model.ErrInvalidArgument,
			m.modelID,
			maxEmbeddingsPerCall,
			len(options.Values),
		)
	}

	dimensionality := googleOptions.OutputDimensionality
	if dimensionality == nil {
		dimensionality = options.Dimensions
	}

	requests := make([]embedContentRequest, 0, len(options.Values))
	for index, value := range options.Values {
		requests = append(requests, embedContentRequest{
			Model: "models/" + m.modelID,
			Content: embeddingContent{
				Parts: embeddingParts(value, googleOptions.contentAt(index)),
			},
			OutputDimensionality: dimensionality,
			TaskType:             googleOptions.TaskType,
		})
	}

	if len(requests) == 1 {
		return m.embedSingle(ctx, options, requests[0])
	}

	return m.embedBatch(ctx, options, requests)
}

func (m *EmbeddingModel) embedSingle(ctx context.Context, options model.EmbedOptions, body embedContentRequest) (*model.EmbedResult, error) {
	request, err := m.provider.request(ctx, "models/"+m.modelID+":embedContent", options.Headers)
	if err != nil {
		return nil, err
	}

	raw, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var response embedContentResponse
	if err := apicall.DecodeJSON(raw.Body, &response); err != nil {
		return nil, err
	}

	if response.Embedding == nil {
		return nil, fmt.Errorf("%w: embedding response has no embedding", model.ErrInvalidResponse)
	}

	return embedResult(raw, [][]float64{
		response.Embedding.Values,
	}), nil
}

func (m *EmbeddingModel) embedBatch(ctx context.Context, options model.EmbedOptions, requests []embedContentRequest) (*model.EmbedResult, error) {
	request, err := m.provider.request(ctx, "models/"+m.modelID+":batchEmbedContents", options.Headers)
	if err != nil {
		return nil, err
	}

	for index := range requests {
		requests[index].Content.Role = roleUser
	}

	body := batchEmbedContentsRequest{
		Requests: requests,
	}

	raw, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var response batchEmbedContentsResponse
	if err := apicall.DecodeJSON(raw.Body, &response); err != nil {
		return nil, err
	}

	if len(response.Embeddings) != len(requests) {
		return nil, fmt.Errorf("%w: got %d embeddings for %d values", model.ErrInvalidResponse, len(response.Embeddings), len(requests))
	}

	embeddings := make([][]float64, 0, len(response.Embeddings))
	for _, embedding := range response.Embeddings {
		embeddings = append(embeddings, embedding.Values)
	}

	return embedResult(raw, embeddings), nil
}

func embedResult(raw *apicall.Response, embeddings [][]float64) *model.EmbedResult {
	return &model.EmbedResult{
		Embeddings: embeddings,
		Response: model.ResponseInfo{
			Headers: raw.Headers,
			Body:    raw.Body,
		},
	}
}

// embeddingParts combines a value with its multimodal parts. An empty value
// adds no text part when multimodal parts are present.
func embeddingParts(value string, multimodal []EmbeddingContentPart) []EmbeddingContentPart {
	textPart := EmbeddingContentPart{
		Text: new(value),
	}

	if multimodal == nil {
		return []EmbeddingContentPart{
			textPart,
		}
	}

	if value == "" {
		return multimodal
	}

	return slices.Insert(slices.Clone(multimodal), 0, textPart)
}

func embeddingOptions(options model.ProviderOptions, valueCount int) (EmbeddingOptions, error) {
	googleOptions, _, err := model.ProviderValue[EmbeddingOptions](options, Namespace)
	if err != nil {
		return EmbeddingOptions{}, err
	}

	if err := googleOptions.validate(valueCount); err != nil {
		return EmbeddingOptions{}, err
	}

	return googleOptions, nil
}

func (o EmbeddingOptions) validate(valueCount int) error {
	if err := validateEnum("taskType", o.TaskType, embeddingTaskTypes); err != nil {
		return err
	}

	if o.Content == nil {
		return nil
	}

	if len(o.Content) != valueCount {
		return fmt.Errorf(
			"%w: the number of multimodal content entries (%d) must match the number of values (%d)",
			model.ErrInvalidArgument,
			len(o.Content),
			valueCount,
		)
	}

	for index, parts := range o.Content {
		if parts != nil && len(parts) == 0 {
			return fmt.Errorf("%w: multimodal content entry %d must contain at least one part", model.ErrInvalidArgument, index)
		}

		for _, contentPart := range parts {
			if err := contentPart.validate(); err != nil {
				return fmt.Errorf("multimodal content entry %d: %w", index, err)
			}
		}
	}

	return nil
}

func (p EmbeddingContentPart) validate() error {
	setFields := 0
	if p.Text != nil {
		setFields++
	}

	if p.InlineData != nil {
		setFields++
	}

	if p.FileData != nil {
		setFields++
	}

	if setFields != 1 {
		return fmt.Errorf("%w: an embedding content part sets exactly one of text, inlineData and fileData", model.ErrInvalidArgument)
	}

	return nil
}

func (o EmbeddingOptions) contentAt(index int) []EmbeddingContentPart {
	if index >= len(o.Content) {
		return nil
	}

	return o.Content[index]
}
