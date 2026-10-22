package openaicompatible

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// embeddingPath is the embeddings endpoint.
const embeddingPath = "/embeddings"

// defaultMaxEmbeddingsPerCall is the batch limit of the OpenAI API.
const defaultMaxEmbeddingsPerCall = 2048

// EmbeddingConfig configures an EmbeddingModel.
type EmbeddingConfig struct {
	// Provider is the model's provider identifier, such as "name.embedding".
	Provider string

	// BuildURL returns the full URL of an API path such as
	// "/chat/completions", or an error when the settings needed to build it
	// are missing.
	BuildURL URLBuilder

	// Headers returns the provider headers of one request.
	Headers func() (http.Header, error)

	// HTTPClient sends requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// ParseError reads failed response bodies. Nil uses ParseError.
	ParseError ErrorParser

	// MaxEmbeddingsPerCall overrides the batch limit of 2048 when positive.
	MaxEmbeddingsPerCall int

	// SequentialCalls disables concurrent batches.
	SequentialCalls bool
}

// EmbeddingModel is an OpenAI-compatible embedding model.
type EmbeddingModel struct {
	modelID string
	config  EmbeddingConfig
}

// NewEmbeddingModel returns an embedding model that uses config.
func NewEmbeddingModel(modelID string, config EmbeddingConfig) *EmbeddingModel {
	return &EmbeddingModel{
		modelID: modelID,
		config:  config,
	}
}

// Provider returns the configured provider identifier.
func (m *EmbeddingModel) Provider() string {
	return m.config.Provider
}

// ModelID returns the model identifier.
func (m *EmbeddingModel) ModelID() string {
	return m.modelID
}

// MaxEmbeddingsPerCall returns the batch limit.
func (m *EmbeddingModel) MaxEmbeddingsPerCall() int {
	if m.config.MaxEmbeddingsPerCall > 0 {
		return m.config.MaxEmbeddingsPerCall
	}

	return defaultMaxEmbeddingsPerCall
}

// SupportsParallelCalls reports whether batches may be sent concurrently.
func (m *EmbeddingModel) SupportsParallelCalls() bool {
	return !m.config.SequentialCalls
}

type embeddingRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format"`
	Dimensions     *int     `json:"dimensions,omitempty"`
	User           *string  `json:"user,omitempty"`
}

type embeddingResponse struct {
	Data []embeddingData `json:"data"`

	Usage *embeddingUsage `json:"usage"`

	// ProviderMetadata is provider-specific data, keyed by namespace.
	ProviderMetadata map[string]json.RawMessage `json:"providerMetadata"`
}

type embeddingData struct {
	Embedding []float64 `json:"embedding"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
}

// Embed embeds values in one request.
func (m *EmbeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	name := optionsName(m.config.Provider)
	warnings := deprecatedNamespaceWarnings(options.ProviderOptions)
	warnings = append(warnings, deprecatedNameWarnings(name, options.ProviderOptions)...)

	embeddingOptions, err := mergedOptions[EmbeddingOptions](options.ProviderOptions, deprecatedNamespace, Namespace, name)
	if err != nil {
		return nil, err
	}

	if len(options.Values) > m.MaxEmbeddingsPerCall() {
		return nil, fmt.Errorf(
			"%w: too many values for a single embedding call: %s %s accepts %d, got %d",
			model.ErrInvalidArgument,
			m.config.Provider,
			m.modelID,
			m.MaxEmbeddingsPerCall(),
			len(options.Values),
		)
	}

	dimensions := options.Dimensions
	if embeddingOptions.Dimensions != nil {
		dimensions = embeddingOptions.Dimensions
	}

	request, err := newAPIRequest(apiRequestSettings{
		path:        embeddingPath,
		buildURL:    m.config.BuildURL,
		headers:     m.config.Headers,
		callHeaders: options.Headers,
		client:      m.config.HTTPClient,
		parseError:  m.config.ParseError,
	})
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, embeddingRequest{
		Model:          m.modelID,
		Input:          options.Values,
		EncodingFormat: "float",
		Dimensions:     dimensions,
		User:           embeddingOptions.User,
	})
	if err != nil {
		return nil, err
	}

	var decoded embeddingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	return &model.EmbedResult{
		Embeddings:       embeddingVectors(decoded.Data),
		Usage:            embeddingTokens(decoded.Usage),
		ProviderMetadata: embeddingMetadata(decoded.ProviderMetadata),
		Warnings:         warnings,
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

func embeddingVectors(data []embeddingData) [][]float64 {
	vectors := make([][]float64, 0, len(data))
	for _, item := range data {
		vectors = append(vectors, item.Embedding)
	}

	return vectors
}

func embeddingTokens(usage *embeddingUsage) *model.EmbeddingUsage {
	if usage == nil {
		return nil
	}

	return &model.EmbeddingUsage{
		Tokens: usage.PromptTokens,
	}
}

// embeddingMetadata forwards the provider metadata of the response, whose
// values are opaque JSON objects.
func embeddingMetadata(members map[string]json.RawMessage) model.ProviderMetadata {
	if members == nil {
		return nil
	}

	metadata := make(model.ProviderMetadata, len(members))
	for namespace, value := range members {
		metadata[namespace] = value
	}

	return metadata
}

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)
