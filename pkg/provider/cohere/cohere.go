// Package cohere implements the Cohere provider: chat with tools, documents,
// citations and thinking, embeddings and reranking. It is a port of the Vercel
// AI SDK Cohere provider and uses the Cohere v2 API.
package cohere

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the Cohere v2 API.
	defaultBaseURL = "https://api.cohere.com/v2"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "COHERE_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "cohere"
)

// Provider creates Cohere models. Use New to create it.
type Provider struct {
	apiKey     string
	baseURL    string
	headers    http.Header
	httpClient *http.Client
	name       string
	generateID func() string
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from
// COHERE_API_KEY when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the API base URL, for example to use a proxy.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.baseURL = baseURL
	}
}

// WithHeaders adds headers to every request. The headers are copied.
func WithHeaders(headers http.Header) Option {
	snapshot := headers.Clone()

	return func(p *Provider) {
		p.headers = snapshot.Clone()
	}
}

// WithHTTPClient sets the client used for every request. Nil uses
// http.DefaultClient.
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// WithName replaces the provider name reported by the models. The default is
// "cohere". Provider options and metadata keep the "cohere" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for citation sources.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// New returns a Cohere provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL: defaultBaseURL,
		name:    Namespace,
		generateID: func() string {
			return id.New("")
		},
	}

	for _, option := range options {
		option(provider)
	}

	provider.baseURL = strings.TrimRight(provider.baseURL, "/")

	return provider
}

// Chat returns a chat language model.
func (p *Provider) Chat(modelID string) *ChatModel {
	return &ChatModel{
		modelID:  modelID,
		provider: p,
	}
}

// Embedding returns an embedding model.
func (p *Provider) Embedding(modelID string) *EmbeddingModel {
	return &EmbeddingModel{
		modelID:  modelID,
		provider: p,
	}
}

// Reranking returns a reranking model.
func (p *Provider) Reranking(modelID string) *RerankingModel {
	return &RerankingModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel returns the chat model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID), nil
}

// EmbeddingModel returns the embedding model for modelID.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return p.Embedding(modelID), nil
}

// ImageModel reports that Cohere has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that Cohere has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Cohere has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel returns the reranking model for modelID.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return p.Reranking(modelID), nil
}

// apiRequest describes a call to path with the provider and call headers.
func (p *Provider) apiRequest(path string, callHeaders http.Header) (apicall.Request, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return apicall.Request{}, err
	}

	authorization := http.Header{
		"Authorization": {
			"Bearer " + apiKey,
		},
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        p.baseURL + path,
		Headers:    apicall.MergeHeaders(authorization, p.headers, callHeaders),
		ParseError: parseError,
	}, nil
}

// errorResponse is Cohere's error body.
type errorResponse struct {
	Message string `json:"message"`
}

// parseError reads the message of a Cohere error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	return response.Message
}

var _ model.Provider = (*Provider)(nil)
