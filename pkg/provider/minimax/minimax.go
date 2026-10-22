// Package minimax implements the MiniMax provider, which serves the MiniMax
// M-series chat models through an Anthropic-compatible Messages API and the
// MiniMax video models through the video generation API. It is a port of the
// Vercel AI SDK MiniMax provider and builds on the anthropic package.
package minimax

import (
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

const (
	// defaultBaseURL is MiniMax's Anthropic-compatible endpoint.
	defaultBaseURL = "https://api.minimax.io/anthropic/v1"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "MINIMAX_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "minimax"
)

// Provider creates MiniMax models. Use New to create it.
type Provider struct {
	apiKey       string
	baseURL      string
	videoBaseURL string
	headers      http.Header
	httpClient   *http.Client
	name         string
	generateID   func() string
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from
// MINIMAX_API_KEY when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the Messages API base URL, for example to use a
// proxy.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.baseURL = baseURL
	}
}

// WithVideoBaseURL replaces the video API root, "https://api.minimax.io" by
// default, without a version suffix.
func WithVideoBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.videoBaseURL = baseURL
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
// "minimax". Provider options and metadata keep the "minimax" namespace.
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

// New returns a MiniMax provider.
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

// Chat returns a Messages API language model.
func (p *Provider) Chat(modelID string) *ChatModel {
	return &ChatModel{
		modelID:             modelID,
		provider:            p,
		supportsM31Features: supportsM31Features(modelID),
	}
}

// LanguageModel returns the chat model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID), nil
}

// EmbeddingModel reports that MiniMax has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that MiniMax has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that MiniMax has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that MiniMax has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that MiniMax has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// messages returns the Anthropic provider of one call, with the model's
// additional options. The API key is resolved first so that a missing key
// names MINIMAX_API_KEY.
func (p *Provider) messages(modelOptions ...anthropic.Option) (*anthropic.Provider, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return nil, err
	}

	options := []anthropic.Option{
		anthropic.WithAPIKey(apiKey),
		anthropic.WithBaseURL(p.baseURL),
		anthropic.WithHeaders(p.headers),
		anthropic.WithHTTPClient(p.httpClient),
		anthropic.WithName(Namespace + ".messages"),
		anthropic.WithIDGenerator(p.generateID),
	}

	return anthropic.New(append(options, modelOptions...)...), nil
}

var _ model.Provider = (*Provider)(nil)
