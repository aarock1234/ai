// Package openrouter implements OpenRouter, which routes OpenAI-compatible
// chat completions to many model providers. It builds on the openaicompatible
// package and adds OpenRouter's reasoning, usage accounting, provider routing,
// generated images, web citations and reasoning detail replay.
package openrouter

import (
	"net/http"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

const (
	// defaultBaseURL is the OpenRouter API.
	defaultBaseURL = "https://openrouter.ai/api/v1"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "OPENROUTER_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "openrouter"
)

// supportedURLs are the image URLs that OpenRouter downloads itself.
var supportedURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
}

// Provider creates OpenRouter models. Use New to create it.
type Provider struct {
	apiKey     string
	baseURL    string
	headers    http.Header
	appURL     string
	appName    string
	httpClient *http.Client
	name       string
	generateID func() string
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from
// OPENROUTER_API_KEY when a request is made.
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

// WithAppURL sends the HTTP-Referer header, which OpenRouter uses to
// attribute requests to an application.
func WithAppURL(appURL string) Option {
	return func(p *Provider) {
		p.appURL = appURL
	}
}

// WithAppName sends the X-Title header, the application name that OpenRouter
// shows in its dashboards and rankings.
func WithAppName(name string) Option {
	return func(p *Provider) {
		p.appName = name
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
// "openrouter". Provider options and metadata keep the "openrouter"
// namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for citation sources and
// for tool calls that arrive without one.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// New returns an OpenRouter provider.
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

// Chat returns a chat model. Model IDs name the upstream provider, such as
// "openai/gpt-5" or "anthropic/claude-sonnet-4.5".
func (p *Provider) Chat(modelID string) *ChatModel {
	chat := openaicompatible.NewChatModel(modelID, openaicompatible.ChatConfig{
		Provider:                  Namespace + ".chat",
		BuildURL:                  openaicompatible.InfallibleURL(p.url),
		Headers:                   p.requestHeaders,
		HTTPClient:                p.httpClient,
		IncludeUsage:              true,
		SupportsStructuredOutputs: true,
		SupportedURLs:             supportedURLs,
		TransformRequestBody:      transformRequestBody,
		MetadataExtractor:         metadataExtractor{},
		ConvertUsage:              convertUsage,
		GenerateID:                p.generateID,
	})

	return &ChatModel{
		provider:   p.name + ".chat",
		chat:       chat,
		generateID: p.generateID,
	}
}

// Embedding returns an embedding model.
func (p *Provider) Embedding(modelID string) *EmbeddingModel {
	embedding := openaicompatible.NewEmbeddingModel(modelID, openaicompatible.EmbeddingConfig{
		Provider:   Namespace + ".embedding",
		BuildURL:   openaicompatible.InfallibleURL(p.url),
		Headers:    p.requestHeaders,
		HTTPClient: p.httpClient,
	})

	return &EmbeddingModel{
		provider:  p.name + ".embedding",
		embedding: embedding,
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

// ImageModel reports that OpenRouter has no image models. Image generation
// models are used as chat models, which return images as files.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that OpenRouter has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that OpenRouter has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that OpenRouter has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

func (p *Provider) url(path string) string {
	return p.baseURL + path
}

// requestHeaders returns the authorization, attribution and custom headers.
func (p *Provider) requestHeaders() (http.Header, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return nil, err
	}

	headers := make(http.Header)
	headers.Set("Authorization", "Bearer "+apiKey)

	if p.appURL != "" {
		headers.Set("HTTP-Referer", p.appURL)
	}

	if p.appName != "" {
		headers.Set("X-Title", p.appName)
	}

	for key, values := range p.headers {
		headers.Del(key)
		for _, value := range values {
			headers.Add(key, value)
		}
	}

	return headers, nil
}

var _ model.Provider = (*Provider)(nil)
