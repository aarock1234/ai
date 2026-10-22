// Package deepinfra implements DeepInfra, which serves open models through
// OpenAI-compatible chat completions, completions and embeddings and through
// its native image inference API. It is a port of the Vercel AI SDK DeepInfra
// provider and builds on the openaicompatible package.
package deepinfra

import (
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

const (
	// defaultBaseURL is the DeepInfra API.
	defaultBaseURL = "https://api.deepinfra.com/v1"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "DEEPINFRA_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "deepinfra"
)

// API path prefixes below the base URL.
const (
	openAIPrefix    = "/openai"
	inferencePrefix = "/inference"
)

// Provider creates DeepInfra models. Use New to create it.
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
// DEEPINFRA_API_KEY when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the API base URL, for example to use a proxy. The
// OpenAI-compatible and image inference paths are added to it.
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
// "deepinfra". Provider options and metadata keep the "deepinfra" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for tool calls that
// arrive without one.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// New returns a DeepInfra provider.
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

// ChatModel is a DeepInfra chat model. It sends OpenAI-compatible chat
// completions with structured outputs and corrects the completion token
// counts that DeepInfra reports for Gemini and Gemma models.
type ChatModel struct {
	*openaicompatible.ChatModel

	provider string
}

// Provider returns the provider identifier, "deepinfra.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider
}

// CompletionModel is a DeepInfra legacy completions model.
type CompletionModel struct {
	*openaicompatible.CompletionModel

	provider string
}

// Provider returns the provider identifier, "deepinfra.completion" by
// default.
func (m *CompletionModel) Provider() string {
	return m.provider
}

// EmbeddingModel is a DeepInfra embedding model.
type EmbeddingModel struct {
	*openaicompatible.EmbeddingModel

	provider string
}

// Provider returns the provider identifier, "deepinfra.embedding" by default.
func (m *EmbeddingModel) Provider() string {
	return m.provider
}

// Chat returns a chat model, such as "meta-llama/Llama-3.3-70B-Instruct".
func (p *Provider) Chat(modelID string) *ChatModel {
	chat := openaicompatible.NewChatModel(modelID, openaicompatible.ChatConfig{
		Provider:                  Namespace + ".chat",
		BuildURL:                  openaicompatible.InfallibleURL(p.openAIURL),
		Headers:                   p.requestHeaders,
		HTTPClient:                p.httpClient,
		SupportsStructuredOutputs: true,
		ConvertUsage:              convertUsage,
		GenerateID:                p.generateID,
	})

	return &ChatModel{
		ChatModel: chat,
		provider:  p.name + ".chat",
	}
}

// Completion returns a legacy completions model. It accepts the chat model
// IDs.
func (p *Provider) Completion(modelID string) *CompletionModel {
	completion := openaicompatible.NewCompletionModel(modelID, openaicompatible.CompletionConfig{
		Provider:   Namespace + ".completion",
		BuildURL:   openaicompatible.InfallibleURL(p.openAIURL),
		Headers:    p.requestHeaders,
		HTTPClient: p.httpClient,
	})

	return &CompletionModel{
		CompletionModel: completion,
		provider:        p.name + ".completion",
	}
}

// Embedding returns an embedding model, such as "BAAI/bge-m3".
func (p *Provider) Embedding(modelID string) *EmbeddingModel {
	embedding := openaicompatible.NewEmbeddingModel(modelID, openaicompatible.EmbeddingConfig{
		Provider:   Namespace + ".embedding",
		BuildURL:   openaicompatible.InfallibleURL(p.openAIURL),
		Headers:    p.requestHeaders,
		HTTPClient: p.httpClient,
	})

	return &EmbeddingModel{
		EmbeddingModel: embedding,
		provider:       p.name + ".embedding",
	}
}

// Image returns an image generation and editing model, such as
// "black-forest-labs/FLUX-1-schnell".
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
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

// ImageModel returns the image model for modelID.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel reports that DeepInfra has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that DeepInfra has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that DeepInfra has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// openAIURL returns the URL of an OpenAI-compatible API path.
func (p *Provider) openAIURL(path string) string {
	return p.baseURL + openAIPrefix + path
}

// requestHeaders returns the authorization header followed by the custom
// headers.
func (p *Provider) requestHeaders() (http.Header, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return nil, err
	}

	authorization := http.Header{
		"Authorization": {
			"Bearer " + apiKey,
		},
	}

	return apicall.MergeHeaders(authorization, p.headers), nil
}

var _ model.Provider = (*Provider)(nil)
