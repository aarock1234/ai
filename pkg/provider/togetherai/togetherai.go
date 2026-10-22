// Package togetherai implements the Together AI provider: OpenAI-compatible
// chat, completion and embedding models, image generation and reranking. It
// is a port of the Vercel AI SDK Together AI provider and builds on the
// openaicompatible package.
package togetherai

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
	// defaultBaseURL is the Together AI API.
	defaultBaseURL = "https://api.together.xyz/v1"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "TOGETHER_API_KEY"

	// deprecatedAPIKeyVariable is the former API key variable, read only
	// when apiKeyVariable is unset.
	deprecatedAPIKeyVariable = "TOGETHER_AI_API_KEY"

	// Namespace is the provider options namespace.
	Namespace = "togetherai"
)

// structuredOutputModel is the chat model that accepts JSON schema response
// formats.
const structuredOutputModel = "deepseek-ai/DeepSeek-V4-Flash-0731"

// Provider creates Together AI models. Use New to create it.
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
// TOGETHER_API_KEY, or the deprecated TOGETHER_AI_API_KEY, when a request is
// made.
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
// "togetherai". Provider options keep the "togetherai" namespace.
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

// New returns a Together AI provider.
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

// ChatModel is a Together AI chat completions model.
type ChatModel struct {
	*openaicompatible.ChatModel

	provider string
}

// Provider returns the provider identifier, "togetherai.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider
}

// CompletionModel is a Together AI legacy completions model.
type CompletionModel struct {
	*openaicompatible.CompletionModel

	provider string
}

// Provider returns the provider identifier, "togetherai.completion" by
// default.
func (m *CompletionModel) Provider() string {
	return m.provider
}

// EmbeddingModel is a Together AI embedding model.
type EmbeddingModel struct {
	*openaicompatible.EmbeddingModel

	provider string
}

// Provider returns the provider identifier, "togetherai.embedding" by
// default.
func (m *EmbeddingModel) Provider() string {
	return m.provider
}

// Chat returns a chat completions language model. Only
// "deepseek-ai/DeepSeek-V4-Flash-0731" receives JSON schema response formats.
func (p *Provider) Chat(modelID string) *ChatModel {
	chat := openaicompatible.NewChatModel(modelID, openaicompatible.ChatConfig{
		Provider:                  Namespace + ".chat",
		BuildURL:                  openaicompatible.InfallibleURL(p.url),
		Headers:                   p.requestHeaders,
		HTTPClient:                p.httpClient,
		IncludeUsage:              true,
		SupportsStructuredOutputs: modelID == structuredOutputModel,
		GenerateID:                p.generateID,
	})

	return &ChatModel{
		ChatModel: chat,
		provider:  p.name + ".chat",
	}
}

// Completion returns a legacy completions language model.
func (p *Provider) Completion(modelID string) *CompletionModel {
	completion := openaicompatible.NewCompletionModel(modelID, openaicompatible.CompletionConfig{
		Provider:     Namespace + ".completion",
		BuildURL:     openaicompatible.InfallibleURL(p.url),
		Headers:      p.requestHeaders,
		HTTPClient:   p.httpClient,
		IncludeUsage: true,
	})

	return &CompletionModel{
		CompletionModel: completion,
		provider:        p.name + ".completion",
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
		EmbeddingModel: embedding,
		provider:       p.name + ".embedding",
	}
}

// Image returns an image generation model.
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
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

// ImageModel returns the image model for modelID.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel reports that Together AI has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Together AI has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel returns the reranking model for modelID.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return p.Reranking(modelID), nil
}

// url returns the full URL of an API path.
func (p *Provider) url(path string) string {
	return p.baseURL + path
}

// apiRequest describes a call to path with the provider and call headers.
func (p *Provider) apiRequest(path string, callHeaders http.Header) (apicall.Request, error) {
	headers, err := p.requestHeaders()
	if err != nil {
		return apicall.Request{}, err
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        p.url(path),
		Headers:    apicall.MergeHeaders(headers, callHeaders),
		ParseError: openaicompatible.ParseError,
	}, nil
}

// requestHeaders returns the authorization header followed by the custom
// headers, which may replace it.
func (p *Provider) requestHeaders() (http.Header, error) {
	apiKey, err := p.loadAPIKey()
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

// loadAPIKey returns the configured key, then TOGETHER_API_KEY, then the
// deprecated TOGETHER_AI_API_KEY.
func (p *Provider) loadAPIKey() (string, error) {
	return apikey.Load(p.apiKey, apiKeyVariable, deprecatedAPIKeyVariable)
}

var _ model.Provider = (*Provider)(nil)
