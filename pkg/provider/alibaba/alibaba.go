// Package alibaba implements Alibaba Cloud Model Studio (DashScope): Qwen chat
// completions with thinking and explicit context caching, text embeddings and
// Wan video generation. It is a port of the Vercel AI SDK Alibaba provider.
package alibaba

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

// Default endpoints. Chat uses the OpenAI-compatible mode; embeddings and
// video use the native DashScope API.
const (
	defaultBaseURL          = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	defaultVideoBaseURL     = "https://dashscope-intl.aliyuncs.com"
	defaultEmbeddingBaseURL = "https://dashscope-intl.aliyuncs.com/api/v1"
)

const (
	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "ALIBABA_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "alibaba"
)

// Provider creates Alibaba models. Use New to create it.
type Provider struct {
	apiKey           string
	baseURL          string
	videoBaseURL     string
	embeddingBaseURL string
	headers          http.Header
	httpClient       *http.Client
	name             string
	generateID       func() string
	includeUsage     bool
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from
// ALIBABA_API_KEY when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the chat API base URL, for example to use a proxy or
// a regional endpoint.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.baseURL = baseURL
	}
}

// WithVideoBaseURL replaces the base URL of the native video generation API.
func WithVideoBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.videoBaseURL = baseURL
	}
}

// WithEmbeddingBaseURL replaces the base URL of the native embedding API.
func WithEmbeddingBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.embeddingBaseURL = baseURL
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
// "alibaba". Provider options and metadata keep the "alibaba" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for streamed reasoning
// and for tool calls that arrive without one.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// WithIncludeUsage controls whether streams report token usage in their
// final chunk. It defaults to true.
func WithIncludeUsage(include bool) Option {
	return func(p *Provider) {
		p.includeUsage = include
	}
}

// New returns an Alibaba provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL:          defaultBaseURL,
		videoBaseURL:     defaultVideoBaseURL,
		embeddingBaseURL: defaultEmbeddingBaseURL,
		name:             Namespace,
		includeUsage:     true,
		generateID: func() string {
			return id.New("")
		},
	}

	for _, option := range options {
		option(provider)
	}

	provider.baseURL = strings.TrimRight(provider.baseURL, "/")
	provider.videoBaseURL = strings.TrimRight(provider.videoBaseURL, "/")
	provider.embeddingBaseURL = strings.TrimRight(provider.embeddingBaseURL, "/")

	return provider
}

// Chat returns a chat model, such as "qwen-plus".
func (p *Provider) Chat(modelID string) *ChatModel {
	return &ChatModel{
		modelID:  modelID,
		provider: p,
	}
}

// Embedding returns a text embedding model, such as "text-embedding-v4".
func (p *Provider) Embedding(modelID string) *EmbeddingModel {
	return &EmbeddingModel{
		modelID:  modelID,
		provider: p,
	}
}

// Video returns a Wan video generation model, such as "wan2.6-t2v".
func (p *Provider) Video(modelID string) *VideoModel {
	return &VideoModel{
		modelID:  modelID,
		provider: p,
	}
}

// VideoModel returns Video(modelID).
func (p *Provider) VideoModel(modelID string) (model.VideoModel, error) {
	return p.Video(modelID), nil
}

// LanguageModel returns the chat model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID), nil
}

// EmbeddingModel returns the embedding model for modelID.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return p.Embedding(modelID), nil
}

// ImageModel reports that Alibaba has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that Alibaba has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Alibaba has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Alibaba has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// apiRequest describes a call to url with the provider and call headers.
// The header sets are applied in order, later sets replacing earlier ones.
func (p *Provider) apiRequest(url string, parse apicall.ErrorParser, headerSets ...http.Header) (apicall.Request, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return apicall.Request{}, err
	}

	authorization := http.Header{
		"Authorization": {
			"Bearer " + apiKey,
		},
	}

	sets := append([]http.Header{
		authorization,
		p.headers,
	}, headerSets...)

	return apicall.Request{
		Client:     p.httpClient,
		URL:        url,
		Headers:    apicall.MergeHeaders(sets...),
		ParseError: parse,
	}, nil
}

// ErrorData is an error response of the OpenAI-compatible chat API.
type ErrorData struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail is the error object of a chat API error response.
type ErrorDetail struct {
	Message string  `json:"message"`
	Code    *string `json:"code,omitempty"`
	Type    *string `json:"type,omitempty"`
}

// parseError reads the message of a chat API error response.
func parseError(body []byte) string {
	var data ErrorData
	if err := json.Unmarshal(body, &data); err != nil {
		return ""
	}

	return data.Error.Message
}

// NativeErrorData is an error response of the native DashScope embedding and
// video APIs.
type NativeErrorData struct {
	Code      *string `json:"code,omitempty"`
	Message   string  `json:"message"`
	RequestID *string `json:"request_id,omitempty"`
}

// parseNativeError reads the message of a native API error response.
func parseNativeError(body []byte) string {
	var data NativeErrorData
	if err := json.Unmarshal(body, &data); err != nil {
		return ""
	}

	return data.Message
}

var _ model.Provider = (*Provider)(nil)
