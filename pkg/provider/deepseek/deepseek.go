// Package deepseek implements DeepSeek: chat completions with thinking,
// reasoning efforts, image input and assistant prefix completion, and file
// uploads for images. It is a port of the Vercel AI SDK DeepSeek provider.
package deepseek

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
	// defaultBaseURL is the DeepSeek API.
	defaultBaseURL = "https://api.deepseek.com"

	// betaSuffix ends the base URL of DeepSeek's beta API, which supports
	// assistant prefix completion and strict tool calls.
	betaSuffix = "/beta"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "DEEPSEEK_API_KEY"

	// Namespace is the provider options, metadata and file reference
	// namespace.
	Namespace = "deepseek"
)

// Provider creates DeepSeek models. Use New to create it.
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
// DEEPSEEK_API_KEY when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the API base URL. A URL ending in "/beta" enables
// assistant prefix completion and strict tool calls.
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
// "deepseek". Provider options and metadata keep the "deepseek" namespace.
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

// New returns a DeepSeek provider.
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

// Chat returns a chat model, such as "deepseek-v4-flash".
func (p *Provider) Chat(modelID string) *ChatModel {
	isBeta := strings.HasSuffix(p.baseURL, betaSuffix)

	return NewChatModel(modelID, ChatConfig{
		Provider:                          p.name + ".chat",
		OptionsNamespace:                  Namespace,
		BuildURL:                          p.buildURL,
		Headers:                           p.requestHeaders,
		HTTPClient:                        p.httpClient,
		GenerateID:                        p.generateID,
		SupportsAssistantPrefixCompletion: isBeta,
		SupportsStrictToolCalls:           isBeta,
	})
}

// Files returns the files interface, an *Files that uploads images that
// chat prompts reference by ID.
func (p *Provider) Files() model.Files {
	return &Files{
		provider: p,
	}
}

// LanguageModel returns the chat model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID), nil
}

// EmbeddingModel reports that DeepSeek has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that DeepSeek has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that DeepSeek has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that DeepSeek has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that DeepSeek has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

func (p *Provider) url(path string) string {
	return p.baseURL + path
}

// buildURL is url as a ChatConfig.BuildURL, which cannot fail.
func (p *Provider) buildURL(path string) (string, error) {
	return p.url(path), nil
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

// ErrorData is a DeepSeek error response.
type ErrorData struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail is the error object of a DeepSeek error response.
type ErrorDetail struct {
	Message string `json:"message"`

	// Type, Param and Code are optional. Code is a string or a number.
	Type  *string         `json:"type,omitempty"`
	Param json.RawMessage `json:"param,omitempty"`
	Code  json.RawMessage `json:"code,omitempty"`
}

// parseError reads the message of a DeepSeek error response.
func parseError(body []byte) string {
	var data ErrorData
	if err := json.Unmarshal(body, &data); err != nil {
		return ""
	}

	return data.Error.Message
}

var _ model.Provider = (*Provider)(nil)
