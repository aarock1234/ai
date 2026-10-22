// Package gmicloud implements GMI Cloud, which serves open models through an
// OpenAI-compatible chat completions API. It is a port of the Vercel AI SDK
// GMI Cloud provider and builds on the openaicompatible package, adding GMI
// Cloud's nested error diagnostics.
package gmicloud

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
	// defaultBaseURL is the GMI Cloud API.
	defaultBaseURL = "https://api.gmi-serving.com/v1"

	// apiKeyVariable is the environment variable that holds the API key. It
	// matches GMI Cloud's own SDK.
	apiKeyVariable = "GMI_CLOUD_APIKEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "gmicloud"
)

// Provider creates GMI Cloud models. Use New to create it.
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
// GMI_CLOUD_APIKEY when a request is made.
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
// "gmicloud". Provider options and metadata keep the "gmicloud" namespace.
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

// New returns a GMI Cloud provider.
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

// ChatModel is a GMI Cloud chat model. It sends OpenAI-compatible chat
// completions and reports usage in streams.
type ChatModel struct {
	*openaicompatible.ChatModel

	provider string
}

// Provider returns the provider identifier, "gmicloud.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider
}

// Chat returns a chat model. Model IDs follow GMI Cloud's catalog, such as
// "deepseek-ai/DeepSeek-V4-Flash-0731".
func (p *Provider) Chat(modelID string) *ChatModel {
	chat := openaicompatible.NewChatModel(modelID, openaicompatible.ChatConfig{
		Provider:     Namespace + ".chat",
		BuildURL:     openaicompatible.InfallibleURL(p.url),
		Headers:      p.requestHeaders,
		HTTPClient:   p.httpClient,
		ParseError:   parseError,
		IncludeUsage: true,
		GenerateID:   p.generateID,
	})

	return &ChatModel{
		ChatModel: chat,
		provider:  p.name + ".chat",
	}
}

// LanguageModel returns the chat model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID), nil
}

// EmbeddingModel reports that GMI Cloud has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that GMI Cloud has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that GMI Cloud has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that GMI Cloud has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that GMI Cloud has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

func (p *Provider) url(path string) string {
	return p.baseURL + path
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
