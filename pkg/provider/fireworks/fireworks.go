// Package fireworks implements Fireworks AI, which serves open models through
// OpenAI-compatible chat completions, completions and embeddings and through
// its image generation workflows. It is a port of the Vercel AI SDK Fireworks
// provider and builds on the openaicompatible package.
package fireworks

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

const (
	// defaultBaseURL is the Fireworks inference API.
	defaultBaseURL = "https://api.fireworks.ai/inference/v1"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "FIREWORKS_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "fireworks"
)

// Provider creates Fireworks models. Use New to create it.
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
// FIREWORKS_API_KEY when a request is made.
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
// "fireworks". Provider options and metadata keep the "fireworks" namespace.
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

// New returns a Fireworks provider.
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

// CompletionModel is a Fireworks legacy completions model.
type CompletionModel struct {
	*openaicompatible.CompletionModel

	provider string
}

// Provider returns the provider identifier, "fireworks.completion" by
// default.
func (m *CompletionModel) Provider() string {
	return m.provider
}

// EmbeddingModel is a Fireworks embedding model.
type EmbeddingModel struct {
	*openaicompatible.EmbeddingModel

	provider string
}

// Provider returns the provider identifier, "fireworks.embedding" by default.
func (m *EmbeddingModel) Provider() string {
	return m.provider
}

// Chat returns a chat model, such as
// "accounts/fireworks/models/llama-v3p3-70b-instruct".
func (p *Provider) Chat(modelID string) *ChatModel {
	return &ChatModel{
		modelID:  modelID,
		provider: p,
	}
}

// Completion returns a legacy completions model.
func (p *Provider) Completion(modelID string) *CompletionModel {
	completion := openaicompatible.NewCompletionModel(modelID, openaicompatible.CompletionConfig{
		Provider:     Namespace + ".completion",
		BuildURL:     openaicompatible.InfallibleURL(p.url),
		Headers:      p.requestHeaders,
		HTTPClient:   p.httpClient,
		ParseError:   parseError,
		IncludeUsage: true,
	})

	return &CompletionModel{
		CompletionModel: completion,
		provider:        p.name + ".completion",
	}
}

// Embedding returns an embedding model, such as
// "nomic-ai/nomic-embed-text-v1.5".
func (p *Provider) Embedding(modelID string) *EmbeddingModel {
	embedding := openaicompatible.NewEmbeddingModel(modelID, openaicompatible.EmbeddingConfig{
		Provider:   Namespace + ".embedding",
		BuildURL:   openaicompatible.InfallibleURL(p.url),
		Headers:    p.requestHeaders,
		HTTPClient: p.httpClient,
		ParseError: parseError,
	})

	return &EmbeddingModel{
		EmbeddingModel: embedding,
		provider:       p.name + ".embedding",
	}
}

// Image returns an image generation model, such as
// "accounts/fireworks/models/flux-1-dev-fp8".
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

// TranscriptionModel reports that Fireworks has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Fireworks has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Fireworks has no reranking models.
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

// ErrorData is a Fireworks error response. The error is a string or an
// OpenAI-style error object.
type ErrorData struct {
	// Message is the string error or the message of the error object.
	Message string `json:"-"`

	// Detail is the error object, when the error is not a string.
	Detail *ErrorDetail `json:"-"`
}

// ErrorDetail is an OpenAI-style error object.
type ErrorDetail struct {
	Message string `json:"message"`

	// Object, Type, Param and Code are optional. Code is a string or a
	// number.
	Object *string         `json:"object,omitempty"`
	Type   *string         `json:"type,omitempty"`
	Param  json.RawMessage `json:"param,omitempty"`
	Code   json.RawMessage `json:"code,omitempty"`
}

// errorEnvelope holds the error member, which is a string or an object.
type errorEnvelope struct {
	Error json.RawMessage `json:"error"`
}

// UnmarshalJSON accepts a string error or an error object.
func (d *ErrorData) UnmarshalJSON(data []byte) error {
	var envelope errorEnvelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		return err
	}

	var message string
	if err := json.Unmarshal(envelope.Error, &message); err == nil {
		*d = ErrorData{
			Message: message,
		}

		return nil
	}

	var detail ErrorDetail
	if err := json.Unmarshal(envelope.Error, &detail); err != nil {
		return err
	}

	*d = ErrorData{
		Message: detail.Message,
		Detail:  &detail,
	}

	return nil
}

// parseError reads the message of a Fireworks error response.
func parseError(body []byte) string {
	var data ErrorData
	if err := json.Unmarshal(body, &data); err != nil {
		return ""
	}

	return data.Message
}

var _ model.Provider = (*Provider)(nil)
