// Package baseten implements Baseten, which serves Model APIs and dedicated
// deployments through OpenAI-compatible chat completions and embeddings. It
// is a port of the Vercel AI SDK Baseten provider and builds on the
// openaicompatible package.
package baseten

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
	// defaultBaseURL is the Baseten Model APIs endpoint.
	defaultBaseURL = "https://inference.baseten.co/v1"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "BASETEN_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "baseten"
)

// Provider creates Baseten models. Use New to create it.
type Provider struct {
	apiKey            string
	baseURL           string
	modelURL          string
	headers           http.Header
	httpClient        *http.Client
	name              string
	generateID        func() string
	performanceClient PerformanceClientFactory
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from
// BASETEN_API_KEY when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the Model APIs base URL, for example to use a proxy.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.baseURL = baseURL
	}
}

// WithModelURL sets the URL of a dedicated deployment. Chat models require a
// "/sync/v1" URL; embedding models require a "/sync" or "/sync/v1" URL.
// Without it, chat models use the Model APIs.
func WithModelURL(modelURL string) Option {
	return func(p *Provider) {
		p.modelURL = modelURL
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
// "baseten". Provider options and metadata keep the "baseten" namespace.
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

// WithPerformanceClient sends embeddings through a client of Baseten's
// performance client, which batches and hedges requests itself, instead of
// the OpenAI-compatible HTTP endpoint.
func WithPerformanceClient(factory PerformanceClientFactory) Option {
	return func(p *Provider) {
		p.performanceClient = factory
	}
}

// New returns a Baseten provider.
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

// LanguageModel returns the chat model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID)
}

// EmbeddingModel returns the embedding model for modelID.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return p.Embedding(modelID)
}

// ImageModel reports that Baseten has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that Baseten has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Baseten has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Baseten has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// loadAPIKey returns the configured key or the key from the environment.
func (p *Provider) loadAPIKey() (string, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return "", err
	}

	return apiKey, nil
}

// requestHeaders returns the authorization header followed by the custom
// headers.
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

// ErrorData is a Baseten error response. The Model APIs send the error as a
// string; dedicated deployments send an OpenAI-style error object.
type ErrorData struct {
	// Message is the string error or the message of the error object.
	Message string `json:"-"`

	// Detail is the error object, when the error is not a string.
	Detail *ErrorDetail `json:"-"`
}

// ErrorDetail is the OpenAI-style error object of a dedicated deployment.
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

// parseError reads the message of a Baseten error response.
func parseError(body []byte) string {
	var data ErrorData
	if err := json.Unmarshal(body, &data); err != nil {
		return ""
	}

	return data.Message
}

var _ model.Provider = (*Provider)(nil)
