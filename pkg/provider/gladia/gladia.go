// Package gladia implements the Gladia provider: pre-recorded transcription,
// which uploads the audio, starts a job and polls its result. It is a port
// of the Vercel AI SDK Gladia provider.
package gladia

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the Gladia API.
	defaultBaseURL = "https://api.gladia.io"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "GLADIA_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "gladia"
)

var (
	// ErrTranscriptionFailed reports a job that Gladia finished with an
	// error status.
	ErrTranscriptionFailed = errors.New("gladia transcription failed")

	// ErrTranscriptionTimeout reports a job that did not finish within the
	// polling timeout of one minute.
	ErrTranscriptionTimeout = errors.New("gladia transcription timed out")
)

// Provider creates Gladia models. Use New to create it.
type Provider struct {
	apiKey     string
	baseURL    string
	headers    http.Header
	httpClient *http.Client
	name       string
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from
// GLADIA_API_KEY when a request is made.
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
// "gladia". Provider options and metadata keep the "gladia" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// New returns a Gladia provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL: defaultBaseURL,
		name:    Namespace,
	}

	for _, option := range options {
		option(provider)
	}

	provider.baseURL = strings.TrimRight(provider.baseURL, "/")

	return provider
}

// Transcription returns the transcription model. Gladia has a single
// model, so it takes no model ID.
func (p *Provider) Transcription() *TranscriptionModel {
	return &TranscriptionModel{
		provider: p,
	}
}

// LanguageModel reports that Gladia has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that Gladia has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that Gladia has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel returns the transcription model. The model ID is
// ignored, as upstream, because Gladia has a single model.
func (p *Provider) TranscriptionModel(string) (model.TranscriptionModel, error) {
	return p.Transcription(), nil
}

// SpeechModel reports that Gladia has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Gladia has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// requestHeaders returns the authentication, provider and call headers.
func (p *Provider) requestHeaders(callHeaders http.Header) (http.Header, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return nil, err
	}

	authorization := http.Header{
		"X-Gladia-Key": {
			apiKey,
		},
	}

	return apicall.MergeHeaders(authorization, p.headers, callHeaders), nil
}

// apiRequest describes a call to url with the request headers.
func (p *Provider) apiRequest(url string, headers http.Header) apicall.Request {
	return apicall.Request{
		Client:     p.httpClient,
		URL:        url,
		Headers:    headers,
		ParseError: parseError,
	}
}

// errorResponse is Gladia's error body.
type errorResponse struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Message string `json:"message"`
	Code    int    `json:"code"`
}

// parseError reads the message of a Gladia error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	return response.Error.Message
}

var _ model.Provider = (*Provider)(nil)
