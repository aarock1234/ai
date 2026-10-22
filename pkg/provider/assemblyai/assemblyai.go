// Package assemblyai implements the AssemblyAI provider: pre-recorded
// transcription, which uploads the audio, submits a transcript and polls it
// until it completes. It is a port of the Vercel AI SDK AssemblyAI provider.
package assemblyai

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the AssemblyAI API.
	defaultBaseURL = "https://api.assemblyai.com"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "ASSEMBLYAI_API_KEY"

	// defaultPollInterval is the delay between transcript status checks.
	defaultPollInterval = 3 * time.Second

	// Namespace is the provider options and metadata namespace.
	Namespace = "assemblyai"
)

// ErrTranscriptionFailed reports a transcript that AssemblyAI finished with
// an error status.
var ErrTranscriptionFailed = errors.New("transcription failed")

// Provider creates AssemblyAI models. Use New to create it.
type Provider struct {
	apiKey       string
	baseURL      string
	headers      http.Header
	httpClient   *http.Client
	name         string
	pollInterval time.Duration
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from
// ASSEMBLYAI_API_KEY when a request is made.
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
// "assemblyai". Provider options and metadata keep the "assemblyai"
// namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithPollInterval sets the delay between transcript status checks. The
// default is three seconds; non-positive values keep it.
func WithPollInterval(interval time.Duration) Option {
	return func(p *Provider) {
		if interval > 0 {
			p.pollInterval = interval
		}
	}
}

// New returns an AssemblyAI provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL:      defaultBaseURL,
		name:         Namespace,
		pollInterval: defaultPollInterval,
	}

	for _, option := range options {
		option(provider)
	}

	provider.baseURL = strings.TrimRight(provider.baseURL, "/")

	return provider
}

// Transcription returns a transcription model, such as "universal-3-5-pro".
func (p *Provider) Transcription(modelID string) *TranscriptionModel {
	return &TranscriptionModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel reports that AssemblyAI has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that AssemblyAI has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that AssemblyAI has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel returns the transcription model for modelID.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return p.Transcription(modelID), nil
}

// SpeechModel reports that AssemblyAI has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that AssemblyAI has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// apiRequest describes a call to path with the provider and call headers.
func (p *Provider) apiRequest(path string, callHeaders http.Header) (apicall.Request, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return apicall.Request{}, err
	}

	authorization := http.Header{
		"Authorization": {
			apiKey,
		},
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        p.baseURL + path,
		Headers:    apicall.MergeHeaders(authorization, p.headers, callHeaders),
		ParseError: parseError,
	}, nil
}

// errorResponse is AssemblyAI's error body.
type errorResponse struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Message string `json:"message"`
	Code    int    `json:"code"`
}

// parseError reads the message of an AssemblyAI error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	return response.Error.Message
}

var _ model.Provider = (*Provider)(nil)
