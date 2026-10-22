// Package cartesia implements the Cartesia provider: Sonic text-to-speech,
// Ink Whisper batch transcription, Ink 2 streaming transcription and the
// experimental realtime Ink model. It is a port of the Vercel AI SDK
// Cartesia provider.
package cartesia

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the Cartesia API.
	defaultBaseURL = "https://api.cartesia.ai"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "CARTESIA_API_KEY"

	// DefaultVersion is the Cartesia API version sent in the
	// Cartesia-Version header.
	DefaultVersion = "2026-03-01"

	// Namespace is the provider options namespace.
	Namespace = "cartesia"
)

// Provider creates Cartesia models. Use New to create it.
type Provider struct {
	apiKey     string
	baseURL    string
	version    string
	headers    http.Header
	httpClient *http.Client
	name       string
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from
// CARTESIA_API_KEY when a request is made.
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

// WithVersion replaces the API version sent in the Cartesia-Version header.
// The default is DefaultVersion.
func WithVersion(version string) Option {
	return func(p *Provider) {
		p.version = version
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
// "cartesia". Provider options keep the "cartesia" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// New returns a Cartesia provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL: defaultBaseURL,
		version: DefaultVersion,
		name:    Namespace,
	}

	for _, option := range options {
		option(provider)
	}

	provider.baseURL = strings.TrimRight(provider.baseURL, "/")

	return provider
}

// Speech returns a text-to-speech model, such as "sonic-3".
func (p *Provider) Speech(modelID string) *SpeechModel {
	return &SpeechModel{
		modelID:  modelID,
		provider: p,
	}
}

// Transcription returns a transcription model. Only the Ink Whisper family
// supports batch transcription.
func (p *Provider) Transcription(modelID string) *TranscriptionModel {
	return &TranscriptionModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel reports that Cartesia has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that Cartesia has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that Cartesia has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel returns the transcription model for modelID.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return p.Transcription(modelID), nil
}

// SpeechModel returns the speech model for modelID.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return p.Speech(modelID), nil
}

// RerankingModel reports that Cartesia has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// apiRequest describes a call to path with the provider and call headers.
func (p *Provider) apiRequest(path string, callHeaders http.Header) (apicall.Request, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return apicall.Request{}, err
	}

	defaults := http.Header{
		"Authorization": {
			"Bearer " + apiKey,
		},
		"Cartesia-Version": {
			p.version,
		},
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        p.baseURL + path,
		Headers:    apicall.MergeHeaders(defaults, p.headers, callHeaders),
		ParseError: parseError,
	}, nil
}

// errorResponse is Cartesia's error body.
type errorResponse struct {
	ErrorCode *string `json:"error_code"`
	Title     string  `json:"title"`
	Message   string  `json:"message"`
	RequestID string  `json:"request_id"`
	DocURL    string  `json:"doc_url"`
}

// parseError reads the title and message of a Cartesia error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	if response.Title == "" && response.Message == "" {
		return ""
	}

	return response.Title + ": " + response.Message
}

var _ model.Provider = (*Provider)(nil)
