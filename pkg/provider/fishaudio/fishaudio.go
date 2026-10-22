// Package fishaudio implements the Fish Audio provider: text-to-speech and
// speech-to-text. It is a port of the Vercel AI SDK Fish Audio provider.
package fishaudio

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the Fish Audio API.
	defaultBaseURL = "https://api.fish.audio"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "FISH_AUDIO_API_KEY"

	// defaultName is the provider name reported by the models.
	defaultName = "fish-audio"

	// Namespace is the provider options and metadata namespace.
	Namespace = "fishAudio"
)

// Provider creates Fish Audio models. Use New to create it.
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
// FISH_AUDIO_API_KEY when a request is made.
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
// "fish-audio". Provider options and metadata keep the "fishAudio"
// namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// New returns a Fish Audio provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL: defaultBaseURL,
		name:    defaultName,
	}

	for _, option := range options {
		option(provider)
	}

	provider.baseURL = strings.TrimRight(provider.baseURL, "/")

	return provider
}

// Speech returns a text-to-speech model, such as "s2.1-pro".
func (p *Provider) Speech(modelID string) *SpeechModel {
	return &SpeechModel{
		modelID:  modelID,
		provider: p,
	}
}

// Transcription returns the speech-to-text model. The ASR endpoint has no
// model selector, so the ID is a label that defaults to "transcribe-1".
func (p *Provider) Transcription(modelID string) *TranscriptionModel {
	if modelID == "" {
		modelID = defaultTranscriptionModelID
	}

	return &TranscriptionModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel reports that Fish Audio has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that Fish Audio has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that Fish Audio has no image models.
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

// RerankingModel reports that Fish Audio has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// apiRequest describes a call to path with the provider headers, the extra
// headers and the call headers, in that order of precedence.
func (p *Provider) apiRequest(path string, extraHeaders http.Header, callHeaders http.Header) (apicall.Request, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return apicall.Request{}, err
	}

	authorization := http.Header{
		"Authorization": {
			"Bearer " + apiKey,
		},
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        p.baseURL + path,
		Headers:    apicall.MergeHeaders(authorization, p.headers, extraHeaders, callHeaders),
		ParseError: parseError,
	}, nil
}

// errorResponse is Fish Audio's error body.
type errorResponse struct {
	Status  *int    `json:"status"`
	Message *string `json:"message"`
}

// parseError reads the message of a Fish Audio error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	if response.Message == nil {
		return "Unknown Fish Audio error"
	}

	return *response.Message
}

var _ model.Provider = (*Provider)(nil)
