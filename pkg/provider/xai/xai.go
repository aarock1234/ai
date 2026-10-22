// Package xai implements the xAI provider: Grok language models through the
// Responses API with xAI's server-side agentic tools, image generation and
// editing, video generation, editing and extension, text-to-speech, file and
// streaming transcription, and the Files API. It is a port of the Vercel AI
// SDK xAI provider.
package xai

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
	// defaultBaseURL is the xAI API.
	defaultBaseURL = "https://api.x.ai/v1"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "XAI_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "xai"
)

// Provider creates xAI models. Use New to create it.
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

// WithAPIKey sets the API key. Without it, the key is read from XAI_API_KEY
// when a request is made.
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
// "xai". Provider options and metadata keep the "xai" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for URL citation
// sources.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// New returns an xAI provider.
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

// Responses returns a Responses API language model.
func (p *Provider) Responses(modelID string) *ResponsesModel {
	return &ResponsesModel{
		modelID:  modelID,
		provider: p,
	}
}

// Image returns an image generation and editing model.
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// Speech returns the text-to-speech model. xAI has a single speech model,
// so the model ID is only reported.
func (p *Provider) Speech(modelID string) *SpeechModel {
	return &SpeechModel{
		modelID:  modelID,
		provider: p,
	}
}

// Transcription returns a transcription model. An empty model ID uses xAI's
// default transcription model.
func (p *Provider) Transcription(modelID string) *TranscriptionModel {
	return &TranscriptionModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel returns the Responses API model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Responses(modelID), nil
}

// EmbeddingModel reports that xAI has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel returns the image model for modelID.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel returns the transcription model for modelID.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return p.Transcription(modelID), nil
}

// SpeechModel returns the speech model.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return p.Speech(modelID), nil
}

// RerankingModel reports that xAI has no reranking models.
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
			"Bearer " + apiKey,
		},
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        p.baseURL + path,
		Headers:    apicall.MergeHeaders(authorization, p.headers, callHeaders),
		ParseError: parseError,
	}, nil
}

// ErrorData is a structured xAI API error.
type ErrorData struct {
	Message string          `json:"message"`
	Type    *string         `json:"type"`
	Param   json.RawMessage `json:"param,omitempty"`

	// Code is a string or a number.
	Code json.RawMessage `json:"code,omitempty"`
}

// errorBody accepts the API error shape {"error": {...}}, the Responses
// shape {"code": "...", "error": "..."} and the speech shape
// {"error": "..."}.
type errorBody struct {
	Error json.RawMessage `json:"error"`
	Code  *string         `json:"code"`
}

// parseError reads the message of an xAI error response.
func parseError(body []byte) string {
	var decoded errorBody
	if err := json.Unmarshal(body, &decoded); err != nil || len(decoded.Error) == 0 {
		return ""
	}

	var text string
	if err := json.Unmarshal(decoded.Error, &text); err == nil {
		if decoded.Code != nil {
			return *decoded.Code + ": " + text
		}

		return text
	}

	var data ErrorData
	if err := json.Unmarshal(decoded.Error, &data); err != nil {
		return ""
	}

	return data.Message
}

var _ model.Provider = (*Provider)(nil)
