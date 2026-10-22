// Package luma implements the Luma AI provider for Photon image generation.
// Luma queues each generation, so calls poll until the image is ready and
// then download it. It is a port of the Vercel AI SDK Luma provider.
package luma

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the Luma API.
	defaultBaseURL = "https://api.lumalabs.ai"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "LUMA_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "luma"
)

// Provider creates Luma models. Use New to create it.
type Provider struct {
	apiKey     string
	baseURL    string
	headers    http.Header
	httpClient *http.Client
	name       string
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from LUMA_API_KEY
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
// "luma". Provider options keep the "luma" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// New returns a Luma provider.
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

// Image returns an image generation model, such as "photon-1" or
// "photon-flash-1".
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel reports that Luma has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that Luma has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel returns the image model for modelID.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel reports that Luma has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Luma has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Luma has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// apiRequest describes a call to url with the provider and call headers.
func (p *Provider) apiRequest(url string, callHeaders http.Header) (apicall.Request, error) {
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
		URL:        url,
		Headers:    apicall.MergeHeaders(authorization, p.headers, callHeaders),
		ParseError: parseError,
	}, nil
}

// errorResponse is Luma's validation error body.
type errorResponse struct {
	Detail []errorDetail `json:"detail"`
}

type errorDetail struct {
	Type string   `json:"type"`
	Loc  []string `json:"loc"`
	Msg  string   `json:"msg"`
}

// parseError reads the first message of a Luma error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	if len(response.Detail) == 0 {
		return ""
	}

	return response.Detail[0].Msg
}

var _ model.Provider = (*Provider)(nil)
