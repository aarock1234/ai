// Package replicate implements the Replicate provider for image and video
// generation with any model hosted on Replicate. Image predictions wait
// synchronously first and are polled until they finish; video predictions
// use the start/status flow or a webhook. It is a port of the Vercel AI SDK
// Replicate provider.
package replicate

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/origin"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the Replicate API.
	defaultBaseURL = "https://api.replicate.com/v1"

	// apiKeyVariable is the environment variable that holds the API token.
	apiKeyVariable = "REPLICATE_API_TOKEN"

	// Namespace is the provider options and metadata namespace.
	Namespace = "replicate"
)

// Provider creates Replicate models. Use New to create it.
type Provider struct {
	apiKey     string
	baseURL    string
	headers    http.Header
	httpClient *http.Client
	name       string
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API token. Without it, the token is read from
// REPLICATE_API_TOKEN when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the API base URL, for example to use a proxy. An
// empty URL keeps the default.
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
// "replicate". Provider options keep the "replicate" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// New returns a Replicate provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL: defaultBaseURL,
		name:    Namespace,
	}

	for _, option := range options {
		option(provider)
	}

	provider.baseURL = strings.TrimRight(provider.baseURL, "/")
	if strings.TrimSpace(provider.baseURL) == "" {
		provider.baseURL = defaultBaseURL
	}

	return provider
}

// Image returns an image model. The ID is "owner/name" for the latest
// version of a model or "owner/name:version" for a specific version.
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel reports that Replicate has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that Replicate has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel returns the image model for modelID.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel reports that Replicate has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Replicate has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// Video returns a video model. The ID is "owner/name" for the latest version
// of a model or "owner/name:version" for a specific version.
func (p *Provider) Video(modelID string) *VideoModel {
	return &VideoModel{
		modelID:  modelID,
		provider: p,
	}
}

// VideoModel returns Video(modelID).
func (p *Provider) VideoModel(modelID string) (model.VideoModel, error) {
	return p.Video(modelID), nil
}

// RerankingModel reports that Replicate has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// authHeaders returns the provider and call headers with the API token.
func (p *Provider) authHeaders(callHeaders http.Header) (http.Header, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return nil, err
	}

	authorization := http.Header{
		"Authorization": {
			"Bearer " + apiKey,
		},
	}

	return apicall.MergeHeaders(authorization, p.headers, callHeaders), nil
}

// apiRequest describes a call to url with headers.
func (p *Provider) apiRequest(url string, headers http.Header) apicall.Request {
	return apicall.Request{
		Client:     p.httpClient,
		URL:        url,
		Headers:    headers,
		ParseError: parseError,
	}
}

// fetchStatus reads a prediction at location, which comes from a response or
// an operation reference, and returns the response headers and body.
// Credentials are sent only to the API origin; other URLs are validated and
// receive only non-credential headers.
func (p *Provider) fetchStatus(ctx context.Context, location string, headers http.Header) (http.Header, []byte, error) {
	if origin.Same(location, p.baseURL) {
		response, err := apicall.Get(ctx, p.apiRequest(location, headers))
		if err != nil {
			return nil, nil, err
		}

		return response.Headers, response.Body, nil
	}

	downloaded, err := download.Fetch(ctx, download.Request{
		URL:                location,
		Headers:            headers,
		CredentialedOrigin: p.baseURL,
		TrustedOrigin:      p.baseURL,
		Client:             p.httpClient,
	})
	if err != nil {
		return nil, nil, err
	}

	return nil, downloaded.Data, nil
}

// errorResponse is Replicate's error body.
type errorResponse struct {
	Detail *string `json:"detail"`
	Error  *string `json:"error"`
}

// parseError reads the message of a Replicate error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	switch {
	case response.Detail != nil:
		return *response.Detail
	case response.Error != nil:
		return *response.Error
	default:
		return "Unknown Replicate error"
	}
}

var (
	_ model.Provider      = (*Provider)(nil)
	_ model.VideoProvider = (*Provider)(nil)
)
