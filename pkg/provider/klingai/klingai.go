// Package klingai implements the KlingAI provider for text-to-video,
// image-to-video, reference-to-video and motion control generation. It is
// a port of the Vercel AI SDK KlingAI provider.
package klingai

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the KlingAI API.
	defaultBaseURL = "https://api-singapore.klingai.com"

	// Namespace is the provider options and metadata namespace.
	Namespace = "klingai"
)

// Provider creates KlingAI models. Use New to create it.
type Provider struct {
	apiKey     string
	accessKey  string
	secretKey  string
	baseURL    string
	headers    http.Header
	httpClient *http.Client
	name       string
	now        func() time.Time
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key, which is sent as the bearer token. It takes
// precedence over the legacy access key and secret key. Without any
// credentials, the key is read from KLINGAI_API_KEY when a request is made.
// Surrounding whitespace is removed, and a blank key counts as unset.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = strings.TrimSpace(key)
	}
}

// WithAccessKey sets the legacy access key, which signs a short-lived JWT
// together with the secret key. Without it, KLINGAI_ACCESS_KEY is read.
// Surrounding whitespace is removed, and a blank key counts as unset.
func WithAccessKey(key string) Option {
	return func(p *Provider) {
		p.accessKey = strings.TrimSpace(key)
	}
}

// WithSecretKey sets the legacy secret key, which signs a short-lived JWT
// together with the access key. Without it, KLINGAI_SECRET_KEY is read.
// Surrounding whitespace is removed, and a blank key counts as unset.
func WithSecretKey(key string) Option {
	return func(p *Provider) {
		p.secretKey = strings.TrimSpace(key)
	}
}

// WithBaseURL replaces the API base URL, for example to use another region
// or a proxy.
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
// "klingai". Provider options and metadata keep the "klingai" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// New returns a KlingAI provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL: defaultBaseURL,
		name:    Namespace,
		now:     time.Now,
	}

	for _, option := range options {
		option(provider)
	}

	provider.baseURL = strings.TrimRight(provider.baseURL, "/")
	if provider.baseURL == "" {
		provider.baseURL = defaultBaseURL
	}

	return provider
}

// Video returns a video model. The ID suffix selects the mode: "-t2v" for
// text-to-video, "-i2v" for image-to-video and "-motion-control" for motion
// control, such as "kling-v2.6-t2v".
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

// LanguageModel reports that KlingAI has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that KlingAI has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that KlingAI has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that KlingAI has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that KlingAI has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that KlingAI has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// apiRequest describes a call to path with the bearer token, the provider
// headers and the call headers.
func (p *Provider) apiRequest(path string, callHeaders http.Header) (apicall.Request, error) {
	token, err := p.authToken()
	if err != nil {
		return apicall.Request{}, err
	}

	authorization := http.Header{
		"Authorization": {
			"Bearer " + token,
		},
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        p.baseURL + path,
		Headers:    apicall.MergeHeaders(authorization, p.headers, callHeaders),
		ParseError: parseError,
	}, nil
}

// ErrorData is a KlingAI error response.
type ErrorData struct {
	Code    float64 `json:"code"`
	Message string  `json:"message"`
}

// parseError reads the message of a KlingAI error response.
func parseError(body []byte) string {
	var data ErrorData
	if err := json.Unmarshal(body, &data); err != nil {
		return ""
	}

	return data.Message
}

var (
	_ model.Provider      = (*Provider)(nil)
	_ model.VideoProvider = (*Provider)(nil)
)
