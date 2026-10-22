// Package topaz implements the Topaz Labs provider for image and video
// enhancement. Image jobs are submitted, polled until complete and
// downloaded; video requests are started and checked through the video
// start/status flow. It is a port of the Vercel AI SDK Topaz provider.
package topaz

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the Topaz Labs API.
	defaultBaseURL = "https://api.topazlabs.com"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "TOPAZ_API_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "topaz"
)

// Provider creates Topaz Labs models. Use New to create it.
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
// TOPAZ_API_KEY when a request is made.
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
// "topaz". Provider options and metadata keep the "topaz" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// New returns a Topaz Labs provider.
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

// Image returns an image enhancement model, such as "wonder-3.5". Readable
// IDs are mapped onto Topaz API model names; other IDs, such as
// "Wonder 3.5", are sent unchanged.
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel reports that Topaz Labs has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that Topaz Labs has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel returns the image model for modelID.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel reports that Topaz Labs has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Topaz Labs has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Topaz Labs has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// authHeaders returns the provider and call headers with the API key.
func (p *Provider) authHeaders(callHeaders http.Header) (http.Header, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return nil, err
	}

	defaults := http.Header{
		"X-Api-Key": {
			apiKey,
		},
		"Accept": {
			"application/json",
		},
	}

	return apicall.MergeHeaders(defaults, p.headers, callHeaders), nil
}

// client returns the configured HTTP client, or http.DefaultClient.
func (p *Provider) client() *http.Client {
	if p.httpClient == nil {
		return http.DefaultClient
	}

	return p.httpClient
}

// apiRequest describes a call to path with headers.
func (p *Provider) apiRequest(path string, headers http.Header) apicall.Request {
	return apicall.Request{
		Client:     p.client(),
		URL:        p.baseURL + path,
		Headers:    headers,
		ParseError: parseError,
	}
}

// errorResponse is the Topaz error body. The image API reports a message,
// the video API a message and code, and some endpoints a detail or error.
type errorResponse struct {
	Detail    json.RawMessage `json:"detail"`
	Message   *string         `json:"message"`
	Error     *string         `json:"error"`
	ErrorCode *string         `json:"errorCode"`
	Errors    []errorIssue    `json:"errors"`
}

type errorIssue struct {
	Msg *string `json:"msg"`
}

// parseError describes a Topaz error response with its code and field
// issues.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	message := baseErrorMessage(response)
	if response.ErrorCode != nil {
		message = fmt.Sprintf("%s (%s)", message, *response.ErrorCode)
	}

	issues := issueMessages(response.Errors)
	if len(issues) > 0 {
		return message + ": " + strings.Join(issues, "; ")
	}

	return message
}

func baseErrorMessage(response errorResponse) string {
	var detail string
	if json.Unmarshal(response.Detail, &detail) == nil {
		return detail
	}

	var issues []errorIssue
	if json.Unmarshal(response.Detail, &issues) == nil {
		if messages := issueMessages(issues); len(messages) > 0 {
			return strings.Join(messages, "; ")
		}
	}

	switch {
	case response.Message != nil:
		return *response.Message
	case response.Error != nil:
		return *response.Error
	default:
		return "Unknown Topaz API error"
	}
}

func issueMessages(issues []errorIssue) []string {
	var messages []string
	for _, issue := range issues {
		if issue.Msg != nil {
			messages = append(messages, *issue.Msg)
		}
	}

	return messages
}

var _ model.Provider = (*Provider)(nil)
