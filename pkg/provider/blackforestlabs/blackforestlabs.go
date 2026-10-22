// Package blackforestlabs implements the Black Forest Labs provider for FLUX
// image generation and editing and FLUX 3 video generation. Generations are
// queued and polled until ready. It is a port of the Vercel AI SDK Black
// Forest Labs provider.
package blackforestlabs

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the Black Forest Labs API.
	defaultBaseURL = "https://api.bfl.ai/v1"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "BFL_API_KEY"

	// defaultName is the provider name reported by the models.
	defaultName = "black-forest-labs"

	// Namespace is the provider options and metadata namespace.
	Namespace = "blackForestLabs"
)

// Provider creates Black Forest Labs models. Use New to create it.
type Provider struct {
	apiKey       string
	baseURL      string
	headers      http.Header
	httpClient   *http.Client
	name         string
	pollInterval time.Duration
	pollTimeout  time.Duration
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from BFL_API_KEY
// when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the API base URL, for example to use a regional
// endpoint or a proxy.
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
// "black-forest-labs". Provider options keep the "blackForestLabs"
// namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithPollInterval sets the delay between status checks. The default is
// 500 milliseconds for images and 2 seconds for videos.
// ImageOptions.PollIntervalMillis overrides it per image call.
func WithPollInterval(interval time.Duration) Option {
	return func(p *Provider) {
		p.pollInterval = interval
	}
}

// WithPollTimeout sets how long to wait for a generation. The default is 60
// seconds for images and 10 minutes for videos waited for with Generate.
// ImageOptions.PollTimeoutMillis overrides it per image call.
func WithPollTimeout(timeout time.Duration) Option {
	return func(p *Provider) {
		p.pollTimeout = timeout
	}
}

// New returns a Black Forest Labs provider.
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

// Image returns an image model, such as "flux-pro-1.1" or "flux-3-image".
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel reports that Black Forest Labs has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that Black Forest Labs has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel returns the image model for modelID.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel reports that Black Forest Labs has no transcription
// models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Black Forest Labs has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Black Forest Labs has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// authHeaders returns the provider and call headers with the API key.
func (p *Provider) authHeaders(callHeaders http.Header) (http.Header, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return nil, err
	}

	key := http.Header{
		"X-Key": {
			apiKey,
		},
	}

	return apicall.MergeHeaders(key, p.headers, callHeaders), nil
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

// taskStatus is the status of a queued generation.
type taskStatus string

// Generation statuses.
const (
	statusContentModerated taskStatus = "Content Moderated"
	statusGenerating       taskStatus = "Generating"
	statusPending          taskStatus = "Pending"
	statusReasoning        taskStatus = "Reasoning"
	statusReady            taskStatus = "Ready"
	statusError            taskStatus = "Error"
	statusFailed           taskStatus = "Failed"
	statusRequestModerated taskStatus = "Request Moderated"
	statusTaskNotFound     taskStatus = "Task not found"
)

// terminalStatuses are failed generations. Retrying the request starts a new
// generation, so failures with these statuses are not retryable.
var terminalStatuses = []taskStatus{
	statusContentModerated,
	statusError,
	statusFailed,
	statusRequestModerated,
	statusTaskNotFound,
}

// errorResponse is the Black Forest Labs error body.
type errorResponse struct {
	Message *string         `json:"message"`
	Detail  json.RawMessage `json:"detail"`
	Status  *taskStatus     `json:"status"`
	State   *taskStatus     `json:"state"`
}

// status returns the generation status reported with the error, if any.
func (r errorResponse) status() (taskStatus, bool) {
	if r.Status != nil {
		return *r.Status, true
	}

	if r.State != nil {
		return *r.State, true
	}

	return "", false
}

// parseError reads the detail or message of an error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	detail := bytes.TrimSpace(response.Detail)
	if len(detail) > 0 && !bytes.Equal(detail, []byte("null")) {
		var text string
		if err := json.Unmarshal(detail, &text); err == nil {
			return text
		}

		return string(detail)
	}

	if response.Message != nil {
		return *response.Message
	}

	if status, ok := response.status(); ok {
		return fmt.Sprintf("Black Forest Labs generation failed: %s.", status)
	}

	return "Unknown Black Forest Labs error"
}

// classifyError marks failed generations as not retryable. They can use
// server error statuses, but resubmitting starts a new generation.
func classifyError(err error) error {
	var apiErr *model.APICallError
	if !errors.As(err, &apiErr) {
		return err
	}

	var response errorResponse
	if json.Unmarshal([]byte(apiErr.ResponseBody), &response) != nil {
		return err
	}

	if status, ok := response.status(); ok && slices.Contains(terminalStatuses, status) {
		apiErr.IsRetryable = false
	}

	return err
}

var _ model.Provider = (*Provider)(nil)
