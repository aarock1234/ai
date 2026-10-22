// Package prodia implements the Prodia provider: fast image and video
// generation and a multimodal language model that edits images. Prodia
// answers each job with a multipart response holding the job record and its
// outputs. It is a port of the Vercel AI SDK Prodia provider.
package prodia

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the Prodia inference API.
	defaultBaseURL = "https://inference.prodia.com/v2"

	// apiKeyVariable is the environment variable that holds the API token.
	apiKeyVariable = "PRODIA_TOKEN"

	// jobPath creates a job and reports its price.
	jobPath = "/job?price=true"

	// Namespace is the provider options and metadata namespace.
	Namespace = "prodia"
)

// Provider creates Prodia models. Use New to create it.
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

// WithAPIKey sets the API token. Without it, the token is read from
// PRODIA_TOKEN when a request is made.
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
// "prodia". Provider options and metadata keep the "prodia" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithIDGenerator sets the function that creates the IDs of streamed text.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// New returns a Prodia provider.
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

// Chat returns a multimodal language model, such as
// "inference.nano-banana.img2img.v2", that edits the image of the last user
// message and returns text and images.
func (p *Provider) Chat(modelID string) *ChatModel {
	return &ChatModel{
		modelID:  modelID,
		provider: p,
	}
}

// Image returns an image model, such as
// "inference.flux-fast.schnell.txt2img.v2".
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel returns the chat model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID), nil
}

// EmbeddingModel reports that Prodia has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel returns the image model for modelID.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel reports that Prodia has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Prodia has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Prodia has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// jobRequest describes a job creation call that accepts the given response
// media types.
func (p *Provider) jobRequest(accept string, callHeaders http.Header) (apicall.Request, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return apicall.Request{}, err
	}

	authorization := http.Header{
		"Authorization": {
			"Bearer " + apiKey,
		},
	}

	acceptHeader := http.Header{
		"Accept": {
			accept,
		},
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        p.baseURL + jobPath,
		Headers:    apicall.MergeHeaders(authorization, p.headers, callHeaders, acceptHeader),
		ParseError: parseError,
	}, nil
}

// errorResponse is Prodia's error body.
type errorResponse struct {
	Message *string         `json:"message"`
	Detail  json.RawMessage `json:"detail"`
	Error   *string         `json:"error"`
}

// parseError reads the detail, error or message of a Prodia error response.
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

	switch {
	case response.Error != nil:
		return *response.Error
	case response.Message != nil:
		return *response.Message
	default:
		return "Unknown Prodia error"
	}
}

// job is the job record of a completed job.
type job struct {
	ID        string      `json:"id"`
	CreatedAt *string     `json:"created_at"`
	UpdatedAt *string     `json:"updated_at"`
	ExpiresAt *string     `json:"expires_at"`
	State     *jobState   `json:"state"`
	Config    *jobConfig  `json:"config"`
	Metrics   *jobMetrics `json:"metrics"`
	Price     *jobPrice   `json:"price"`
}

type jobState struct {
	Current string `json:"current"`
}

type jobConfig struct {
	Seed *float64 `json:"seed"`
}

type jobMetrics struct {
	Elapsed *float64 `json:"elapsed"`
	IPS     *float64 `json:"ips"`
}

type jobPrice struct {
	Product string  `json:"product"`
	Dollars float64 `json:"dollars"`
}

// JobMetadata describes a completed Prodia job. Nil fields were not
// reported.
type JobMetadata struct {
	JobID               string   `json:"jobId"`
	Seed                *float64 `json:"seed,omitempty"`
	Elapsed             *float64 `json:"elapsed,omitempty"`
	IterationsPerSecond *float64 `json:"iterationsPerSecond,omitempty"`
	CreatedAt           *string  `json:"createdAt,omitempty"`
	UpdatedAt           *string  `json:"updatedAt,omitempty"`
	Dollars             *float64 `json:"dollars,omitempty"`
}

// jobMetadata summarizes the job record.
func jobMetadata(record job) JobMetadata {
	metadata := JobMetadata{
		JobID:     record.ID,
		CreatedAt: record.CreatedAt,
		UpdatedAt: record.UpdatedAt,
	}

	if record.Config != nil {
		metadata.Seed = record.Config.Seed
	}

	if record.Metrics != nil {
		metadata.Elapsed = record.Metrics.Elapsed
		metadata.IterationsPerSecond = record.Metrics.IPS
	}

	if record.Price != nil {
		metadata.Dollars = new(record.Price.Dollars)
	}

	return metadata
}

var _ model.Provider = (*Provider)(nil)
