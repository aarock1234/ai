// Package revai implements the Rev.ai provider: asynchronous transcription,
// which submits a job with the audio, polls it until it is transcribed and
// reads the transcript. It is a port of the Vercel AI SDK Rev.ai provider.
package revai

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the Rev.ai API.
	defaultBaseURL = "https://api.rev.ai"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "REVAI_API_KEY"

	// Namespace is the provider options namespace.
	Namespace = "revai"
)

// Transcription model IDs.
const (
	ModelMachine = "machine"
	ModelLowCost = "low_cost"
	ModelFusion  = "fusion"
)

var (
	// ErrJobSubmissionFailed reports a job that Rev.ai rejected on
	// submission.
	ErrJobSubmissionFailed = errors.New("failed to submit transcription job to Rev.ai")

	// ErrTranscriptionFailed reports a job that Rev.ai finished with a
	// failed status.
	ErrTranscriptionFailed = errors.New("rev.ai transcription failed")

	// ErrTranscriptionTimeout reports a job that was not transcribed within
	// the polling timeout of one minute.
	ErrTranscriptionTimeout = errors.New("rev.ai transcription timed out")
)

// Provider creates Rev.ai models. Use New to create it.
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
// REVAI_API_KEY when a request is made.
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
// "revai". Provider options keep the "revai" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// New returns a Rev.ai provider.
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

// Transcription returns a transcription model: ModelMachine, ModelLowCost
// or ModelFusion. The ID is sent as the transcriber.
func (p *Provider) Transcription(modelID string) *TranscriptionModel {
	return &TranscriptionModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel reports that Rev.ai has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that Rev.ai has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that Rev.ai has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel returns the transcription model for modelID.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return p.Transcription(modelID), nil
}

// SpeechModel reports that Rev.ai has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Rev.ai has no reranking models.
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

// errorResponse is Rev.ai's error body.
type errorResponse struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Message string `json:"message"`
	Code    int    `json:"code"`
}

// parseError reads the message of a Rev.ai error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	return response.Error.Message
}

var _ model.Provider = (*Provider)(nil)
