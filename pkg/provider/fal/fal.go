// Package fal implements the fal.ai provider: image generation and editing,
// speech, Whisper transcription and queued video generation. It is a port of
// the Vercel AI SDK fal provider.
package fal

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the synchronous fal.ai API used by image models.
	defaultBaseURL = "https://fal.run"

	// speechBaseURL and queueBaseURL are the fixed endpoints of speech and
	// transcription models, which do not use the configured base URL.
	speechBaseURL = "https://fal.run"
	queueBaseURL  = "https://queue.fal.run/fal-ai"

	// apiKeyVariable and legacyAPIKeyVariable hold the API key, in order of
	// preference.
	apiKeyVariable       = "FAL_API_KEY"
	legacyAPIKeyVariable = "FAL_KEY"

	// Namespace is the provider options and metadata namespace.
	Namespace = "fal"
)

// Provider creates fal.ai models. Use New to create it.
type Provider struct {
	apiKey     string
	baseURL    string
	headers    http.Header
	httpClient *http.Client
	name       string
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key. Without it, the key is read from FAL_API_KEY,
// then FAL_KEY, when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the base URL of image models, for example to use a
// proxy. Speech, transcription and video models always call fal.ai directly.
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
// "fal". Provider options and metadata keep the "fal" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// New returns a fal.ai provider.
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

// Image returns an image model, such as "fal-ai/flux/dev".
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// Speech returns a text-to-speech model, such as
// "fal-ai/minimax/speech-02-hd".
func (p *Provider) Speech(modelID string) *SpeechModel {
	return &SpeechModel{
		modelID:  modelID,
		provider: p,
	}
}

// Transcription returns a transcription model: "whisper" or "wizper".
func (p *Provider) Transcription(modelID string) *TranscriptionModel {
	return &TranscriptionModel{
		modelID:  modelID,
		provider: p,
	}
}

// Video returns a video model, such as "luma-dream-machine". A "fal-ai/" or
// "fal/" prefix is ignored. Video models always use the fal.ai queue.
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

// LanguageModel reports that fal.ai has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that fal.ai has no embedding models.
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

// SpeechModel returns the speech model for modelID.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return p.Speech(modelID), nil
}

// RerankingModel reports that fal.ai has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// apiRequest describes a call to url with the provider and call headers.
func (p *Provider) apiRequest(url string, callHeaders http.Header, parse apicall.ErrorParser) (apicall.Request, error) {
	apiKey, err := p.loadAPIKey()
	if err != nil {
		return apicall.Request{}, err
	}

	authorization := http.Header{
		"Authorization": {
			"Key " + apiKey,
		},
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        url,
		Headers:    apicall.MergeHeaders(authorization, p.headers, callHeaders),
		ParseError: parse,
	}, nil
}

func (p *Provider) loadAPIKey() (string, error) {
	return apikey.Load(p.apiKey, apiKeyVariable, legacyAPIKeyVariable)
}

// errorResponse is the error body of speech and transcription endpoints.
type errorResponse struct {
	Error *errorDetail `json:"error"`
}

type errorDetail struct {
	Message string `json:"message"`
	Code    int    `json:"code"`
}

// parseError reads the message of a fal.ai error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil || response.Error == nil {
		return ""
	}

	return response.Error.Message
}

// imageErrorResponse is the error body of image endpoints: validation
// issues or a message.
type imageErrorResponse struct {
	Detail  []validationIssue `json:"detail"`
	Message *string           `json:"message"`
}

type validationIssue struct {
	Loc  []string `json:"loc"`
	Msg  string   `json:"msg"`
	Type string   `json:"type"`
}

// parseImageError reads validation issues, one per line, or the message of
// an image error response.
func parseImageError(body []byte) string {
	var response imageErrorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	if len(response.Detail) > 0 {
		lines := make([]string, 0, len(response.Detail))
		for _, issue := range response.Detail {
			lines = append(lines, strings.Join(issue.Loc, ".")+": "+issue.Msg)
		}

		return strings.Join(lines, "\n")
	}

	if response.Message != nil {
		return *response.Message
	}

	return "Unknown fal error"
}

var (
	_ model.Provider      = (*Provider)(nil)
	_ model.VideoProvider = (*Provider)(nil)
)
