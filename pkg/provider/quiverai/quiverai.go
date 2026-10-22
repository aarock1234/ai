// Package quiverai implements the QuiverAI provider: SVG generation,
// vectorization, animation and editing with the Arrow image models, and
// Arrow language models served by the Open Responses API. It is a port of
// the Vercel AI SDK QuiverAI provider and builds its language models on the
// openresponses package.
package quiverai

import (
	"encoding/json"
	"math"
	"net/http"
	"os"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openresponses"
)

const (
	// defaultBaseURL is the QuiverAI API.
	defaultBaseURL = "https://api.quiver.ai/v1"

	// apiKeyVariable is the environment variable that holds the API key.
	apiKeyVariable = "QUIVERAI_API_KEY"

	// baseURLVariable is the environment variable that overrides the base
	// URL.
	baseURLVariable = "QUIVERAI_BASE_URL"

	// customToolID is the provider tool ID of caller-executed custom tools.
	customToolID = "quiverai.custom"

	// Namespace is the provider options and metadata namespace.
	Namespace = "quiverai"
)

// Provider creates QuiverAI models. Use New to create it.
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
// QUIVERAI_API_KEY when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the API base URL. Without it, the base URL is read
// from QUIVERAI_BASE_URL, falling back to the QuiverAI API.
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
// "quiverai". Provider options and metadata keep the "quiverai" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// New returns a QuiverAI provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		name: Namespace,
	}

	for _, option := range options {
		option(provider)
	}

	if provider.baseURL == "" {
		provider.baseURL = os.Getenv(baseURLVariable)
	}

	provider.baseURL = strings.TrimRight(provider.baseURL, "/")
	if provider.baseURL == "" {
		provider.baseURL = defaultBaseURL
	}

	return provider
}

// Image returns an SVG image model, such as "arrow-2".
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// Responses returns an Open Responses API language model, such as "arrow-2"
// or "arrow-2-telos".
func (p *Provider) Responses(modelID string) *ResponsesModel {
	return &ResponsesModel{
		modelID:  modelID,
		provider: p,
	}
}

// CustomTool returns a caller-executed custom tool named name. Custom tools
// accept a raw string, optionally constrained by a grammar.
func (p *Provider) CustomTool(name string, args openresponses.CustomToolArgs) model.ProviderTool {
	return p.responsesProvider("").CustomTool(name, args)
}

// LanguageModel returns the Responses API model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Responses(modelID), nil
}

// EmbeddingModel reports that QuiverAI has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel returns the image model for modelID.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel reports that QuiverAI has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that QuiverAI has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that QuiverAI has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// loadAPIKey returns the configured API key or reads QUIVERAI_API_KEY.
func (p *Provider) loadAPIKey() (string, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return "", err
	}

	return apiKey, nil
}

// responsesProvider returns the Open Responses provider of the QuiverAI
// endpoint, authenticated with apiKey when it is not empty. QuiverAI accepts
// only strict response input and no JSON response formats.
func (p *Provider) responsesProvider(apiKey string) *openresponses.Provider {
	return openresponses.New(
		Namespace,
		p.baseURL+"/responses",
		openresponses.WithAPIKey(apiKey),
		openresponses.WithHeaders(p.headers),
		openresponses.WithHTTPClient(p.httpClient),
		openresponses.WithErrorParser(parseError),
		openresponses.WithResponseErrorClassifier(classifyResponseError),
		openresponses.WithStrictResponseInput(true),
		openresponses.WithCustomToolID(customToolID),
		openresponses.WithStructuredOutputs(false),
	)
}

// apiRequest describes a call to path with the provider and call headers.
func (p *Provider) apiRequest(path string, callHeaders http.Header) (apicall.Request, error) {
	apiKey, err := p.loadAPIKey()
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

// errorResponse is the QuiverAI error body.
type errorResponse struct {
	Status    int    `json:"status"`
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

// parseError reads the message of a QuiverAI error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	return response.Message
}

// responseErrorStatus is the HTTP status that QuiverAI reports inside
// response errors.
type responseErrorStatus struct {
	StatusCode *float64 `json:"status_code"`
}

// classifyResponseError uses the HTTP status of a response error when it is
// an integer client or server error status.
func classifyResponseError(responseError openresponses.ResponseError) openresponses.ResponseErrorMetadata {
	var status responseErrorStatus
	if err := json.Unmarshal(responseError.Raw, &status); err != nil || status.StatusCode == nil {
		return openresponses.ResponseErrorMetadata{}
	}

	code := *status.StatusCode
	isStatus := code == math.Trunc(code) && code >= 400 && code <= 599
	if !isStatus {
		return openresponses.ResponseErrorMetadata{}
	}

	return openresponses.ResponseErrorMetadata{
		StatusCode: int(code),
	}
}

var _ model.Provider = (*Provider)(nil)
