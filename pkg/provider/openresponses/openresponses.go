// Package openresponses implements providers for endpoints that speak the
// Open Responses API, an open specification of the OpenAI Responses API. It
// is a port of the Vercel AI SDK Open Responses provider.
package openresponses

import (
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// defaultCustomToolID is the provider tool ID of caller-executed custom
// tools.
const defaultCustomToolID = "open-responses.custom"

// ResponseError is an error reported inside a response body or stream.
type ResponseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`

	// Raw is the error object as the endpoint sent it, including fields
	// beyond the code and message.
	Raw []byte `json:"-"`
}

// Error returns the endpoint's message.
func (e *ResponseError) Error() string {
	return e.Message
}

// ResponseErrorMetadata is the HTTP status and retryability that an
// endpoint-specific response error implies.
type ResponseErrorMetadata struct {
	// StatusCode is zero to use the default status.
	StatusCode int

	// IsRetryable is nil to use the default retryability.
	IsRetryable *bool
}

// ResponseErrorClassifier extracts HTTP status and retryability metadata
// from an endpoint-specific response error.
type ResponseErrorClassifier func(err ResponseError) ResponseErrorMetadata

// Provider creates Open Responses models for one endpoint. Use New to create
// it.
type Provider struct {
	name string
	url  string

	apiKey     string
	headers    http.Header
	httpClient *http.Client

	parseError          apicall.ErrorParser
	classifyError       ResponseErrorClassifier
	strictResponseInput bool
	customToolID        string
	structuredOutputs   bool
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sends key as a bearer token. Without it, no Authorization
// header is sent. Headers set with WithHeaders may replace it.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
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

// WithErrorParser replaces the reading of failed response bodies. The
// default reads {"error": {"message": ...}}.
func WithErrorParser(parse func(body []byte) string) Option {
	return func(p *Provider) {
		p.parseError = parse
	}
}

// WithResponseErrorClassifier sets how response errors inside successful
// responses and streams map to an HTTP status and retryability.
func WithResponseErrorClassifier(classify ResponseErrorClassifier) Option {
	return func(p *Provider) {
		p.classifyError = classify
	}
}

// WithStrictResponseInput serializes assistant history with the strict
// OpenAI Responses input schemas: assistant messages without an item ID are
// sent as easy input messages, and messages with an item ID as complete
// output items.
func WithStrictResponseInput(strict bool) Option {
	return func(p *Provider) {
		p.strictResponseInput = strict
	}
}

// WithCustomToolID replaces the provider tool ID, "open-responses.custom" by
// default, of caller-executed custom tools. It has the form
// "implementor.tool".
func WithCustomToolID(toolID string) Option {
	return func(p *Provider) {
		p.customToolID = toolID
	}
}

// WithStructuredOutputs controls whether JSON response formats are sent.
// When disabled, they produce an unsupported warning and are omitted. It is
// enabled by default.
func WithStructuredOutputs(enabled bool) Option {
	return func(p *Provider) {
		p.structuredOutputs = enabled
	}
}

// New returns a provider named name that posts to the Open Responses
// endpoint url. The name prefixes every model's Provider value, as in
// "name.responses", and is the provider options and metadata namespace.
func New(name string, url string, options ...Option) *Provider {
	provider := &Provider{
		name:              name,
		url:               url,
		parseError:        parseError,
		customToolID:      defaultCustomToolID,
		structuredOutputs: true,
	}

	for _, option := range options {
		option(provider)
	}

	return provider
}

// Responses returns a Responses API language model.
func (p *Provider) Responses(modelID string) *ResponsesModel {
	return &ResponsesModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel returns the Responses API model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Responses(modelID), nil
}

// EmbeddingModel reports that Open Responses has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that Open Responses has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that Open Responses has no transcription
// models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Open Responses has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Open Responses has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// apiRequest describes a call to the endpoint with the provider and call
// headers.
func (p *Provider) apiRequest(callHeaders http.Header) apicall.Request {
	var authorization http.Header
	if p.apiKey != "" {
		authorization = http.Header{
			"Authorization": {
				"Bearer " + p.apiKey,
			},
		}
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        p.url,
		Headers:    apicall.MergeHeaders(authorization, p.headers, callHeaders),
		ParseError: p.parseError,
	}
}

// classify returns the metadata of a response error.
func (p *Provider) classify(err ResponseError) ResponseErrorMetadata {
	if p.classifyError == nil {
		return ResponseErrorMetadata{}
	}

	return p.classifyError(err)
}

var _ model.Provider = (*Provider)(nil)
