// Package typesafeai implements the TypeSafe provider: System One decision
// models that answer choice, score, box and boolean questions about one
// shared state. Use them with ai.Decide:
//
//	result, err := ai.Decide(ctx, ai.DecideOptions{
//		Model: typesafeai.New().Decision("jev-latest"),
//		State: ...,
//		Questions: ...,
//	})
//
// Decision models are experimental. It is a port of the Vercel AI SDK
// TypeSafe provider.
package typesafeai

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the TypeSafe API.
	defaultBaseURL = "https://api.typesafe.ai/v1"

	// apiKeyVariable holds the API key.
	apiKeyVariable = "TYPESAFE_AI_API_KEY"

	// baseURLVariable holds the base URL.
	baseURLVariable = "TYPESAFE_AI_BASE_URL"

	// defaultName is the provider name reported by the models.
	defaultName = "typesafe"

	// Namespace is the provider options and metadata namespace.
	Namespace = "typesafe"
)

var (
	_ model.Provider         = (*Provider)(nil)
	_ model.DecisionProvider = (*Provider)(nil)
)

// Provider creates TypeSafe decision models. TypeSafe has no other model
// kinds. Use New to create it.
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

// WithAPIKey sets the API key. Without it, the key is read from
// TYPESAFE_AI_API_KEY when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the API base URL. Without it, the URL is read from
// TYPESAFE_AI_BASE_URL when the provider is created and then defaults to
// https://api.typesafe.ai/v1.
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
// "typesafe". Provider options and metadata keep the "typesafe" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithIDGenerator sets the function that creates the default asset
// references of images.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// New returns a TypeSafe provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		name: defaultName,
		generateID: func() string {
			return id.New("")
		},
	}

	for _, option := range options {
		option(provider)
	}

	provider.baseURL = resolveBaseURL(provider.baseURL)

	return provider
}

// Decision returns a System One decision model, such as "jev-latest".
func (p *Provider) Decision(modelID string) *DecisionModel {
	return &DecisionModel{
		modelID:  modelID,
		provider: p,
	}
}

// DecisionModel returns Decision(modelID).
func (p *Provider) DecisionModel(modelID string) (model.DecisionModel, error) {
	return p.Decision(modelID), nil
}

// LanguageModel reports that TypeSafe has no language models.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.LanguageModelKind, modelID)
}

// EmbeddingModel reports that TypeSafe has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that TypeSafe has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that TypeSafe has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that TypeSafe has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that TypeSafe has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// resolveBaseURL returns the configured, environment or default base URL
// without a trailing slash.
func resolveBaseURL(configured string) string {
	baseURL := configured
	if baseURL == "" {
		baseURL = os.Getenv(baseURLVariable)
	}

	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	return strings.TrimRight(baseURL, "/")
}

// apiRequest describes a call to path. Later headers take precedence:
// authorization, provider, feature and call headers.
func (p *Provider) apiRequest(path string, featureHeaders http.Header, callHeaders http.Header) (apicall.Request, error) {
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
		Headers:    apicall.MergeHeaders(authorization, p.headers, featureHeaders, callHeaders),
		ParseError: parseError,
	}, nil
}

// errorBody is a TypeSafe error response. Error is a string or an object
// with a message; Detail is any JSON value.
type errorBody struct {
	Message   *string         `json:"message"`
	Detail    json.RawMessage `json:"detail"`
	Error     json.RawMessage `json:"error"`
	ErrorType *string         `json:"error_type"`
}

type errorObject struct {
	Message *string `json:"message"`
}

// parseError reads the message, the error, the detail or the error type of
// a TypeSafe error response, in that order.
func parseError(body []byte) string {
	var decoded errorBody
	if err := json.Unmarshal(body, &decoded); err != nil {
		return ""
	}

	if decoded.Message != nil {
		return *decoded.Message
	}

	if message, ok := errorMessage(decoded.Error); ok {
		return message
	}

	if message, ok := detailMessage(decoded.Detail); ok {
		return message
	}

	if decoded.ErrorType != nil {
		return *decoded.ErrorType
	}

	return "TypeSafe request failed"
}

func errorMessage(raw json.RawMessage) (string, bool) {
	if !wirejson.HasValue(raw) {
		return "", false
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, true
	}

	var object errorObject
	if err := json.Unmarshal(raw, &object); err == nil && object.Message != nil {
		return *object.Message, true
	}

	return "", false
}

// detailMessage returns a string detail, or any other detail as JSON.
func detailMessage(raw json.RawMessage) (string, bool) {
	if !wirejson.HasValue(raw) {
		return "", false
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, true
	}

	var compact bytes.Buffer
	if err := json.Compact(&compact, raw); err != nil {
		return string(raw), true
	}

	return compact.String(), true
}
