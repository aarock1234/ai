// Package openaicompatible implements providers that speak the OpenAI chat
// completions, completions, embeddings and images APIs. It is a port of the
// Vercel AI SDK openai-compatible provider and is exported so that other
// providers can build on its models, hooks and stream helpers.
package openaicompatible

import (
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// Provider creates models for one OpenAI-compatible API. Use New to create
// it.
type Provider struct {
	name    string
	baseURL string

	apiKey      string
	headers     http.Header
	queryParams url.Values
	httpClient  *http.Client

	includeUsage                 bool
	supportsStructuredOutputs    bool
	supportsMultiPartToolContent bool
	supportedURLs                map[string][]*regexp.Regexp
	transformRequestBody         RequestBodyTransform
	metadataExtractor            MetadataExtractor
	convertUsage                 UsageConverter
	generateID                   func() string
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sends key as a bearer token in the Authorization header. Headers
// set with WithHeaders are applied after it and may replace it.
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

// WithQueryParams adds query parameters to every request URL. The values are
// copied.
func WithQueryParams(params url.Values) Option {
	snapshot := cloneValues(params)

	return func(p *Provider) {
		p.queryParams = cloneValues(snapshot)
	}
}

// WithHTTPClient sets the client used for every request. Nil uses
// http.DefaultClient.
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// WithIncludeUsage asks streaming chat and completion calls to report usage
// through stream_options.include_usage.
func WithIncludeUsage(include bool) Option {
	return func(p *Provider) {
		p.includeUsage = include
	}
}

// WithStructuredOutputs reports that chat models accept JSON schema response
// formats.
func WithStructuredOutputs(supported bool) Option {
	return func(p *Provider) {
		p.supportsStructuredOutputs = supported
	}
}

// WithMultiPartToolContent reports that chat models accept multi-part content
// in tool results. Without it, multi-part results are sent as JSON text.
func WithMultiPartToolContent(supported bool) Option {
	return func(p *Provider) {
		p.supportsMultiPartToolContent = supported
	}
}

// WithSupportedURLs sets the URLs that chat models download themselves, keyed
// by media type pattern.
func WithSupportedURLs(supportedURLs map[string][]*regexp.Regexp) Option {
	return func(p *Provider) {
		p.supportedURLs = supportedURLs
	}
}

// WithRequestBodyTransform rewrites every chat request body before it is
// sent, for proxies that expect a different request format.
func WithRequestBodyTransform(transform RequestBodyTransform) Option {
	return func(p *Provider) {
		p.transformRequestBody = transform
	}
}

// WithMetadataExtractor captures provider-specific metadata from chat
// responses and streams.
func WithMetadataExtractor(extractor MetadataExtractor) Option {
	return func(p *Provider) {
		p.metadataExtractor = extractor
	}
}

// WithUsageConverter replaces the default chat usage conversion, for
// providers whose token accounting differs from OpenAI's.
func WithUsageConverter(convert UsageConverter) Option {
	return func(p *Provider) {
		p.convertUsage = convert
	}
}

// WithIDGenerator sets the function that creates IDs for tool calls that
// arrive without one.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// New returns a provider named name that sends requests to baseURL. The name
// prefixes every model's Provider value, such as "name.chat", and selects the
// provider options namespace.
func New(name string, baseURL string, options ...Option) *Provider {
	provider := &Provider{
		name:    name,
		baseURL: strings.TrimRight(baseURL, "/"),
	}

	for _, option := range options {
		option(provider)
	}

	return provider
}

// Chat returns a chat completions language model.
func (p *Provider) Chat(modelID string) *ChatModel {
	return NewChatModel(modelID, ChatConfig{
		Provider:                     p.name + ".chat",
		BuildURL:                     p.url,
		Headers:                      p.requestHeaders,
		HTTPClient:                   p.httpClient,
		IncludeUsage:                 p.includeUsage,
		SupportsStructuredOutputs:    p.supportsStructuredOutputs,
		SupportsMultiPartToolContent: p.supportsMultiPartToolContent,
		SupportedURLs:                p.supportedURLs,
		TransformRequestBody:         p.transformRequestBody,
		MetadataExtractor:            p.metadataExtractor,
		ConvertUsage:                 p.convertUsage,
		GenerateID:                   p.generateID,
	})
}

// Completion returns a legacy completions language model.
func (p *Provider) Completion(modelID string) *CompletionModel {
	return NewCompletionModel(modelID, CompletionConfig{
		Provider:     p.name + ".completion",
		BuildURL:     p.url,
		Headers:      p.requestHeaders,
		HTTPClient:   p.httpClient,
		IncludeUsage: p.includeUsage,
	})
}

// Embedding returns an embedding model.
func (p *Provider) Embedding(modelID string) *EmbeddingModel {
	return NewEmbeddingModel(modelID, EmbeddingConfig{
		Provider:   p.name + ".embedding",
		BuildURL:   p.url,
		Headers:    p.requestHeaders,
		HTTPClient: p.httpClient,
	})
}

// Image returns an image generation and editing model.
func (p *Provider) Image(modelID string) *ImageModel {
	return NewImageModel(modelID, ImageConfig{
		Provider:   p.name + ".image",
		BuildURL:   p.url,
		Headers:    p.requestHeaders,
		HTTPClient: p.httpClient,
	})
}

// LanguageModel returns the chat model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID), nil
}

// EmbeddingModel returns the embedding model for modelID.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return p.Embedding(modelID), nil
}

// ImageModel returns the image model for modelID.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel reports that the provider has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that the provider has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that the provider has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// url joins the base URL and path and applies the configured query
// parameters.
func (p *Provider) url(path string) (string, error) {
	address := p.baseURL + path
	if len(p.queryParams) == 0 {
		return address, nil
	}

	return address + "?" + p.queryParams.Encode(), nil
}

// requestHeaders returns the authorization header followed by the configured
// headers.
func (p *Provider) requestHeaders() (http.Header, error) {
	headers := make(http.Header)
	if p.apiKey != "" {
		headers.Set("Authorization", "Bearer "+p.apiKey)
	}

	for key, values := range p.headers {
		headers[http.CanonicalHeaderKey(key)] = slices.Clone(values)
	}

	return headers, nil
}

func cloneValues(values url.Values) url.Values {
	if values == nil {
		return nil
	}

	cloned := make(url.Values, len(values))
	for key, entries := range values {
		cloned[key] = slices.Clone(entries)
	}

	return cloned
}

var _ model.Provider = (*Provider)(nil)
