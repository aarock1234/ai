// Package google provides Google Generative AI (Gemini API) models: Gemini
// language models with thinking, grounding and provider-executed tools,
// Interactions API language models and agents, embedding models, Gemini
// image models, Gemini text-to-speech, unary and live transcription, Veo
// video generation, the Files API, and the experimental batch, decision,
// realtime (Live API) and speech translation models.
package google

import (
	"context"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	defaultBaseURL      = "https://generativelanguage.googleapis.com/v1beta"
	defaultProviderName = "google.generative-ai"

	// defaultProviderSuffix ends the default provider name. The batch and
	// decision identifiers drop it, such as "google.batch".
	defaultProviderSuffix = ".generative-ai"
	apiKeyEnvironment     = "GOOGLE_GENERATIVE_AI_API_KEY"
	apiKeyHeader          = "X-Goog-Api-Key"

	// Namespace keys Google provider options and metadata.
	Namespace = "google"
)

var _ model.Provider = (*Provider)(nil)

// Provider creates Google Generative AI models. It is immutable after New and
// safe for concurrent use.
type Provider struct {
	apiKey     string
	baseURL    string
	headers    http.Header
	httpClient *http.Client
	name       string
	generateID func() string

	// resolveBaseURL replaces baseURL when set.
	resolveBaseURL func() (string, error)

	// authHeaders replaces API key authentication when set.
	authHeaders AuthHeadersFunc

	// vertex enables the Vertex AI dialect when set.
	vertex *VertexAISettings

	// batchMaxLineBytes bounds one line of a batch result file. Zero
	// selects 64 MiB.
	batchMaxLineBytes int
}

// AuthHeadersFunc returns the authentication headers of one request, such
// as an OAuth bearer token. Its error is returned from the call.
type AuthHeadersFunc func(ctx context.Context) (http.Header, error)

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key sent in the x-goog-api-key header. Without it,
// the provider reads GOOGLE_GENERATIVE_AI_API_KEY when a request is made.
func WithAPIKey(apiKey string) Option {
	return func(p *Provider) {
		p.apiKey = apiKey
	}
}

// WithBaseURL replaces the API URL prefix, for example to use a proxy. The
// default is https://generativelanguage.googleapis.com/v1beta.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.baseURL = strings.TrimRight(baseURL, "/")
	}
}

// WithBaseURLResolver resolves the API URL prefix when each request is made,
// for prefixes that depend on settings read lazily, such as a Vertex AI
// project and location. Its error is returned from the call. It replaces
// WithBaseURL.
func WithBaseURLResolver(resolve func() (string, error)) Option {
	return func(p *Provider) {
		p.resolveBaseURL = resolve
	}
}

// WithAuthHeaders replaces API key authentication. authHeaders is called for
// every request.
func WithAuthHeaders(authHeaders AuthHeadersFunc) Option {
	return func(p *Provider) {
		p.authHeaders = authHeaders
	}
}

// WithHeaders adds headers to every request. The headers are copied.
func WithHeaders(headers http.Header) Option {
	snapshot := headers.Clone()

	return func(p *Provider) {
		p.headers = apicall.MergeHeaders(p.headers, snapshot)
	}
}

// WithHTTPClient sets the HTTP client used for every request.
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// WithName replaces the provider name reported by the models. The default is
// "google.generative-ai".
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for sources and for
// tool calls that arrive without one.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// New returns a Google Generative AI provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL: defaultBaseURL,
		name:    defaultProviderName,
		generateID: func() string {
			return id.New("")
		},
	}

	for _, option := range options {
		option(provider)
	}

	if provider.baseURL == "" {
		provider.baseURL = defaultBaseURL
	}

	return provider
}

// Chat returns a Gemini language model that uses the generateContent API.
func (p *Provider) Chat(modelID string) *ChatModel {
	return &ChatModel{
		modelID:  modelID,
		provider: p,
	}
}

// Embedding returns an embedding model.
func (p *Provider) Embedding(modelID string) *EmbeddingModel {
	return &EmbeddingModel{
		modelID:  modelID,
		provider: p,
	}
}

// Image returns a Gemini image model.
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// Speech returns a Gemini text-to-speech model.
func (p *Provider) Speech(modelID string) *SpeechModel {
	return &SpeechModel{
		modelID:  modelID,
		provider: p,
	}
}

// Transcription returns a unary transcription model that uses the
// Interactions API.
func (p *Provider) Transcription(modelID string) *TranscriptionModel {
	return &TranscriptionModel{
		modelID:  modelID,
		provider: p,
	}
}

// Files returns the files interface, an *Files.
func (p *Provider) Files() model.Files {
	return &Files{
		provider: p,
	}
}

// LanguageModel returns Chat(modelID).
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID), nil
}

// EmbeddingModel returns Embedding(modelID).
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return p.Embedding(modelID), nil
}

// ImageModel returns Image(modelID).
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// TranscriptionModel returns Transcription(modelID).
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return p.Transcription(modelID), nil
}

// SpeechModel returns Speech(modelID).
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return p.Speech(modelID), nil
}

// RerankingModel reports that Google offers no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// request describes a call to path below the base URL, with the
// authentication headers, the provider headers and the call headers.
func (p *Provider) request(ctx context.Context, path string, callHeaders http.Header) (apicall.Request, error) {
	authentication, err := p.authenticationHeaders(ctx)
	if err != nil {
		return apicall.Request{}, err
	}

	baseURL, err := p.currentBaseURL()
	if err != nil {
		return apicall.Request{}, err
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        baseURL + "/" + path,
		Headers:    apicall.MergeHeaders(authentication, p.headers, callHeaders),
		ParseError: parseErrorMessage,
	}, nil
}

// authenticationHeaders returns the configured authentication headers, or
// the API key header.
func (p *Provider) authenticationHeaders(ctx context.Context) (http.Header, error) {
	if p.authHeaders != nil {
		return p.authHeaders(ctx)
	}

	apiKey, err := p.loadAPIKey()
	if err != nil {
		return nil, err
	}

	return http.Header{
		apiKeyHeader: {
			apiKey,
		},
	}, nil
}

// currentBaseURL returns the resolved base URL, or the configured one.
func (p *Provider) currentBaseURL() (string, error) {
	if p.resolveBaseURL == nil {
		return p.baseURL, nil
	}

	baseURL, err := p.resolveBaseURL()
	if err != nil {
		return "", err
	}

	return strings.TrimRight(baseURL, "/"), nil
}

func (p *Provider) loadAPIKey() (string, error) {
	return apikey.Load(p.apiKey, apiKeyEnvironment)
}
