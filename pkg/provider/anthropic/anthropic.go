// Package anthropic provides Anthropic Claude language models through the
// Messages API, including extended thinking, citations, prompt caching and
// Anthropic's provider-defined tools.
package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	defaultBaseURL      = "https://api.anthropic.com/v1"
	unversionedBaseURL  = "https://api.anthropic.com"
	apiVersion          = "2023-06-01"
	defaultProviderName = "anthropic.messages"

	apiKeyVariable  = "ANTHROPIC_API_KEY"
	baseURLVariable = "ANTHROPIC_BASE_URL"

	// Namespace keys Anthropic provider options, file references and
	// metadata.
	Namespace = "anthropic"
)

var _ model.Provider = (*Provider)(nil)

// Provider creates Anthropic models. Its settings are fixed at construction;
// the API key and base URL fall back to the environment when each request is
// sent.
type Provider struct {
	apiKey     string
	authToken  string
	baseURL    string
	headers    http.Header
	httpClient *http.Client
	name       string
	generateID func() string

	// host adapts requests for a platform that hosts the Messages API.
	host hostSettings

	// batchMaxLineBytes bounds one line of a batch result file. Zero
	// selects the default.
	batchMaxLineBytes int
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key sent in the x-api-key header. Without it, the
// provider reads ANTHROPIC_API_KEY when a request is sent.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithAuthToken sets a token sent as an Authorization bearer header instead
// of an API key. Configuring both an API key and an auth token is an error.
func WithAuthToken(token string) Option {
	return func(p *Provider) {
		p.authToken = token
	}
}

// WithBaseURL sets the URL prefix for API calls, such as a proxy. Without it,
// the provider reads ANTHROPIC_BASE_URL and then uses
// https://api.anthropic.com/v1.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.baseURL = baseURL
	}
}

// WithHeaders adds headers to every request. The headers are copied when the
// option is created.
func WithHeaders(headers http.Header) Option {
	snapshot := headers.Clone()

	return func(p *Provider) {
		p.headers = apicall.MergeHeaders(p.headers, snapshot)
	}
}

// WithHTTPClient sets the client that sends requests. Nil uses
// http.DefaultClient.
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// WithName sets the provider identifier reported by models. The part before
// the first dot also names an additional provider options namespace that is
// read alongside "anthropic". The default is "anthropic.messages".
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for citation sources.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// New returns a provider configured by options.
func New(options ...Option) *Provider {
	provider := &Provider{
		name: defaultProviderName,
		generateID: func() string {
			return id.New("")
		},
	}

	for _, option := range options {
		option(provider)
	}

	return provider
}

// Messages returns a Messages API language model.
func (p *Provider) Messages(modelID string) *MessagesModel {
	return &MessagesModel{
		modelID:  modelID,
		provider: p,
	}
}

// Chat returns a Messages API language model. It is equivalent to Messages.
func (p *Provider) Chat(modelID string) *MessagesModel {
	return p.Messages(modelID)
}

// Files returns the files interface, an *Files.
func (p *Provider) Files() model.Files {
	return &Files{
		provider: p,
	}
}

// Skills returns the skills interface, an *Skills.
func (p *Provider) Skills() model.Skills {
	return &Skills{
		provider: p,
	}
}

// LanguageModel returns a Messages API language model.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Messages(modelID), nil
}

// EmbeddingModel reports that Anthropic has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that Anthropic has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that Anthropic has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Anthropic has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Anthropic has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// resolveBaseURL returns the configured, environment or default base URL
// without a trailing slash. The bare Anthropic host gains the /v1 prefix.
func (p *Provider) resolveBaseURL() (string, error) {
	if p.host.baseURL != nil {
		return p.host.baseURL()
	}

	baseURL := p.baseURL
	if baseURL == "" {
		baseURL = os.Getenv(baseURLVariable)
	}

	if baseURL == "" {
		return defaultBaseURL, nil
	}

	if strings.TrimSpace(baseURL) == "" {
		return "", fmt.Errorf("%w: base URL must not be blank", model.ErrInvalidArgument)
	}

	baseURL = strings.TrimSuffix(baseURL, "/")
	if baseURL == unversionedBaseURL {
		return defaultBaseURL, nil
	}

	return baseURL, nil
}

// configuredHeaders returns the version, authentication and custom headers
// shared by every request. A host's header function replaces the version
// and authentication headers.
func (p *Provider) configuredHeaders(ctx context.Context) (http.Header, error) {
	if p.host.headers != nil {
		hostHeaders, err := p.host.headers(ctx)
		if err != nil {
			return nil, err
		}

		return apicall.MergeHeaders(hostHeaders, p.headers), nil
	}

	if p.apiKey != "" && p.authToken != "" {
		return nil, fmt.Errorf(
			"%w: both an API key and an auth token were provided; use only one authentication method",
			model.ErrInvalidArgument,
		)
	}

	headers := make(http.Header)
	headers.Set("anthropic-version", apiVersion)

	if p.authToken != "" {
		headers.Set("Authorization", "Bearer "+p.authToken)

		return apicall.MergeHeaders(headers, p.headers), nil
	}

	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return nil, err
	}

	headers.Set("x-api-key", apiKey)

	return apicall.MergeHeaders(headers, p.headers), nil
}

// optionsNamespace returns the provider options namespace derived from the
// provider name, such as "anthropic" for "anthropic.messages".
func (p *Provider) optionsNamespace() string {
	namespace, _, _ := strings.Cut(p.name, ".")

	return namespace
}

// baseName returns the provider name without its ".messages" family, such
// as "anthropic" for "anthropic.messages". Batch and decision identifiers
// extend it.
func (p *Provider) baseName() string {
	return strings.TrimSuffix(p.name, ".messages")
}
