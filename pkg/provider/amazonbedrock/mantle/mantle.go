// Package mantle serves models through Amazon Bedrock Mantle, Bedrock's
// OpenAI-compatible endpoint for models such as openai.gpt-oss-120b that are
// only available there. Requests are signed with AWS SigV4 for the
// "bedrock-mantle" service, or authenticated with a Bedrock API key. Models
// are OpenAI Chat Completions and Responses models, so the OpenAI provider
// options under "openai" apply.
//
// It is a port of the Vercel AI SDK Amazon Bedrock Mantle provider.
package mantle

import (
	"context"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/awssigv4"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

const (
	// providerName prefixes model provider identifiers, such as
	// "bedrock-mantle.chat".
	providerName = "bedrock-mantle"

	// signingService is the SigV4 service name of the Mantle endpoint.
	signingService = "bedrock-mantle"
)

var _ model.Provider = (*Provider)(nil)

// Provider creates Amazon Bedrock Mantle models. Its settings are fixed at
// construction; the region, API key and credentials fall back to the
// environment when each request is sent.
type Provider struct {
	config     bedrock.Config
	headers    http.Header
	name       string
	generateID func() string
}

// Option configures a Provider.
type Option func(*Provider)

// WithRegion sets the AWS region. Without it, the provider reads
// AWS_REGION.
func WithRegion(region string) Option {
	return func(p *Provider) {
		p.config.Region = region
	}
}

// WithAPIKey sets a Bedrock API key, sent as a bearer token instead of
// signing requests with SigV4. Without it, the provider reads
// AWS_BEARER_TOKEN_BEDROCK; a blank key selects SigV4.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.config.APIKey = key
	}
}

// WithAccessKeyID sets the AWS access key ID. Without it, the provider
// reads AWS_ACCESS_KEY_ID.
func WithAccessKeyID(accessKeyID string) Option {
	return func(p *Provider) {
		p.config.Credentials.AccessKeyID = accessKeyID
	}
}

// WithSecretAccessKey sets the AWS secret access key. Without it, the
// provider reads AWS_SECRET_ACCESS_KEY.
func WithSecretAccessKey(secretAccessKey string) Option {
	return func(p *Provider) {
		p.config.Credentials.SecretAccessKey = secretAccessKey
	}
}

// WithSessionToken sets the AWS session token of temporary credentials.
// Without it, the provider reads AWS_SESSION_TOKEN.
func WithSessionToken(sessionToken string) Option {
	return func(p *Provider) {
		p.config.Credentials.SessionToken = sessionToken
	}
}

// WithCredentialProvider sets a function that supplies credentials for each
// request, replacing the configured keys and the environment.
func WithCredentialProvider(provider amazonbedrock.CredentialProvider) Option {
	return func(p *Provider) {
		p.config.Credentials.Provider = bedrock.SigningCredentials(provider)
	}
}

// WithBaseURL replaces the regional Mantle endpoint, such as
// "https://bedrock-mantle.us-east-1.api.aws/v1". Request paths are appended
// to it for every model.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.config.BaseURL = baseURL
	}
}

// WithHeaders adds headers to every request. The headers are copied when
// the option is created.
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
		p.config.HTTPClient = client
	}
}

// WithName sets the prefix of model provider identifiers. The default is
// "bedrock-mantle".
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

// New returns a provider configured by options.
func New(options ...Option) *Provider {
	provider := &Provider{
		name: providerName,
	}

	for _, option := range options {
		option(provider)
	}

	provider.config.SigningService = signingService

	return provider
}

// Chat returns a Chat Completions model, which has the broadest model
// support on Mantle, such as "openai.gpt-oss-120b".
func (p *Provider) Chat(modelID string) *openai.ChatModel {
	return p.openaiProvider().Chat(modelID)
}

// Responses returns a Responses API model. Not every Mantle model supports
// the Responses API; the gpt-oss-safeguard models are Chat only.
func (p *Provider) Responses(modelID string) *openai.ResponsesModel {
	return p.openaiProvider().Responses(modelID)
}

// LanguageModel returns Chat(modelID).
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID), nil
}

// EmbeddingModel reports that Mantle has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that Mantle has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that Mantle has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Mantle has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Mantle has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.RerankingModelKind, modelID)
}

// openaiProvider returns the OpenAI provider that sends Mantle requests. Its
// client adds the bearer API key or signs each request with SigV4, so the
// authentication headers it builds are empty.
func (p *Provider) openaiProvider() *openai.Provider {
	client := awssigv4.WrapClient(p.config.HTTPClient, func(base http.RoundTripper) http.RoundTripper {
		return p.config.Transport(base, p.serviceBaseURL)
	})

	options := []openai.Option{
		openai.WithName(p.name),
		openai.WithHeaders(p.headers),
		openai.WithHTTPClient(client),
		openai.WithRequestURL(p.requestURL),
		openai.WithAuthHeaders(noAuthHeaders),
		openai.WithoutWebSearchSourcesInclude(),
	}

	if p.generateID != nil {
		options = append(options, openai.WithIDGenerator(p.generateID))
	}

	return openai.New(options...)
}

// noAuthHeaders returns no headers, because the client's transport
// authenticates requests.
func noAuthHeaders(context.Context) (http.Header, error) {
	return http.Header{}, nil
}
