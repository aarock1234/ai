// Package amazonbedrock provides Amazon Bedrock models: chat through the
// Converse API, Titan, Cohere and Nova embeddings, Nova Canvas images and
// reranking. Requests are signed with AWS SigV4, or authenticated with a
// Bedrock API key. It is a port of the Vercel AI SDK Amazon Bedrock
// provider.
//
// Claude models can also be served through Bedrock's native Anthropic
// Messages API with the amazonbedrock/anthropic package, and models served
// only by the OpenAI-compatible Bedrock Mantle endpoint with the
// amazonbedrock/mantle package.
package amazonbedrock

import (
	"context"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/awssigv4"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
)

const (
	// Namespace is the provider options and metadata namespace.
	Namespace = "amazonBedrock"

	// LegacyNamespace is read when the call has no options under Namespace,
	// and metadata is also stored under it.
	LegacyNamespace = "bedrock"

	// providerName is the provider identifier the models report.
	providerName = "amazon-bedrock"
)

var _ model.Provider = (*Provider)(nil)

// Credentials are AWS credentials that sign requests with SigV4: an access
// key ID, a secret access key and, for temporary credentials, a session
// token.
type Credentials = bedrock.Credentials

// CredentialProvider returns credentials for each request, such as
// temporary credentials that rotate.
type CredentialProvider = bedrock.CredentialProvider

// Provider creates Amazon Bedrock models. Its settings are fixed at
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
// When both keys are configured explicitly, only this token is used;
// otherwise AWS_SESSION_TOKEN is read when it is not set.
func WithSessionToken(sessionToken string) Option {
	return func(p *Provider) {
		p.config.Credentials.SessionToken = sessionToken
	}
}

// WithCredentialProvider sets a function that supplies credentials for each
// request, replacing the configured keys and the environment.
func WithCredentialProvider(provider CredentialProvider) Option {
	return func(p *Provider) {
		p.config.Credentials.Provider = bedrock.SigningCredentials(provider)
	}
}

// WithBaseURL replaces the regional endpoint, for example to use a proxy.
// Without it, the provider reads the service's AWS_ENDPOINT_URL_* variable,
// then AWS_ENDPOINT_URL, and otherwise uses the regional endpoint.
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

// WithName sets the provider identifier reported by models. The default is
// "amazon-bedrock". Provider options and metadata keep their namespaces.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for tool calls that
// arrive without one.
func WithIDGenerator(generateID func() string) Option {
	return func(p *Provider) {
		p.generateID = generateID
	}
}

// New returns a provider configured by options.
func New(options ...Option) *Provider {
	provider := &Provider{
		config: bedrock.Config{
			Credentials: awssigv4.CredentialSettings{
				ExplicitKeysOwnSessionToken: true,
			},
		},
		name: providerName,
		generateID: func() string {
			return id.New("")
		},
	}

	for _, option := range options {
		option(provider)
	}

	return provider
}

// Chat returns a Converse API language model.
func (p *Provider) Chat(modelID string, options ...ChatModelOption) *ChatModel {
	chat := &ChatModel{
		modelID:  modelID,
		provider: p,
	}

	for _, option := range options {
		option(chat)
	}

	return chat
}

// Embedding returns an embedding model.
func (p *Provider) Embedding(modelID string, options ...EmbeddingModelOption) *EmbeddingModel {
	embedding := &EmbeddingModel{
		modelID:  modelID,
		provider: p,
	}

	for _, option := range options {
		option(embedding)
	}

	return embedding
}

// Image returns an image model.
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// Reranking returns a reranking model.
func (p *Provider) Reranking(modelID string) *RerankingModel {
	return &RerankingModel{
		modelID:  modelID,
		provider: p,
	}
}

// LanguageModel returns the Converse model for modelID.
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

// TranscriptionModel reports that Amazon Bedrock has no transcription
// models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Amazon Bedrock has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.name, model.SpeechModelKind, modelID)
}

// RerankingModel returns the reranking model for modelID.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return p.Reranking(modelID), nil
}

// apiRequest prepares an authenticated call to path on endpoint. Call
// headers override provider headers.
func (p *Provider) apiRequest(ctx context.Context, endpoint bedrock.Endpoint, path string, callHeaders http.Header) (apicall.Request, error) {
	baseURL, err := p.config.ResolveBaseURL(endpoint)
	if err != nil {
		return apicall.Request{}, err
	}

	request := apicall.Request{
		URL:        baseURL + path,
		Headers:    apicall.MergeHeaders(p.headers, callHeaders),
		ParseError: parseError,
	}

	return p.config.Authorize(ctx, request)
}
