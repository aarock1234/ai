// Package anthropic serves Claude models through Amazon Bedrock's native
// Anthropic Messages API, the InvokeModel endpoints, rather than the
// Converse API. Requests are signed with AWS SigV4, or authenticated with a
// Bedrock API key. Models are Anthropic Messages models, so the Anthropic
// provider options under "anthropic" (or "bedrock") and the provider tools
// of the anthropic package apply.
//
// It is a port of the Vercel AI SDK Amazon Bedrock Anthropic provider.
package anthropic

import (
	"context"
	"net/http"
	"regexp"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/awssigv4"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
	anthropicmessages "github.com/aarock1234/ai/pkg/provider/anthropic"
)

// providerName is the identifier models report. Its first segment,
// "bedrock", is also read as a provider options namespace.
const providerName = "bedrock.anthropic.messages"

var _ model.Provider = (*Provider)(nil)

// Provider creates Claude models served by Amazon Bedrock. Its settings are
// fixed at construction; the region, API key and credentials fall back to
// the environment when each request is sent.
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

// WithBaseURL replaces the regional Bedrock runtime endpoint, for example
// to use a proxy. Without it, the provider reads
// AWS_ENDPOINT_URL_BEDROCK_RUNTIME, then AWS_ENDPOINT_URL, and otherwise
// uses the regional endpoint.
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

// WithName sets the provider identifier reported by models. The part
// before the first dot names the provider options namespace read alongside
// "anthropic". The default is "bedrock.anthropic.messages".
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
		name: providerName,
	}

	for _, option := range options {
		option(provider)
	}

	return provider
}

// Messages returns a Claude model served by the Bedrock InvokeModel API.
func (p *Provider) Messages(modelID string) *anthropicmessages.MessagesModel {
	client := awssigv4.WrapClient(p.config.HTTPClient, func(base http.RoundTripper) http.RoundTripper {
		return &eventStreamTransport{
			base: p.config.Transport(base, p.runtimeBaseURL),
		}
	})

	options := []anthropicmessages.Option{
		anthropicmessages.WithName(p.name),
		anthropicmessages.WithHeaders(p.headers),
		anthropicmessages.WithHTTPClient(client),
		anthropicmessages.WithAuthHeaders(noAuthHeaders),
		anthropicmessages.WithRequestURL(func(stream bool) (string, error) {
			return p.requestURL(modelID, stream)
		}),
		anthropicmessages.WithRequestBodyTransform(transformRequestBody),
		// Bedrock does not download URLs, so files are sent inline.
		anthropicmessages.WithSupportedURLs(map[string][]*regexp.Regexp{}),
		anthropicmessages.WithNativeStructuredOutput(bedrock.SupportsNativeStructuredOutput(modelID)),
		anthropicmessages.WithStrictTools(bedrock.SupportsStrictTools(modelID)),
	}

	if p.generateID != nil {
		options = append(options, anthropicmessages.WithIDGenerator(p.generateID))
	}

	return anthropicmessages.New(options...).Messages(modelID)
}

// Chat returns a Claude model. It is equivalent to Messages.
func (p *Provider) Chat(modelID string) *anthropicmessages.MessagesModel {
	return p.Messages(modelID)
}

// LanguageModel returns the Claude model for modelID.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Messages(modelID), nil
}

// EmbeddingModel reports that the provider has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that the provider has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that the provider has no transcription
// models.
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

// runtimeBaseURL returns the Bedrock runtime base URL, whose origin receives
// credentials.
func (p *Provider) runtimeBaseURL() (string, error) {
	return p.config.ResolveBaseURL(bedrock.Runtime)
}

// noAuthHeaders returns no headers, because the client's transport
// authenticates requests.
func noAuthHeaders(context.Context) (http.Header, error) {
	return http.Header{}, nil
}

// requestURL returns the InvokeModel URL of a call, or the streaming
// variant.
func (p *Provider) requestURL(modelID string, stream bool) (string, error) {
	baseURL, err := p.runtimeBaseURL()
	if err != nil {
		return "", err
	}

	action := "invoke"
	if stream {
		action = "invoke-with-response-stream"
	}

	return baseURL + bedrock.ModelPath(modelID, action), nil
}
