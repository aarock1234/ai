// Package anthropicaws provides Claude models through Claude Platform on
// AWS, the Anthropic Messages API hosted in AWS. Requests are signed with
// AWS SigV4, or authenticated with an AWS-provisioned API key. Models are
// Anthropic Messages models, so the Anthropic provider options under
// "anthropic" (or "anthropic-aws") and the provider tools of the anthropic
// package apply.
//
// It is a port of the Vercel AI SDK Anthropic AWS provider, including the
// Anthropic files and skills APIs served by the same endpoint.
package anthropicaws

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/awssigv4"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

const (
	// providerName is the identifier models report. Its first segment,
	// "anthropic-aws", is also read as a provider options namespace.
	providerName = "anthropic-aws.messages"

	apiVersion = "2023-06-01"

	apiKeyVariable      = "ANTHROPIC_AWS_API_KEY"
	workspaceIDVariable = "ANTHROPIC_AWS_WORKSPACE_ID"
)

var (
	_ model.Provider       = (*Provider)(nil)
	_ model.FilesProvider  = (*Provider)(nil)
	_ model.SkillsProvider = (*Provider)(nil)
)

// Credentials are AWS credentials that sign requests with SigV4.
type Credentials struct {
	AccessKeyID     string
	SecretAccessKey string

	// SessionToken is set for temporary credentials.
	SessionToken string
}

// CredentialProvider returns credentials for each request, such as
// temporary credentials that rotate.
type CredentialProvider func(ctx context.Context) (Credentials, error)

// Provider creates Claude models served by Claude Platform on AWS. Its
// settings are fixed at construction; the region, workspace, API key and
// credentials fall back to the environment when each request is sent.
type Provider struct {
	settings   settings
	headers    http.Header
	httpClient *http.Client
	name       string
	generateID func() string

	messages *anthropic.Provider
}

// settings are the endpoint and authentication settings read for each
// request.
type settings struct {
	region      string
	workspaceID string
	apiKey      string
	baseURL     string
	credentials awssigv4.CredentialSettings
}

// Option configures a Provider.
type Option func(*Provider)

// WithRegion sets the AWS region. Without it, the provider reads
// AWS_REGION; there is no default.
func WithRegion(region string) Option {
	return func(p *Provider) {
		p.settings.region = region
	}
}

// WithWorkspaceID sets the Anthropic workspace ID of the AWS account, sent
// in the anthropic-workspace-id header. Without it, the provider reads
// ANTHROPIC_AWS_WORKSPACE_ID.
func WithWorkspaceID(workspaceID string) Option {
	return func(p *Provider) {
		p.settings.workspaceID = workspaceID
	}
}

// WithAPIKey sets an API key, sent in the x-api-key header instead of
// signing requests with SigV4. Without it, the provider reads
// ANTHROPIC_AWS_API_KEY.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.settings.apiKey = key
	}
}

// WithAccessKeyID sets the AWS access key ID. Without it, the provider
// reads AWS_ACCESS_KEY_ID.
func WithAccessKeyID(accessKeyID string) Option {
	return func(p *Provider) {
		p.settings.credentials.AccessKeyID = accessKeyID
	}
}

// WithSecretAccessKey sets the AWS secret access key. Without it, the
// provider reads AWS_SECRET_ACCESS_KEY.
func WithSecretAccessKey(secretAccessKey string) Option {
	return func(p *Provider) {
		p.settings.credentials.SecretAccessKey = secretAccessKey
	}
}

// WithSessionToken sets the AWS session token of temporary credentials.
// Without it, the provider reads AWS_SESSION_TOKEN.
func WithSessionToken(sessionToken string) Option {
	return func(p *Provider) {
		p.settings.credentials.SessionToken = sessionToken
	}
}

// WithCredentialProvider sets a function that supplies credentials for each
// request, replacing the configured keys and the environment.
func WithCredentialProvider(provider CredentialProvider) Option {
	return func(p *Provider) {
		p.settings.credentials.Provider = signingCredentials(provider)
	}
}

// WithBaseURL replaces the regional endpoint
// https://aws-external-anthropic.{region}.api.aws/v1, for example to use a
// proxy.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.settings.baseURL = baseURL
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
		p.httpClient = client
	}
}

// WithName sets the provider identifier reported by models. The part
// before the first dot names the provider options namespace read alongside
// "anthropic". The default is "anthropic-aws.messages".
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
		generateID: func() string {
			return id.New("")
		},
	}

	for _, option := range options {
		option(provider)
	}

	provider.messages = provider.messagesProvider()

	return provider
}

// messagesProvider returns the Anthropic provider that serves the models,
// with this provider's endpoint, headers and authentication.
func (p *Provider) messagesProvider() *anthropic.Provider {
	current := p.settings

	client := awssigv4.WrapClient(p.httpClient, func(base http.RoundTripper) http.RoundTripper {
		return &authTransport{
			settings: current,
			base:     base,
		}
	})

	return anthropic.New(
		anthropic.WithName(p.name),
		anthropic.WithHeaders(p.headers),
		anthropic.WithHTTPClient(client),
		anthropic.WithIDGenerator(p.generateID),
		anthropic.WithAuthHeaders(func(context.Context) (http.Header, error) {
			return current.versionHeaders()
		}),
		anthropic.WithBaseURLResolver(current.resolveBaseURL),
		anthropic.WithRequestURL(func(bool) (string, error) {
			baseURL, err := current.resolveBaseURL()
			if err != nil {
				return "", err
			}

			return baseURL + "/messages", nil
		}),
	)
}

// Messages returns a Claude model.
func (p *Provider) Messages(modelID string) *anthropic.MessagesModel {
	return p.messages.Messages(modelID)
}

// Chat returns a Claude model. It is equivalent to Messages.
func (p *Provider) Chat(modelID string) *anthropic.MessagesModel {
	return p.Messages(modelID)
}

// Files returns the Anthropic Files API of the AWS endpoint, an
// *anthropic.Files.
func (p *Provider) Files() model.Files {
	return p.messages.Files()
}

// Skills returns the Anthropic Skills API of the AWS endpoint, an
// *anthropic.Skills.
func (p *Provider) Skills() model.Skills {
	return p.messages.Skills()
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

// versionHeaders returns the API version and workspace headers.
func (s settings) versionHeaders() (http.Header, error) {
	workspaceID := s.workspaceID
	if workspaceID == "" {
		workspaceID = os.Getenv(workspaceIDVariable)
	}

	if workspaceID == "" {
		return nil, fmt.Errorf(
			"%w: Anthropic AWS workspace ID setting is missing; configure it with WithWorkspaceID or set the %s environment variable",
			model.ErrInvalidArgument,
			workspaceIDVariable,
		)
	}

	return http.Header{
		"Anthropic-Version": {
			apiVersion,
		},
		"Anthropic-Workspace-Id": {
			workspaceID,
		},
	}, nil
}

// resolveBaseURL returns the configured base URL without a trailing slash,
// or the regional endpoint.
func (s settings) resolveBaseURL() (string, error) {
	if s.baseURL != "" {
		return strings.TrimSuffix(s.baseURL, "/"), nil
	}

	region, err := awssigv4.LoadHostRegion(s.region)
	if err != nil {
		return "", err
	}

	return "https://aws-external-anthropic." + region + ".api.aws/v1", nil
}

// signingCredentials adapts a public credential provider to the signer.
func signingCredentials(provider CredentialProvider) awssigv4.CredentialProvider {
	if provider == nil {
		return nil
	}

	return func(ctx context.Context) (awssigv4.Credentials, error) {
		credentials, err := provider(ctx)
		if err != nil {
			return awssigv4.Credentials{}, err
		}

		return awssigv4.Credentials{
			AccessKeyID:     credentials.AccessKeyID,
			SecretAccessKey: credentials.SecretAccessKey,
			SessionToken:    credentials.SessionToken,
		}, nil
	}
}
