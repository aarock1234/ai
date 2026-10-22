// Package anthropic serves Claude models through Google Vertex AI's
// Anthropic partner model endpoints. Requests carry a Google OAuth access
// token from the configured token source. Models are Anthropic Messages
// models, so the Anthropic provider options under "anthropic" (or
// "googleVertex") and the provider tools of the anthropic package apply;
// Vertex AI recognizes the bash, text editor, computer 20241022, web search
// 20250305 and tool search tools.
//
// It is a port of the Vercel AI SDK Google Vertex Anthropic provider.
package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
	anthropicmessages "github.com/aarock1234/ai/pkg/provider/anthropic"
	"github.com/aarock1234/ai/pkg/provider/googlevertex"
	"github.com/aarock1234/ai/pkg/provider/googlevertex/internal/vertex"
)

const (
	// providerName is the identifier models report. Its first segment,
	// "googleVertex", is also read as a provider options namespace.
	providerName = "googleVertex.anthropic.messages"

	// Namespace is the namespace of the Vertex AI provider options.
	Namespace = "vertex"

	// vertexAnthropicVersion is the API version Vertex AI expects in the
	// body.
	vertexAnthropicVersion = "vertex-2023-10-16"

	// labelsHeader carries request labels for billing and reporting.
	labelsHeader = "X-Vertex-Ai-Labels"
)

var _ model.Provider = (*Provider)(nil)

// TokenSource returns a Google OAuth access token for a request.
type TokenSource = googlevertex.TokenSource

// ProviderOptions are the Vertex AI provider options of a call, read from
// the "vertex" namespace.
type ProviderOptions struct {
	// Labels are sent in the X-Vertex-AI-Labels header unless the call
	// sets that header itself.
	Labels map[string]string `json:"labels,omitempty"`
}

// Provider creates Claude models served by Vertex AI. Its settings are
// fixed at construction; the project and location fall back to the
// environment when each request is sent.
type Provider struct {
	project     string
	location    string
	baseURL     string
	headers     http.Header
	httpClient  *http.Client
	tokenSource TokenSource
	name        string
	generateID  func() string
}

// Option configures a Provider.
type Option func(*Provider)

// WithProject sets the Google Cloud project ID. Without it, the provider
// reads GOOGLE_VERTEX_PROJECT.
func WithProject(project string) Option {
	return func(p *Provider) {
		p.project = project
	}
}

// WithLocation sets the Google Cloud location, such as "us-east5",
// "global", "us" or "eu". Without it, the provider reads
// GOOGLE_VERTEX_LOCATION.
func WithLocation(location string) Option {
	return func(p *Provider) {
		p.location = location
	}
}

// WithBaseURL replaces the URL prefix of the Anthropic publisher models,
// for example to use a proxy. Without it, the URL is built from the
// location and project.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.baseURL = baseURL
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

// WithTokenSource sets the function that returns the OAuth access token
// sent as a bearer token with each request. It is required.
func WithTokenSource(source TokenSource) Option {
	return func(p *Provider) {
		p.tokenSource = source
	}
}

// WithName sets the provider identifier reported by models. The part
// before the first dot names the provider options namespace read alongside
// "anthropic". The default is "googleVertex.anthropic.messages".
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

	return provider
}

// Messages returns a Claude model served by Vertex AI.
func (p *Provider) Messages(modelID string) *anthropicmessages.MessagesModel {
	messages := anthropicmessages.New(
		anthropicmessages.WithName(p.name),
		anthropicmessages.WithHeaders(p.headers),
		anthropicmessages.WithHTTPClient(p.httpClient),
		anthropicmessages.WithIDGenerator(p.generateID),
		anthropicmessages.WithAuthHeaders(p.authorizationHeaders),
		anthropicmessages.WithRequestURL(func(stream bool) (string, error) {
			return p.requestURL(modelID, stream)
		}),
		anthropicmessages.WithRequestBodyTransform(transformRequestBody),
		anthropicmessages.WithRequestHeadersTransform(withLabels),
		// Vertex AI does not download URLs, so files are sent inline.
		anthropicmessages.WithSupportedURLs(map[string][]*regexp.Regexp{}),
		// The structured output beta header is not supported, so JSON
		// responses use the JSON tool.
		anthropicmessages.WithNativeStructuredOutput(false),
		anthropicmessages.WithStrictTools(false),
	)

	return messages.Messages(modelID)
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

// authorizationHeaders returns the bearer token header of a request.
func (p *Provider) authorizationHeaders(ctx context.Context) (http.Header, error) {
	if p.tokenSource == nil {
		return nil, fmt.Errorf("%w: configure a Google access token source with WithTokenSource", model.ErrLoadAPIKey)
	}

	token, err := p.tokenSource(ctx)
	if err != nil {
		return nil, fmt.Errorf("%w: generate Google access token: %w", model.ErrLoadAPIKey, err)
	}

	if token == "" {
		return nil, fmt.Errorf("%w: Google access token source returned an empty token", model.ErrLoadAPIKey)
	}

	return http.Header{
		"Authorization": {
			"Bearer " + token,
		},
	}, nil
}

// requestURL returns the rawPredict URL of a call, or streamRawPredict for
// streaming calls.
func (p *Provider) requestURL(modelID string, stream bool) (string, error) {
	baseURL, err := p.resolveBaseURL()
	if err != nil {
		return "", err
	}

	action := "rawPredict"
	if stream {
		action = "streamRawPredict"
	}

	return baseURL + "/" + modelID + ":" + action, nil
}

// resolveBaseURL returns the configured base URL without a trailing slash,
// or the Anthropic publisher models URL of the project and location.
func (p *Provider) resolveBaseURL() (string, error) {
	if p.baseURL != "" {
		return strings.TrimSuffix(p.baseURL, "/"), nil
	}

	location, err := vertex.LoadLocation(p.location, "")
	if err != nil {
		return "", err
	}

	project, err := vertex.LoadProject(p.project)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf(
		"https://%s/v1/projects/%s/locations/%s/publishers/anthropic/models",
		vertex.Host(location),
		project,
		location,
	), nil
}

// transformRequestBody removes the model, which is part of the URL, and
// adds the Vertex AI API version.
func transformRequestBody(body json.RawMessage, _ []string) (json.RawMessage, error) {
	object, err := jsonobject.Parse(body)
	if err != nil {
		return nil, fmt.Errorf("%w: request body: %w", model.ErrInvalidArgument, err)
	}

	transformed := object.
		Without("model").
		With("anthropic_version", jsonobject.EncodeString(vertexAnthropicVersion))

	return transformed.Encode()
}

// withLabels sends the call's labels in the labels header. A labels header
// set by the caller takes precedence.
func withLabels(headers http.Header, options model.ProviderOptions) (http.Header, error) {
	vertexOptions, _, err := model.ProviderValue[ProviderOptions](options, Namespace)
	if err != nil {
		return nil, err
	}

	if vertexOptions.Labels == nil || len(headers.Values(labelsHeader)) > 0 {
		return headers, nil
	}

	encoded, err := json.Marshal(vertexOptions.Labels)
	if err != nil {
		return nil, fmt.Errorf("%w: encode vertex labels: %w", model.ErrInvalidArgument, err)
	}

	labeled := headers.Clone()
	labeled.Set(labelsHeader, base64.StdEncoding.EncodeToString(encoded))

	return labeled, nil
}
