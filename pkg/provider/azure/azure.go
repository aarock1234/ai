// Package azure provides Azure OpenAI models: the Responses API (the default
// language model), Chat Completions, legacy Completions, Azure-hosted
// DeepSeek chat, embeddings, images, speech and transcription. MAI-Image
// models use the MAI image API, and MAI-Voice and MAI-Transcribe models use
// Azure Speech. It is a port of the Vercel AI SDK Azure provider and reuses
// the openai and deepseek packages for their wire formats.
package azure

import (
	"context"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

const (
	// defaultName prefixes model provider identifiers, such as
	// "azure.responses".
	defaultName = "azure"

	// Namespace is the provider options and metadata namespace.
	Namespace = "azure"

	apiKeyVariable       = "AZURE_API_KEY"
	resourceNameVariable = "AZURE_RESOURCE_NAME"

	defaultAPIVersion = "v1"
)

var _ model.Provider = (*Provider)(nil)

// TokenSource returns a Microsoft Entra ID access token. It is called for
// every request.
type TokenSource func(ctx context.Context) (string, error)

// Provider creates Azure OpenAI models. Use New to create it. It is immutable
// and safe for concurrent use.
type Provider struct {
	settings settings
	openai   *openai.Provider
}

// settings holds the provider configuration.
type settings struct {
	name string

	resourceName string
	baseURL      string
	apiKey       string
	tokenSource  TokenSource
	headers      http.Header
	httpClient   *http.Client
	generateID   func() string

	apiVersion             string
	useDeploymentBasedURLs bool

	speechBaseURL string
	maiBaseURL    string
}

// Option configures a Provider.
type Option func(*settings)

// WithResourceName sets the Azure OpenAI resource name, a single DNS label
// used in https://{resourceName}.openai.azure.com/openai/v1. Without it, the
// name is read from AZURE_RESOURCE_NAME when a request is made. A base URL
// replaces the resource name for OpenAI requests.
func WithResourceName(resourceName string) Option {
	return func(s *settings) {
		s.resourceName = resourceName
	}
}

// WithBaseURL replaces the URL prefix of OpenAI requests, for example to use
// a proxy. Azure OpenAI base URLs receive the /v1 suffix unless they already
// end in /openai/v1; other gateways receive the request path directly.
func WithBaseURL(baseURL string) Option {
	return func(s *settings) {
		s.baseURL = strings.TrimRight(baseURL, "/")
	}
}

// WithAPIKey sets the API key. Without it, the key is read from AZURE_API_KEY
// when a request is made. It cannot be combined with WithTokenSource.
func WithAPIKey(apiKey string) Option {
	return func(s *settings) {
		s.apiKey = apiKey
	}
}

// WithTokenSource authenticates requests with Microsoft Entra ID access
// tokens instead of an API key. Requests that already carry an Authorization
// header keep it.
func WithTokenSource(tokenSource TokenSource) Option {
	return func(s *settings) {
		s.tokenSource = tokenSource
	}
}

// WithHeaders adds headers to every request. The headers are copied.
func WithHeaders(headers http.Header) Option {
	snapshot := headers.Clone()

	return func(s *settings) {
		s.headers = snapshot.Clone()
	}
}

// WithHTTPClient sets the client used for every request. Nil uses
// http.DefaultClient.
func WithHTTPClient(client *http.Client) Option {
	return func(s *settings) {
		s.httpClient = client
	}
}

// WithName sets the provider name that prefixes model provider
// identifiers, such as "azure.responses". The default is "azure".
func WithName(name string) Option {
	return func(s *settings) {
		s.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for sources and for
// tool calls that arrive without one.
func WithIDGenerator(generateID func() string) Option {
	return func(s *settings) {
		s.generateID = generateID
	}
}

// WithAPIVersion sets the api-version query parameter. The default is "v1".
// Complete v1 base URLs are used as they are.
func WithAPIVersion(apiVersion string) Option {
	return func(s *settings) {
		s.apiVersion = apiVersion
	}
}

// WithDeploymentBasedURLs uses the legacy URL format
// {baseURL}/deployments/{deploymentID}{path}?api-version={apiVersion}
// instead of {baseURL}/v1{path}?api-version={apiVersion}.
func WithDeploymentBasedURLs() Option {
	return func(s *settings) {
		s.useDeploymentBasedURLs = true
	}
}

// WithSpeechBaseURL sets the URL prefix of Azure Speech requests, which
// serve MAI-Transcribe transcription and MAI-Voice speech, such as the
// regional endpoint https://eastus.api.cognitive.microsoft.com. The default
// is https://{resourceName}.cognitiveservices.azure.com.
func WithSpeechBaseURL(speechBaseURL string) Option {
	return func(s *settings) {
		s.speechBaseURL = strings.TrimRight(speechBaseURL, "/")
	}
}

// WithMAIBaseURL sets the URL prefix of MAI API requests, which serve
// MAI-Image models. The default is
// https://{resourceName}.services.ai.azure.com/mai/v1.
func WithMAIBaseURL(maiBaseURL string) Option {
	return func(s *settings) {
		s.maiBaseURL = strings.TrimRight(maiBaseURL, "/")
	}
}

// New returns an Azure OpenAI provider.
func New(options ...Option) *Provider {
	s := settings{
		name:       defaultName,
		apiVersion: defaultAPIVersion,
	}

	for _, option := range options {
		option(&s)
	}

	return &Provider{
		settings: s,
		openai:   newOpenAIProvider(s),
	}
}

// newOpenAIProvider returns the OpenAI provider that serves the OpenAI wire
// formats with Azure URLs and authentication.
func newOpenAIProvider(s settings) *openai.Provider {
	options := []openai.Option{
		openai.WithName(s.name),
		openai.WithOptionsNamespace(Namespace),
		openai.WithRequestURL(s.openaiURL),
		openai.WithAuthHeaders(func(context.Context) (http.Header, error) {
			return s.authHeaders(apiOpenAI)
		}),
		openai.WithHTTPClient(s.client()),
	}

	if s.headers != nil {
		options = append(options, openai.WithHeaders(s.headers))
	}

	if s.generateID != nil {
		options = append(options, openai.WithIDGenerator(s.generateID))
	}

	if s.baseURLInfo().isFoundryProject {
		options = append(options, openai.WithExplicitMessageItems())
	}

	return openai.New(options...)
}

// ModelOption configures one language model.
type ModelOption func(*modelSettings)

type modelSettings struct {
	modelID string
}

// WithModelID sets the underlying model ID, such as "gpt-5", used for the
// reported model identity and capability detection. The default is the
// deployment ID. Requests always use the deployment ID.
func WithModelID(modelID string) ModelOption {
	return func(s *modelSettings) {
		s.modelID = modelID
	}
}

// Responses returns a Responses API language model for a deployment.
func (p *Provider) Responses(deploymentID string, options ...ModelOption) *openai.ResponsesModel {
	return p.openai.Responses(
		underlyingModelID(deploymentID, options),
		openai.WithRequestModelID(deploymentID),
	)
}

// Chat returns a Chat Completions language model for a deployment.
func (p *Provider) Chat(deploymentID string, options ...ModelOption) *openai.ChatModel {
	return p.openai.Chat(
		underlyingModelID(deploymentID, options),
		openai.WithRequestModelID(deploymentID),
	)
}

// Completion returns a legacy Completions language model for a deployment.
func (p *Provider) Completion(deploymentID string, options ...ModelOption) *openai.CompletionModel {
	return p.openai.Completion(
		underlyingModelID(deploymentID, options),
		openai.WithRequestModelID(deploymentID),
	)
}

// underlyingModelID returns the model ID that a language model reports,
// which defaults to the deployment ID.
func underlyingModelID(deploymentID string, options []ModelOption) string {
	s := modelSettings{
		modelID: deploymentID,
	}

	for _, option := range options {
		option(&s)
	}

	return s.modelID
}

// Embedding returns an embedding model for a deployment.
func (p *Provider) Embedding(deploymentID string) *EmbeddingModel {
	return &EmbeddingModel{
		EmbeddingModel: p.openai.Embedding(deploymentID),
		provider:       p.settings.name + ".embeddings",
	}
}

// Image returns an image model for a deployment. MAI-Image models use the
// MAI image API by default and other deployments use the OpenAI image API;
// ImageOptions.API overrides the choice per call.
func (p *Provider) Image(deploymentID string) *ImageModel {
	return &ImageModel{
		modelID:  deploymentID,
		provider: p.settings.name + ".image",
		openai:   p.openai.Image(deploymentID),
		mai: &maiImageModel{
			modelID:  deploymentID,
			settings: &p.settings,
		},
	}
}

// Speech returns a speech model for a deployment. MAI-Voice models use Azure
// Speech by default and other deployments use the OpenAI speech API;
// SpeechOptions.API overrides the choice per call.
func (p *Provider) Speech(deploymentID string) *SpeechModel {
	return &SpeechModel{
		modelID:  deploymentID,
		provider: p.settings.name + ".speech",
		openai:   p.openai.Speech(deploymentID),
		speech: &speechSynthesisModel{
			modelID:  deploymentID,
			settings: &p.settings,
		},
	}
}

// Transcription returns a transcription model for a deployment.
// MAI-Transcribe models use Azure Speech by default, MAI-Transcribe-2-Streaming
// uses the MAI realtime API, and other deployments use the OpenAI
// transcription API; TranscriptionOptions.API overrides the choice per call.
func (p *Provider) Transcription(deploymentID string) *TranscriptionModel {
	return &TranscriptionModel{
		modelID:  deploymentID,
		provider: p.settings.name + ".transcription",
		openai:   p.openai.Transcription(deploymentID),
		speech: &speechTranscriptionModel{
			modelID:  deploymentID,
			settings: &p.settings,
		},
	}
}

// LanguageModel returns the Responses API model for a deployment.
func (p *Provider) LanguageModel(deploymentID string) (model.LanguageModel, error) {
	return p.Responses(deploymentID), nil
}

// EmbeddingModel returns the embedding model for a deployment.
func (p *Provider) EmbeddingModel(deploymentID string) (model.EmbeddingModel, error) {
	return p.Embedding(deploymentID), nil
}

// ImageModel returns the image model for a deployment.
func (p *Provider) ImageModel(deploymentID string) (model.ImageModel, error) {
	return p.Image(deploymentID), nil
}

// TranscriptionModel returns the transcription model for a deployment.
func (p *Provider) TranscriptionModel(deploymentID string) (model.TranscriptionModel, error) {
	return p.Transcription(deploymentID), nil
}

// SpeechModel returns the speech model for a deployment.
func (p *Provider) SpeechModel(deploymentID string) (model.SpeechModel, error) {
	return p.Speech(deploymentID), nil
}

// RerankingModel reports that Azure OpenAI has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.settings.name, model.RerankingModelKind, modelID)
}
