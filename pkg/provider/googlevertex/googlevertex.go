// Package googlevertex provides Google Vertex AI models: Gemini language
// models with grounding and provider-executed tools, Interactions API
// language models and agents, text embedding models, Gemini image models,
// Gemini and Chirp 3: HD text-to-speech, and Gemini and Cloud
// Speech-to-Text transcription. It is a port of the Vercel AI SDK
// Google Vertex provider and reuses the google package for the Gemini wire
// format.
//
// Requests authenticate with an OAuth access token from WithTokenSource, or
// with an API key in Vertex AI express mode.
package googlevertex

import (
	"context"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/google"
)

const (
	// defaultName prefixes model provider identifiers, such as
	// "google.vertex.chat".
	defaultName = "google.vertex"

	// Namespace is the provider options and metadata namespace. Options are
	// also read from LegacyNamespace and, as a fallback, from "google".
	Namespace = "googleVertex"

	// LegacyNamespace is the earlier provider options and metadata
	// namespace. Metadata is written under both namespaces.
	LegacyNamespace = "vertex"

	apiKeyVariable = "GOOGLE_VERTEX_API_KEY"
)

// optionNamespaces are read in order for provider options.
var optionNamespaces = []string{
	Namespace,
	LegacyNamespace,
	google.Namespace,
}

var _ model.Provider = (*Provider)(nil)

// TokenSource returns an OAuth 2.0 access token with the
// https://www.googleapis.com/auth/cloud-platform scope, such as a token from
// golang.org/x/oauth2/google. It is called for every request. The maas, xai
// and anthropic subpackages share it.
type TokenSource func(ctx context.Context) (string, error)

// Provider creates Google Vertex AI models. Use New to create it. It is
// immutable and safe for concurrent use.
type Provider struct {
	settings settings
}

// settings holds the provider configuration.
type settings struct {
	name        string
	apiKey      string
	tokenSource TokenSource
	project     string
	location    string
	baseURL     string
	headers     http.Header
	httpClient  *http.Client
	generateID  func() string

	toolResultDownloadMaxBytes int64
}

// Option configures a Provider.
type Option func(*settings)

// WithAPIKey sets the API key of Vertex AI express mode. Without it, the key
// is read from GOOGLE_VERTEX_API_KEY when a request is made. When a key is
// set, requests use express mode instead of the token source.
func WithAPIKey(apiKey string) Option {
	return func(s *settings) {
		s.apiKey = apiKey
	}
}

// WithTokenSource authenticates requests with OAuth access tokens. It is
// required unless an express mode API key is set.
func WithTokenSource(tokenSource TokenSource) Option {
	return func(s *settings) {
		s.tokenSource = tokenSource
	}
}

// WithProject sets the Google Cloud project. Without it, the project is read
// from GOOGLE_VERTEX_PROJECT when a request is made.
func WithProject(project string) Option {
	return func(s *settings) {
		s.project = project
	}
}

// WithLocation sets the Google Cloud location, such as "us-central1",
// "global", "us" or "eu". Without it, the location is read from
// GOOGLE_VERTEX_LOCATION when a request is made.
func WithLocation(location string) Option {
	return func(s *settings) {
		s.location = location
	}
}

// WithBaseURL replaces the URL prefix of Vertex AI requests, for example to
// use a proxy. The default is derived from the project and location, or is
// https://aiplatform.googleapis.com/v1/publishers/google in express mode.
func WithBaseURL(baseURL string) Option {
	return func(s *settings) {
		s.baseURL = strings.TrimRight(baseURL, "/")
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

// WithName sets the prefix of model provider identifiers, such as
// "google.vertex.chat". The default is "google.vertex".
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

// WithToolResultDownloadMaxBytes bounds each tool result file URL that
// language models download before sending it inline. The default is 7 MiB.
func WithToolResultDownloadMaxBytes(maxBytes int64) Option {
	return func(s *settings) {
		s.toolResultDownloadMaxBytes = maxBytes
	}
}

// New returns a Google Vertex AI provider.
func New(options ...Option) *Provider {
	s := settings{
		name: defaultName,
	}

	for _, option := range options {
		option(&s)
	}

	return &Provider{
		settings: s,
	}
}

// Chat returns a Gemini language model. Tuned models are addressed by their
// endpoint resource, such as "endpoints/1234567890", and require a token
// source.
func (p *Provider) Chat(modelID string) *google.ChatModel {
	return p.googleProvider("chat", isEndpointModel(modelID)).Chat(modelID)
}

// Embedding returns a text embedding model.
func (p *Provider) Embedding(modelID string) *EmbeddingModel {
	return &EmbeddingModel{
		modelID:  modelID,
		settings: &p.settings,
	}
}

// Image returns a Gemini image model.
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p.settings.name + ".image",
		language: p.googleProvider("image", false).Chat(modelID),
	}
}

// Speech returns a text-to-speech model. Chirp models, such as "chirp-3-hd",
// use Cloud Text-to-Speech and require a token source; other models use
// Gemini text-to-speech.
func (p *Provider) Speech(modelID string) *SpeechModel {
	speech := &SpeechModel{
		modelID:  modelID,
		provider: p.settings.name + ".speech",
	}

	if strings.HasPrefix(modelID, "chirp") {
		speech.cloud = &cloudSpeechModel{
			modelID:  modelID,
			settings: &p.settings,
		}
	} else {
		speech.gemini = p.googleProvider("speech", false).Speech(modelID)
	}

	return speech
}

// Transcription returns a transcription model. Gemini models, such as
// "gemini-3.5-transcribe", use the generateContent API; other models, such
// as "chirp_3", use Cloud Speech-to-Text. Both require a token source.
func (p *Provider) Transcription(modelID string) *TranscriptionModel {
	return &TranscriptionModel{
		modelID:  modelID,
		settings: &p.settings,
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

// RerankingModel reports that Google Vertex AI offers no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.settings.name, model.RerankingModelKind, modelID)
}

// googleProvider returns a Google provider that speaks the Vertex AI dialect
// with this provider's URLs and authentication. Endpoint providers address
// tuned models, whose resource replaces the publisher model path.
func (p *Provider) googleProvider(family string, endpoint bool) *google.Provider {
	s := &p.settings

	options := []google.Option{
		google.WithName(s.name + "." + family),
		google.WithBaseURLResolver(func() (string, error) {
			return s.resolveBaseURL(endpoint)
		}),
		google.WithVertexAI(google.VertexAISettings{
			ToolResultDownloadMaxBytes: s.toolResultDownloadMaxBytes,
		}),
	}

	return google.New(append(options, p.connectionOptions()...)...)
}

// connectionOptions are the Google provider options that every Vertex AI
// model shares: authentication, the HTTP client, headers and IDs.
func (p *Provider) connectionOptions() []google.Option {
	s := &p.settings

	options := []google.Option{
		google.WithAuthHeaders(s.authHeaders),
		google.WithHTTPClient(s.httpClient),
	}

	if s.headers != nil {
		options = append(options, google.WithHeaders(s.headers))
	}

	if s.generateID != nil {
		options = append(options, google.WithIDGenerator(s.generateID))
	}

	return options
}
