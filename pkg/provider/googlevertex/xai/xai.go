// Package xai serves xAI Grok partner models, such as
// "xai/grok-4.20-reasoning", through Google Vertex AI's OpenAI-compatible
// Chat Completions API. It is a port of the Vercel AI SDK Google Vertex xAI
// provider and, like it, builds on the openaicompatible package.
package xai

import (
	"net/http"
	"os"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/urlpath"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/googlevertex/internal/vertex"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

const (
	// defaultName prefixes model provider identifiers, such as
	// "googleVertex.xai.chat". Its first segment, "googleVertex", is the
	// provider options namespace.
	defaultName = "googleVertex.xai"

	defaultLocation = "global"
)

// supportedURLs are the image URLs that Vertex AI downloads itself.
var supportedURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
}

var _ model.Provider = (*Provider)(nil)

// Provider creates Grok models served by Vertex AI. Use New to create it. It
// is immutable and safe for concurrent use.
type Provider struct {
	settings settings
}

// settings holds the provider configuration.
type settings struct {
	name        string
	project     string
	location    string
	baseURL     string
	tokenSource TokenSource
	headers     http.Header
	httpClient  *http.Client
	generateID  func() string
}

// Option configures a Provider.
type Option func(*settings)

// WithProject sets the Google Cloud project. Without it, the project is read
// from GOOGLE_VERTEX_PROJECT when a request is made.
func WithProject(project string) Option {
	return func(s *settings) {
		s.project = project
	}
}

// WithLocation sets the Google Cloud location. Without it, the location is
// read from GOOGLE_VERTEX_LOCATION when a request is made, and defaults to
// "global".
func WithLocation(location string) Option {
	return func(s *settings) {
		s.location = location
	}
}

// WithBaseURL replaces the URL prefix of requests. The default is derived
// from the project and location.
func WithBaseURL(baseURL string) Option {
	return func(s *settings) {
		s.baseURL = strings.TrimRight(baseURL, "/")
	}
}

// WithTokenSource authenticates requests with OAuth access tokens. Without
// it, requests carry only the configured headers.
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

// WithName sets the prefix of model provider identifiers, such as
// "googleVertex.xai.chat". The default is "googleVertex.xai".
func WithName(name string) Option {
	return func(s *settings) {
		s.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for tool calls that
// arrive without one.
func WithIDGenerator(generateID func() string) Option {
	return func(s *settings) {
		s.generateID = generateID
	}
}

// New returns a Vertex AI xAI provider.
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

// Chat returns a chat completions language model, such as
// "xai/grok-4.1-fast-reasoning".
func (p *Provider) Chat(modelID string) *openaicompatible.ChatModel {
	return openaicompatible.NewChatModel(modelID, openaicompatible.ChatConfig{
		Provider:                  p.settings.name + ".chat",
		BuildURL:                  p.settings.url,
		Headers:                   p.settings.requestHeaders,
		HTTPClient:                p.settings.client(),
		IncludeUsage:              true,
		SupportsStructuredOutputs: true,
		SupportedURLs:             supportedURLs,
		TransformRequestBody:      transformRequestBody,
		ConvertUsage:              convertUsage,
		GenerateID:                p.settings.generateID,
	})
}

// LanguageModel returns Chat(modelID).
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Chat(modelID), nil
}

// EmbeddingModel reports that Vertex AI xAI has no embedding models.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return nil, model.NoSuchModelError(p.settings.name, model.EmbeddingModelKind, modelID)
}

// ImageModel reports that Vertex AI xAI has no image models.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return nil, model.NoSuchModelError(p.settings.name, model.ImageModelKind, modelID)
}

// TranscriptionModel reports that Vertex AI xAI has no transcription models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.settings.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Vertex AI xAI has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.settings.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Vertex AI xAI has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.settings.name, model.RerankingModelKind, modelID)
}

// url returns the URL of an API path, or the error of settings that prevent
// building it.
func (s settings) url(path string) (string, error) {
	baseURL, err := s.resolveBaseURL()
	if err != nil {
		return "", err
	}

	return baseURL + path, nil
}

// requestHeaders returns the configured headers.
func (s settings) requestHeaders() (http.Header, error) {
	return s.headers.Clone(), nil
}

// resolveBaseURL returns the base URL, or the OpenAI-compatible endpoint of
// the project and location on the global Vertex AI host.
func (s settings) resolveBaseURL() (string, error) {
	if s.baseURL != "" {
		return s.baseURL, nil
	}

	project, err := vertex.LoadProject(s.project)
	if err != nil {
		return "", err
	}

	location := urlpath.Escape(s.loadLocation())

	return "https://aiplatform.googleapis.com/v1/projects/" + project + "/locations/" + location + "/endpoints/openapi", nil
}

// loadLocation returns the configured location, GOOGLE_VERTEX_LOCATION or
// "global".
func (s settings) loadLocation() string {
	if s.location != "" {
		return s.location
	}

	if location := os.Getenv(vertex.LocationVariable); location != "" {
		return location
	}

	return defaultLocation
}
