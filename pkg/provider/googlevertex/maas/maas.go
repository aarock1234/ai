// Package maas provides partner and open models served by Google Vertex AI
// Model as a Service (MaaS), such as DeepSeek, Llama, Qwen and gpt-oss,
// through Vertex AI's OpenAI-compatible Chat Completions API. It is a port of
// the Vercel AI SDK Google Vertex MaaS provider and builds on the
// openaicompatible package.
package maas

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/googlevertex"
	"github.com/aarock1234/ai/pkg/provider/googlevertex/internal/vertex"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

const (
	// defaultName prefixes model provider identifiers, such as
	// "vertex.maas.chat". Its first segment, "vertex", is the provider
	// options namespace.
	defaultName = "vertex.maas"

	defaultLocation = "global"
)

// maxOutputTokensByModel are the output limits sent to models that require
// max_tokens.
var maxOutputTokensByModel = map[string]int{
	"meta/llama-4-maverick-17b-128e-instruct-maas": 8192,
	"meta/llama-4-scout-17b-16e-instruct-maas":     8192,
}

var _ model.Provider = (*Provider)(nil)

// TokenSource returns an OAuth 2.0 access token with the
// https://www.googleapis.com/auth/cloud-platform scope. It is called for
// every request.
type TokenSource = googlevertex.TokenSource

// Provider creates Vertex AI MaaS models. Use New to create it. It is
// immutable and safe for concurrent use.
type Provider struct {
	settings settings
}

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
// "vertex.maas.chat". The default is "vertex.maas".
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

// New returns a Vertex AI MaaS provider.
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
// "deepseek-ai/deepseek-v3.2-maas".
func (p *Provider) Chat(modelID string) *openaicompatible.ChatModel {
	return openaicompatible.NewChatModel(modelID, openaicompatible.ChatConfig{
		Provider:             p.settings.name + ".chat",
		BuildURL:             p.settings.url,
		Headers:              p.settings.requestHeaders,
		HTTPClient:           p.settings.client(),
		TransformRequestBody: transformRequestBody,
		GenerateID:           p.settings.generateID,
	})
}

// Completion returns a legacy completions language model.
func (p *Provider) Completion(modelID string) *openaicompatible.CompletionModel {
	return openaicompatible.NewCompletionModel(modelID, openaicompatible.CompletionConfig{
		Provider:   p.settings.name + ".completion",
		BuildURL:   p.settings.url,
		Headers:    p.settings.requestHeaders,
		HTTPClient: p.settings.client(),
	})
}

// Embedding returns an embedding model.
func (p *Provider) Embedding(modelID string) *openaicompatible.EmbeddingModel {
	return openaicompatible.NewEmbeddingModel(modelID, openaicompatible.EmbeddingConfig{
		Provider:   p.settings.name + ".embedding",
		BuildURL:   p.settings.url,
		Headers:    p.settings.requestHeaders,
		HTTPClient: p.settings.client(),
	})
}

// Image returns an image model.
func (p *Provider) Image(modelID string) *openaicompatible.ImageModel {
	return openaicompatible.NewImageModel(modelID, openaicompatible.ImageConfig{
		Provider:   p.settings.name + ".image",
		BuildURL:   p.settings.url,
		Headers:    p.settings.requestHeaders,
		HTTPClient: p.settings.client(),
	})
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

// TranscriptionModel reports that Vertex AI MaaS has no transcription
// models.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return nil, model.NoSuchModelError(p.settings.name, model.TranscriptionModelKind, modelID)
}

// SpeechModel reports that Vertex AI MaaS has no speech models.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return nil, model.NoSuchModelError(p.settings.name, model.SpeechModelKind, modelID)
}

// RerankingModel reports that Vertex AI MaaS has no reranking models.
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
// the project and location.
func (s settings) resolveBaseURL() (string, error) {
	if s.baseURL != "" {
		return s.baseURL, nil
	}

	location, err := vertex.LoadLocation(s.location, defaultLocation)
	if err != nil {
		return "", err
	}

	project, err := vertex.LoadProject(s.project)
	if err != nil {
		return "", err
	}

	return "https://" + vertex.Host(location) + "/v1/projects/" + project + "/locations/" + location + "/endpoints/openapi", nil
}

// transformRequestBody sends the output limit of models that require
// max_tokens when the call sets none.
func transformRequestBody(body map[string]json.RawMessage) (map[string]json.RawMessage, []model.Warning, error) {
	if _, ok := body["max_tokens"]; ok {
		return body, nil, nil
	}

	var modelID string
	if err := json.Unmarshal(body["model"], &modelID); err != nil {
		return nil, nil, fmt.Errorf("decode request model: %w", err)
	}

	maxOutputTokens, ok := maxOutputTokensByModel[modelID]
	if !ok {
		return body, nil, nil
	}

	encoded, err := json.Marshal(maxOutputTokens)
	if err != nil {
		return nil, nil, fmt.Errorf("encode max_tokens: %w", err)
	}

	body["max_tokens"] = encoded

	return body, nil, nil
}
