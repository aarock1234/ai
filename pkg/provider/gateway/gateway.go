// Package gateway implements the Vercel AI Gateway provider. The gateway
// accepts the normalized model calls of every model kind and routes them to
// the underlying providers, so its models forward the call options as JSON
// and decode normalized results. The package also reads the gateway's model
// catalog, credits, spend reports and generation details. It is a port of
// the Vercel AI SDK gateway provider.
package gateway

import (
	"net/http"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultBaseURL is the AI Gateway model API.
	defaultBaseURL = "https://ai-gateway.vercel.sh/v4/ai"

	// defaultName is the provider name reported by the models.
	defaultName = "gateway"

	// defaultMetadataCacheRefresh is how long the model catalog is cached.
	defaultMetadataCacheRefresh = 5 * time.Minute

	// Namespace is the provider options and metadata namespace.
	Namespace = "gateway"
)

// Provider creates AI Gateway models and reads gateway account data. It is
// safe for concurrent use. Use New to create it.
type Provider struct {
	apiKey               string
	baseURL              string
	teamIDOrSlug         string
	headers              http.Header
	httpClient           *http.Client
	name                 string
	metadataCacheRefresh time.Duration

	// batchMaxLineBytes bounds one line of a batch result stream. Zero
	// selects the default.
	batchMaxLineBytes int

	catalog *catalogCache
}

// catalogCache holds the last model catalog. The lock channel, of capacity
// one, serializes refreshes while letting waiting callers give up when
// their context ends.
type catalogCache struct {
	lock      chan struct{}
	models    *AvailableModels
	fetchedAt time.Time
}

// Option configures a Provider.
type Option func(*Provider)

// WithAPIKey sets the API key or Vercel access token. Without it, the key is
// read from AI_GATEWAY_API_KEY, falling back to the Vercel OIDC token in
// VERCEL_OIDC_TOKEN, when a request is made.
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithBaseURL replaces the model API base URL. Account endpoints use its
// origin.
func WithBaseURL(baseURL string) Option {
	return func(p *Provider) {
		p.baseURL = baseURL
	}
}

// WithTeamIDOrSlug scopes requests to a Vercel team, for access tokens that
// can access several teams.
func WithTeamIDOrSlug(teamIDOrSlug string) Option {
	return func(p *Provider) {
		p.teamIDOrSlug = teamIDOrSlug
	}
}

// WithHeaders adds headers to every request. The headers are copied.
func WithHeaders(headers http.Header) Option {
	snapshot := headers.Clone()

	return func(p *Provider) {
		p.headers = snapshot.Clone()
	}
}

// WithHTTPClient sets the client used for every request. Nil uses
// http.DefaultClient.
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// WithName replaces the provider name reported by the models. The default is
// "gateway". Provider options keep the "gateway" namespace.
func WithName(name string) Option {
	return func(p *Provider) {
		p.name = name
	}
}

// WithMetadataCacheRefresh sets how long AvailableModels reuses the model
// catalog. The default is five minutes.
func WithMetadataCacheRefresh(refresh time.Duration) Option {
	return func(p *Provider) {
		p.metadataCacheRefresh = refresh
	}
}

// New returns an AI Gateway provider.
func New(options ...Option) *Provider {
	provider := &Provider{
		baseURL:              defaultBaseURL,
		name:                 defaultName,
		metadataCacheRefresh: defaultMetadataCacheRefresh,
		catalog: &catalogCache{
			lock: make(chan struct{}, 1),
		},
	}

	for _, option := range options {
		option(provider)
	}

	provider.baseURL = strings.TrimRight(provider.baseURL, "/")

	return provider
}

// Chat returns a language model, such as "openai/gpt-5" or
// "anthropic/claude-sonnet-4.5".
func (p *Provider) Chat(modelID string) *ChatModel {
	return &ChatModel{
		modelID:  modelID,
		provider: p,
	}
}

// Embedding returns an embedding model, such as
// "openai/text-embedding-3-small".
func (p *Provider) Embedding(modelID string) *EmbeddingModel {
	return &EmbeddingModel{
		modelID:  modelID,
		provider: p,
	}
}

// Image returns an image model, such as "google/imagen-4.0-generate-001".
func (p *Provider) Image(modelID string) *ImageModel {
	return &ImageModel{
		modelID:  modelID,
		provider: p,
	}
}

// Reranking returns a reranking model, such as "cohere/rerank-v3.5".
func (p *Provider) Reranking(modelID string) *RerankingModel {
	return &RerankingModel{
		modelID:  modelID,
		provider: p,
	}
}

// Speech returns a text-to-speech model.
func (p *Provider) Speech(modelID string) *SpeechModel {
	return &SpeechModel{
		modelID:  modelID,
		provider: p,
	}
}

// Transcription returns a transcription model.
func (p *Provider) Transcription(modelID string) *TranscriptionModel {
	return &TranscriptionModel{
		modelID:  modelID,
		provider: p,
	}
}

// Video returns a video model, such as "google/veo-3.1-generate-001".
func (p *Provider) Video(modelID string) *VideoModel {
	return &VideoModel{
		modelID:  modelID,
		provider: p,
	}
}

// VideoModel returns Video(modelID).
func (p *Provider) VideoModel(modelID string) (model.VideoModel, error) {
	return p.Video(modelID), nil
}

// LanguageModel returns the language model for modelID.
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

// TranscriptionModel returns the transcription model for modelID.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return p.Transcription(modelID), nil
}

// SpeechModel returns the speech model for modelID.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return p.Speech(modelID), nil
}

// RerankingModel returns the reranking model for modelID.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return p.Reranking(modelID), nil
}

var (
	_ model.Provider      = (*Provider)(nil)
	_ model.VideoProvider = (*Provider)(nil)
)
