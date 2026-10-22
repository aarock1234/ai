// Package openai provides OpenAI models: the Responses API (the default
// language model), Chat Completions, legacy Completions, embeddings, images,
// speech and transcription, and the experimental batch, decision, realtime
// and speech translation APIs.
package openai

import (
	"context"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"

	// Namespace keys OpenAI provider options, file references and metadata.
	Namespace = "openai"
)

// Provider creates OpenAI models. The zero value is not usable; call New.
type Provider struct {
	config config
}

// config is shared by every model a provider creates.
type config struct {
	// name prefixes model provider identifiers, such as "openai.responses".
	name string

	baseURL      string
	apiKey       string
	organization string
	project      string
	headers      http.Header
	httpClient   *http.Client
	generateID   func() string

	// requestURL replaces joining baseURL and the path when set.
	requestURL RequestURLFunc

	// authHeaders replaces the bearer API key header when set.
	authHeaders AuthHeadersFunc

	// optionsNamespace keys the Responses API provider options and metadata.
	// Options under Namespace are read when it has none.
	optionsNamespace string

	// explicitMessageItems adds "type": "message" to Responses API message
	// items.
	explicitMessageItems bool

	// omitWebSearchSourcesInclude stops Responses API calls with a web
	// search tool from requesting the search sources, for hosts that reject
	// that include value.
	omitWebSearchSourcesInclude bool

	// requestModelID is the model ID sent in requests. Model constructors
	// set it to the model ID unless WithRequestModelID replaces it.
	requestModelID string

	// batchMaxLineBytes bounds one line of a batch result file. Zero
	// selects 64 MiB.
	batchMaxLineBytes int
}

// Option configures a Provider.
type Option func(*config)

// WithAPIKey sets the API key. The default is the OPENAI_API_KEY
// environment variable, read when a request is sent.
func WithAPIKey(apiKey string) Option {
	return func(c *config) {
		c.apiKey = apiKey
	}
}

// WithBaseURL sets the API base URL. The default is the OPENAI_BASE_URL
// environment variable, or https://api.openai.com/v1.
func WithBaseURL(baseURL string) Option {
	return func(c *config) {
		c.baseURL = strings.TrimSuffix(baseURL, "/")
	}
}

// WithOrganization sets the OpenAI-Organization header.
func WithOrganization(organization string) Option {
	return func(c *config) {
		c.organization = organization
	}
}

// WithProject sets the OpenAI-Project header.
func WithProject(project string) Option {
	return func(c *config) {
		c.project = project
	}
}

// WithHeaders adds headers to every request.
func WithHeaders(headers http.Header) Option {
	return func(c *config) {
		c.headers = headers.Clone()
	}
}

// WithHTTPClient sets the HTTP client. The default is http.DefaultClient.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) {
		c.httpClient = client
	}
}

// WithName sets the provider name used in model provider identifiers. The
// default is "openai".
func WithName(name string) Option {
	return func(c *config) {
		c.name = name
	}
}

// WithIDGenerator sets the function that creates IDs for sources and for
// tool calls that arrive without one.
func WithIDGenerator(generateID func() string) Option {
	return func(c *config) {
		c.generateID = generateID
	}
}

// WithBatchResultMaxLineBytes bounds one line of a downloaded batch result
// file. The default is 64 MiB.
func WithBatchResultMaxLineBytes(maxLineBytes int) Option {
	return func(c *config) {
		c.batchMaxLineBytes = maxLineBytes
	}
}

// New returns a provider configured by options.
func New(options ...Option) *Provider {
	c := config{
		name:             "openai",
		optionsNamespace: Namespace,
		generateID: func() string {
			return id.New("")
		},
		baseURL: strings.TrimSuffix(os.Getenv("OPENAI_BASE_URL"), "/"),
	}

	for _, option := range options {
		option(&c)
	}

	if c.baseURL == "" {
		c.baseURL = defaultBaseURL
	}

	return &Provider{
		config: c,
	}
}

// Responses returns a Responses API language model.
func (p *Provider) Responses(modelID string, options ...ModelOption) *ResponsesModel {
	return newResponsesModel(modelID, p.config.forModel(modelID, options))
}

// Chat returns a Chat Completions language model.
func (p *Provider) Chat(modelID string, options ...ModelOption) *ChatModel {
	return newChatModel(modelID, p.config.forModel(modelID, options))
}

// Completion returns a legacy Completions language model.
func (p *Provider) Completion(modelID string, options ...ModelOption) *CompletionModel {
	return newCompletionModel(modelID, p.config.forModel(modelID, options))
}

// Embedding returns an embedding model.
func (p *Provider) Embedding(modelID string) *EmbeddingModel {
	return newEmbeddingModel(modelID, p.config.forModel(modelID, nil))
}

// Image returns an image model.
func (p *Provider) Image(modelID string) *ImageModel {
	return newImageModel(modelID, p.config.forModel(modelID, nil))
}

// Speech returns a speech model.
func (p *Provider) Speech(modelID string) *SpeechModel {
	return newSpeechModel(modelID, p.config.forModel(modelID, nil))
}

// Transcription returns a transcription model.
func (p *Provider) Transcription(modelID string) *TranscriptionModel {
	return newTranscriptionModel(modelID, p.config.forModel(modelID, nil))
}

// Decision returns a Decisions API model. Decision models are
// experimental.
func (p *Provider) Decision(modelID string) *DecisionModel {
	return newDecisionModel(modelID, p.config.forModel(modelID, nil))
}

// SpeechTranslation returns a realtime speech translation model, such as
// gpt-realtime-translate. Speech translation is experimental.
func (p *Provider) SpeechTranslation(modelID string) *SpeechTranslationModel {
	return newSpeechTranslationModel(modelID, p.config.forModel(modelID, nil))
}

// Realtime returns a Realtime API model, such as gpt-realtime. Realtime
// models are experimental.
func (p *Provider) Realtime(modelID string) *RealtimeModel {
	return newRealtimeModel(modelID, p.config.forModel(modelID, nil))
}

// Live returns a Live API model, such as gpt-live-1, for continuous
// conversations. Live models are experimental.
func (p *Provider) Live(modelID string) *LiveModel {
	return newLiveModel(modelID, p.config.forModel(modelID, nil))
}

// Batch returns the batch interface, an *Batch that processes Responses
// API requests with the Batch API. Batches are experimental.
func (p *Provider) Batch() model.Batch {
	return &Batch{
		config: p.config,
	}
}

// DecisionModel returns a Decisions API model.
func (p *Provider) DecisionModel(modelID string) (model.DecisionModel, error) {
	return p.Decision(modelID), nil
}

// SpeechTranslationModel returns a realtime speech translation model.
func (p *Provider) SpeechTranslationModel(modelID string) (model.SpeechTranslationModel, error) {
	return p.SpeechTranslation(modelID), nil
}

// RealtimeModel returns a realtime model: a Live API model for gpt-live-1
// and a Realtime API model otherwise.
func (p *Provider) RealtimeModel(modelID string) (model.RealtimeModel, error) {
	if slices.Contains(liveModelIDs, modelID) {
		return p.Live(modelID), nil
	}

	return p.Realtime(modelID), nil
}

// Files returns the files interface, an *Files that uploads, inspects,
// downloads and deletes files.
func (p *Provider) Files() model.Files {
	return &Files{
		config: p.config,
	}
}

// Skills returns the skills interface, an *Skills.
func (p *Provider) Skills() model.Skills {
	return &Skills{
		config: p.config,
	}
}

// LanguageModel returns a Responses API model.
func (p *Provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return p.Responses(modelID), nil
}

// EmbeddingModel returns an embedding model.
func (p *Provider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return p.Embedding(modelID), nil
}

// ImageModel returns an image model.
func (p *Provider) ImageModel(modelID string) (model.ImageModel, error) {
	return p.Image(modelID), nil
}

// SpeechModel returns a speech model.
func (p *Provider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return p.Speech(modelID), nil
}

// TranscriptionModel returns a transcription model.
func (p *Provider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return p.Transcription(modelID), nil
}

// RerankingModel reports that OpenAI has no reranking models.
func (p *Provider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return nil, model.NoSuchModelError(p.config.name, model.RerankingModelKind, modelID)
}

// request prepares a call to path, such as "/responses". Call headers
// override provider headers.
func (c config) request(ctx context.Context, path string, callHeaders http.Header) (apicall.Request, error) {
	headers, err := c.authenticationHeaders(ctx)
	if err != nil {
		return apicall.Request{}, err
	}

	if c.organization != "" {
		headers.Set("OpenAI-Organization", c.organization)
	}

	if c.project != "" {
		headers.Set("OpenAI-Project", c.project)
	}

	requestURL, err := c.url(path)
	if err != nil {
		return apicall.Request{}, err
	}

	return apicall.Request{
		Client:     c.httpClient,
		URL:        requestURL,
		Headers:    apicall.MergeHeaders(headers, c.headers, callHeaders),
		ParseError: parseError,
	}, nil
}

// authenticationHeaders returns the configured authentication headers, or a
// bearer header with the API key.
func (c config) authenticationHeaders(ctx context.Context) (http.Header, error) {
	if c.authHeaders != nil {
		headers, err := c.authHeaders(ctx)
		if err != nil {
			return nil, err
		}

		return apicall.MergeHeaders(headers), nil
	}

	apiKey, err := apikey.Load(c.apiKey, "OPENAI_API_KEY")
	if err != nil {
		return nil, err
	}

	return http.Header{
		"Authorization": {
			"Bearer " + apiKey,
		},
	}, nil
}

// url returns the URL of path for the model's request model ID.
func (c config) url(path string) (string, error) {
	if c.requestURL == nil {
		return c.baseURL + path, nil
	}

	return c.requestURL(Endpoint{
		Path:    path,
		ModelID: c.requestModelID,
	})
}

// modelProvider returns the provider identifier of one model family, such
// as "openai.chat".
func (c config) modelProvider(family string) string {
	return c.name + "." + family
}
