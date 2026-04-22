package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// Option configures a generation or streaming request.
type Option func(*config)

// config holds all generation parameters resolved from options.
type config struct {
	system           string
	prompt           string
	messages         []Message
	tools            []Tool
	activeTools      []string
	toolChoice       *model.ToolChoice
	maxSteps         int
	maxRetries       int
	maxTokens        *int
	temperature      *float64
	topP             *float64
	topK             *int
	presencePenalty  *float64
	frequencyPenalty *float64
	seed             *int64
	stopSequences    []string
	timeout          time.Duration
	responseFormat   *model.ResponseFormat
	modalities       []model.Modality
	audio            *model.AudioOptions
	reasoning        *model.ReasoningOptions
	prediction       *model.PredictionOptions
	headers          http.Header
	providerOpts     model.ProviderOptions
	stopWhen         []StopWhenFunc
	prepareStep      []PrepareStepFunc
	onStepFinish     []OnStepFinishFunc
	onFinish         []OnFinishFunc
	err              error
}

// StopWhenFunc decides whether a generation loop should stop after the
// current step.
type StopWhenFunc func(step Step) bool

// PrepareStepFunc mutates a request immediately before a generation step runs.
type PrepareStepFunc func(ctx context.Context, step int, req *model.Request) error

// OnStepFinishFunc observes a completed generation step.
type OnStepFinishFunc func(ctx context.Context, step Step) error

// OnFinishFunc observes the final result of a generation request.
type OnFinishFunc func(ctx context.Context, result *GenerateResult) error

// WithSystem sets the system prompt.
func WithSystem(system string) Option {
	return func(c *config) { c.system = system }
}

// WithPrompt sets a single user message as the prompt.
func WithPrompt(prompt string) Option {
	return func(c *config) { c.prompt = prompt }
}

// WithMessages sets the full conversation history. If WithPrompt is
// also used, the prompt is appended as the final user message.
func WithMessages(msgs ...Message) Option {
	return func(c *config) { c.messages = msgs }
}

// WithTools registers tools available to the model during generation.
// Set WithMaxSteps to enable the automatic tool execution loop.
func WithTools(tools ...Tool) Option {
	return func(c *config) { c.tools = tools }
}

// WithActiveTools limits the set of tool definitions included in the request.
func WithActiveTools(names ...string) Option {
	return func(c *config) { c.activeTools = append([]string(nil), names...) }
}

// WithToolChoice sets the model's tool selection policy.
func WithToolChoice(choice model.ToolChoice) Option {
	return func(c *config) {
		c.toolChoice = &model.ToolChoice{
			Type:     choice.Type,
			ToolName: choice.ToolName,
		}
	}
}

// WithToolChoiceAuto lets the model decide whether to call a tool.
func WithToolChoiceAuto() Option {
	return WithToolChoice(model.ToolChoice{Type: "auto"})
}

// WithToolChoiceNone disables tool calling.
func WithToolChoiceNone() Option {
	return WithToolChoice(model.ToolChoice{Type: "none"})
}

// WithToolChoiceRequired requires the model to call a tool.
func WithToolChoiceRequired() Option {
	return WithToolChoice(model.ToolChoice{Type: "required"})
}

// WithToolChoiceNamed forces the model to call a specific tool.
func WithToolChoiceNamed(name string) Option {
	return WithToolChoice(model.ToolChoice{
		Type:     "tool",
		ToolName: name,
	})
}

// WithMaxSteps sets the maximum number of generation steps in a tool
// loop. A value of 1 (the default) means no tool loop — the model
// generates once. Higher values allow the model to call tools and
// continue generating.
func WithMaxSteps(n int) Option {
	return func(c *config) { c.maxSteps = n }
}

// WithMaxRetries retries retryable provider failures up to n times.
func WithMaxRetries(n int) Option {
	return func(c *config) { c.maxRetries = n }
}

// WithMaxTokens limits the response length.
func WithMaxTokens(n int) Option {
	return func(c *config) { c.maxTokens = &n }
}

// WithTemperature controls randomness (0.0 to 2.0).
func WithTemperature(t float64) Option {
	return func(c *config) { c.temperature = &t }
}

// WithTopP controls nucleus sampling.
func WithTopP(p float64) Option {
	return func(c *config) { c.topP = &p }
}

// WithTopK controls top-k sampling.
func WithTopK(k int) Option {
	return func(c *config) { c.topK = &k }
}

// WithPresencePenalty discourages introducing already-seen topics.
func WithPresencePenalty(p float64) Option {
	return func(c *config) { c.presencePenalty = &p }
}

// WithFrequencyPenalty discourages repeating existing tokens.
func WithFrequencyPenalty(p float64) Option {
	return func(c *config) { c.frequencyPenalty = &p }
}

// WithSeed requests deterministic sampling when supported.
func WithSeed(seed int64) Option {
	return func(c *config) { c.seed = &seed }
}

// WithStopSequences sets stop sequences that cause the model to stop
// generating.
func WithStopSequences(seqs ...string) Option {
	return func(c *config) { c.stopSequences = seqs }
}

// WithHeaders adds per-request headers.
func WithHeaders(headers http.Header) Option {
	return func(c *config) {
		if len(headers) == 0 {
			return
		}

		if c.headers == nil {
			c.headers = make(http.Header)
		}

		for key, values := range headers {
			for _, value := range values {
				c.headers.Add(key, value)
			}
		}
	}
}

// WithHeader adds a single per-request header.
func WithHeader(key, value string) Option {
	return WithHeaders(http.Header{
		key: {value},
	})
}

// WithProviderOption sets a provider-specific request option for a single
// call. The value is JSON-encoded and stored under the provider name.
func WithProviderOption(provider, key string, value any) Option {
	return func(c *config) {
		if provider == "" || key == "" {
			return
		}

		encoded, err := json.Marshal(value)
		if err != nil {
			c.err = errors.Join(c.err, fmt.Errorf("provider option %s.%s: %w", provider, key, err))
			return
		}

		if c.providerOpts == nil {
			c.providerOpts = make(model.ProviderOptions)
		}

		if c.providerOpts[provider] == nil {
			c.providerOpts[provider] = make(model.ProviderOptionSet)
		}

		c.providerOpts[provider][key] = encoded
	}
}

// WithTimeout sets an overall timeout for the request.
func WithTimeout(d time.Duration) Option {
	return func(c *config) { c.timeout = d }
}

// WithModalities requests specific output types from the model. Most
// providers default to text-only; pass ModalityImage or ModalityAudio
// to unlock multimodal output on models that support it (e.g. OpenAI
// gpt-audio, Gemini image-out via OpenRouter).
func WithModalities(modalities ...model.Modality) Option {
	return func(c *config) {
		c.modalities = append([]model.Modality(nil), modalities...)
	}
}

// WithAudio configures audio output. Required when ModalityAudio is
// requested on providers that emit audio via chat completions.
func WithAudio(voice, format string) Option {
	return func(c *config) {
		c.audio = &model.AudioOptions{
			Voice:  voice,
			Format: format,
		}
	}
}

// WithReasoning configures reasoning token behavior for reasoning
// models. Effort is one of "low", "medium", "high"; MaxTokens bounds
// the reasoning budget explicitly.
func WithReasoning(opts model.ReasoningOptions) Option {
	return func(c *config) {
		c.reasoning = &model.ReasoningOptions{
			Effort:    opts.Effort,
			MaxTokens: opts.MaxTokens,
			Exclude:   opts.Exclude,
		}
	}
}

// WithReasoningEffort is a convenience wrapper that sets only the
// reasoning effort level.
func WithReasoningEffort(effort string) Option {
	return WithReasoning(model.ReasoningOptions{Effort: effort})
}

// WithPrediction attaches predicted output content to the request,
// biasing the model toward regenerating the provided text.
func WithPrediction(content string) Option {
	return func(c *config) {
		c.prediction = &model.PredictionOptions{Content: content}
	}
}

// WithStopWhen registers a generation stop predicate. If any predicate
// returns true, the current result is returned immediately.
func WithStopWhen(fn StopWhenFunc) Option {
	return func(c *config) {
		if fn != nil {
			c.stopWhen = append(c.stopWhen, fn)
		}
	}
}

// WithPrepareStep registers a hook that runs before each generation step.
func WithPrepareStep(fn PrepareStepFunc) Option {
	return func(c *config) {
		if fn != nil {
			c.prepareStep = append(c.prepareStep, fn)
		}
	}
}

// WithOnStepFinish registers a hook that runs after each generation step.
func WithOnStepFinish(fn OnStepFinishFunc) Option {
	return func(c *config) {
		if fn != nil {
			c.onStepFinish = append(c.onStepFinish, fn)
		}
	}
}

// WithOnFinish registers a hook that runs when generation finishes.
func WithOnFinish(fn OnFinishFunc) Option {
	return func(c *config) {
		if fn != nil {
			c.onFinish = append(c.onFinish, fn)
		}
	}
}
