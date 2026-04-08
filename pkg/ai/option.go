package ai

import (
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// Option configures a generation or streaming request.
type Option func(*config)

// config holds all generation parameters resolved from options.
type config struct {
	system         string
	prompt         string
	messages       []Message
	tools          []Tool
	maxSteps       int
	maxTokens      *int
	temperature    *float64
	topP           *float64
	stopSequences  []string
	timeout        time.Duration
	responseFormat *model.ResponseFormat
}

// WithSystem sets the system prompt.
func WithSystem(system string) Option {
	return func(c *config) { c.system = system }
}

// WithPrompt sets a single user message as the prompt.
func WithPrompt(prompt string) Option {
	return func(c *config) { c.prompt = prompt }
}

// WithMessages sets the full conversation history. This is mutually
// exclusive with WithPrompt — if both are used, messages takes
// precedence and prompt is appended.
func WithMessages(msgs ...Message) Option {
	return func(c *config) { c.messages = msgs }
}

// WithTools registers tools available to the model during generation.
// Set WithMaxSteps to enable the automatic tool execution loop.
func WithTools(tools ...Tool) Option {
	return func(c *config) { c.tools = tools }
}

// WithMaxSteps sets the maximum number of generation steps in a tool
// loop. A value of 1 (the default) means no tool loop — the model
// generates once. Higher values allow the model to call tools and
// continue generating.
func WithMaxSteps(n int) Option {
	return func(c *config) { c.maxSteps = n }
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

// WithStopSequences sets stop sequences that cause the model to stop
// generating.
func WithStopSequences(seqs ...string) Option {
	return func(c *config) { c.stopSequences = seqs }
}

// WithTimeout sets an overall timeout for the request.
func WithTimeout(d time.Duration) Option {
	return func(c *config) { c.timeout = d }
}
