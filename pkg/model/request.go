package model

import (
	"encoding/json"
	"net/http"
)

// ProviderOptions carries provider-scoped request options. The first key
// is the provider name (e.g. "anthropic", "openai"); the nested option
// set contains provider-defined JSON values for that request.
type ProviderOptions map[string]ProviderOptionSet

// Clone returns a deep copy of the provider options.
func (o ProviderOptions) Clone() ProviderOptions {
	if len(o) == 0 {
		return nil
	}

	cloned := make(ProviderOptions, len(o))
	for provider, opts := range o {
		cloned[provider] = opts.Clone()
	}

	return cloned
}

// Lookup returns the raw JSON value for a provider-specific request option.
func (o ProviderOptions) Lookup(provider, key string) (json.RawMessage, bool) {
	if len(o) == 0 {
		return nil, false
	}

	opts, ok := o[provider]
	if !ok {
		return nil, false
	}

	return opts.Lookup(key)
}

// ProviderOptionSet is a single provider's request option map.
type ProviderOptionSet map[string]json.RawMessage

// Clone returns a deep copy of the option set.
func (s ProviderOptionSet) Clone() ProviderOptionSet {
	if len(s) == 0 {
		return nil
	}

	cloned := make(ProviderOptionSet, len(s))
	for key, value := range s {
		cloned[key] = append(json.RawMessage(nil), value...)
	}

	return cloned
}

// Lookup returns a copied raw JSON value for a provider-specific option.
func (s ProviderOptionSet) Lookup(key string) (json.RawMessage, bool) {
	if len(s) == 0 {
		return nil, false
	}

	value, ok := s[key]
	if !ok {
		return nil, false
	}

	return append(json.RawMessage(nil), value...), true
}

// Modality identifies an output type requested from or produced by the
// model.
type Modality string

const (
	// ModalityText requests text output.
	ModalityText Modality = "text"

	// ModalityImage requests image output.
	ModalityImage Modality = "image"

	// ModalityAudio requests audio output.
	ModalityAudio Modality = "audio"
)

// AudioOptions configures audio output. Used by models that can emit
// audio via the chat completions API (e.g. OpenAI gpt-audio).
type AudioOptions struct {
	// Voice is the voice identifier (e.g. "alloy", "sage").
	Voice string

	// Format is the audio encoding (e.g. "wav", "mp3", "pcm16").
	Format string
}

// ReasoningOptions configures reasoning token output. Providers that
// support reasoning models (OpenAI o-series, Anthropic extended thinking,
// DeepSeek-R1 via OpenRouter) interpret these fields.
type ReasoningOptions struct {
	// Effort is a coarse intensity dial ("low", "medium", "high").
	// Providers map this to a sensible reasoning budget.
	Effort string

	// MaxTokens sets an explicit reasoning token budget. When non-zero,
	// it takes precedence over Effort on providers that honor both.
	MaxTokens int

	// Exclude asks the provider to compute reasoning without returning
	// it in the response.
	Exclude bool
}

// PredictionOptions configures predicted outputs. OpenAI uses this to
// bias the model toward content the caller expects to regenerate
// largely unchanged (e.g. code-edit workflows).
type PredictionOptions struct {
	// Content is the predicted content string.
	Content string
}

// Request is the normalized request shape that the SDK builds and passes
// to Model.DoGenerate and Model.DoStream. Providers translate this into
// their wire format.
type Request struct {
	// Messages is the conversation history in chronological order.
	Messages []Message

	// Temperature controls randomness. Nil means use provider default.
	Temperature *float64

	// MaxTokens limits the response length. Nil means use provider default.
	MaxTokens *int

	// TopP controls nucleus sampling. Nil means use provider default.
	TopP *float64

	// TopK limits sampling to the K most likely tokens. Nil means use
	// provider default.
	TopK *int

	// PresencePenalty discourages repeating existing topics. Nil means
	// use provider default.
	PresencePenalty *float64

	// FrequencyPenalty discourages repeating existing tokens. Nil means
	// use provider default.
	FrequencyPenalty *float64

	// Seed requests deterministic sampling when supported by the
	// provider. Nil means use provider default.
	Seed *int64

	// StopSequences causes the model to stop generating when any of
	// these strings are produced.
	StopSequences []string

	// Tools describes the tools available to the model.
	Tools []ToolDefinition

	// ActiveTools limits the set of tools exposed for this request.
	// When empty, all tools in Tools are active.
	ActiveTools []string

	// ToolChoice controls how the model selects tools.
	ToolChoice *ToolChoice

	// ResponseFormat requests a specific output format from the model.
	ResponseFormat *ResponseFormat

	// Modalities requests specific output types (text, image, audio).
	// When empty, the provider default applies.
	Modalities []Modality

	// Audio configures audio output parameters. Required by most
	// providers when Modalities contains ModalityAudio.
	Audio *AudioOptions

	// Reasoning configures reasoning token behavior. Nil uses the
	// provider default.
	Reasoning *ReasoningOptions

	// Prediction carries predicted output content, when supported.
	Prediction *PredictionOptions

	// Headers are additional HTTP headers sent with this request.
	Headers http.Header

	// ProviderOptions carries provider-specific request options keyed by
	// provider name.
	ProviderOptions ProviderOptions
}

// Message is a single message in the conversation. The Parts field
// carries the actual content; Role identifies the speaker.
type Message struct {
	Role  Role
	Parts []Part
}

// Role identifies the speaker of a message.
type Role string

const (
	// RoleSystem identifies a system-level message that sets context.
	RoleSystem Role = "system"

	// RoleUser identifies a message from the user.
	RoleUser Role = "user"

	// RoleAssistant identifies a message from the model.
	RoleAssistant Role = "assistant"

	// RoleTool identifies a message carrying a tool execution result.
	RoleTool Role = "tool"
)

// Part is the interface for message content segments. A message may
// contain multiple parts (e.g. text followed by a tool call).
type Part interface {
	partType()
}

// TextPart is a plain text segment.
type TextPart struct {
	Text string
}

func (TextPart) partType() {}

// ImagePart is an image segment, either inline data or a URL. This is
// equivalent to FilePart with an image/* MIMEType but kept as a
// distinct type for clarity because vision models are the most common
// multimodal case.
type ImagePart struct {
	// URL is the image URL. Mutually exclusive with Data.
	URL string

	// Data is the raw image bytes. Mutually exclusive with URL.
	Data []byte

	// MIMEType is the image media type (e.g. "image/png").
	MIMEType string
}

func (ImagePart) partType() {}

// FilePart is a generic file segment (PDF, audio, non-image). User
// messages use FilePart for document inputs; assistant messages use it
// to round-trip generated files back to the provider in follow-up
// requests.
type FilePart struct {
	// URL is the file URL. Mutually exclusive with Data.
	URL string

	// Data is the raw file bytes. Mutually exclusive with URL.
	Data []byte

	// MIMEType is the file media type (e.g. "application/pdf",
	// "audio/wav").
	MIMEType string

	// Filename is an optional filename hint for the provider.
	Filename string
}

func (FilePart) partType() {}

// ReasoningPart carries reasoning content from a prior assistant turn.
// Sending reasoning back to providers that support it (Anthropic
// extended thinking, OpenRouter reasoning models) preserves the
// reasoning state across turns.
type ReasoningPart struct {
	// Text is the plaintext reasoning content. Empty for redacted blocks.
	Text string

	// Signature is the provider-supplied opaque signature for this block.
	Signature string

	// ID is the provider-assigned identifier for this block.
	ID string

	// Format identifies the reasoning block's provider-specific shape.
	Format string

	// Redacted indicates the reasoning is encrypted or filtered.
	Redacted bool
}

func (ReasoningPart) partType() {}

// ToolCallPart represents the model requesting a tool invocation.
type ToolCallPart struct {
	ToolCallID string
	ToolName   string
	Args       json.RawMessage

	// ProviderExecuted is true when the provider ran the tool itself.
	ProviderExecuted bool
}

func (ToolCallPart) partType() {}

// ToolResultPart carries the result of a tool invocation back to the
// model.
type ToolResultPart struct {
	ToolCallID string
	ToolName   string
	Content    string
	IsError    bool
}

func (ToolResultPart) partType() {}

// ToolDefinition describes a tool available to the model.
type ToolDefinition struct {
	Name        string
	Description string
	InputSchema json.RawMessage
	Strict      bool
}

// ToolChoice controls how the model selects tools.
type ToolChoice struct {
	// Type is one of "auto", "none", "required", or "tool".
	Type string

	// ToolName is the specific tool to call when Type is "tool".
	ToolName string
}

// ResponseFormat requests a specific output structure.
type ResponseFormat struct {
	// Type is "json_object" or "json_schema".
	Type string

	// JSONSchema is the schema to enforce when Type is "json_schema".
	JSONSchema *JSONSchema
}

// JSONSchema describes a JSON schema for structured output.
type JSONSchema struct {
	Name   string
	Schema json.RawMessage
	Strict bool
}
