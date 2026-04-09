package model

import "encoding/json"

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

	// StopSequences causes the model to stop generating when any of
	// these strings are produced.
	StopSequences []string

	// Tools describes the tools available to the model.
	Tools []ToolDefinition

	// ToolChoice controls how the model selects tools.
	ToolChoice *ToolChoice

	// ResponseFormat requests a specific output format from the model.
	ResponseFormat *ResponseFormat

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

// ImagePart is an image segment, either inline data or a URL.
type ImagePart struct {
	// URL is the image URL. Mutually exclusive with Data.
	URL string

	// Data is the raw image bytes. Mutually exclusive with URL.
	Data []byte

	// MIMEType is the image media type (e.g. "image/png").
	MIMEType string
}

func (ImagePart) partType() {}

// ToolCallPart represents the model requesting a tool invocation.
type ToolCallPart struct {
	ToolCallID string
	ToolName   string
	Args       json.RawMessage
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
