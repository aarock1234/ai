package model

import (
	"encoding/json"
	"net/http"
)

// CallOptions are the inputs of one language model call. Nil pointers and
// empty values leave the provider default.
type CallOptions struct {
	Prompt []Message

	MaxOutputTokens  *int
	Temperature      *float64
	StopSequences    []string
	TopP             *float64
	TopK             *int
	PresencePenalty  *float64
	FrequencyPenalty *float64
	Seed             *int

	// ResponseFormat requests text or JSON output. Nil means text.
	ResponseFormat *ResponseFormat

	Tools      []Tool
	ToolChoice *ToolChoice

	// Reasoning selects a provider-independent reasoning effort.
	Reasoning ReasoningEffort

	// IncludeRawChunks asks streaming calls to emit Raw parts with the
	// provider's native events.
	IncludeRawChunks bool

	// Headers are additional HTTP request headers.
	Headers http.Header

	ProviderOptions ProviderOptions
}

// ResponseFormatType selects the output format.
type ResponseFormatType string

// Response format types.
const (
	ResponseFormatText ResponseFormatType = "text"
	ResponseFormatJSON ResponseFormatType = "json"
)

// ResponseFormat requests a specific output format.
type ResponseFormat struct {
	Type ResponseFormatType

	// Schema optionally constrains JSON output.
	Schema json.RawMessage

	// Name and Description optionally describe the JSON output.
	Name        string
	Description string
}

// ReasoningEffort is a provider-independent reasoning level. The empty value
// leaves the request unchanged.
type ReasoningEffort string

// Reasoning effort levels.
const (
	ReasoningProviderDefault ReasoningEffort = "provider-default"
	ReasoningNone            ReasoningEffort = "none"
	ReasoningMinimal         ReasoningEffort = "minimal"
	ReasoningLow             ReasoningEffort = "low"
	ReasoningMedium          ReasoningEffort = "medium"
	ReasoningHigh            ReasoningEffort = "high"
	ReasoningXHigh           ReasoningEffort = "xhigh"
	ReasoningMax             ReasoningEffort = "max"
)
