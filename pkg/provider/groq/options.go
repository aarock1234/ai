package groq

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// ReasoningFormat selects how reasoning is returned.
type ReasoningFormat string

// Reasoning formats.
const (
	ReasoningFormatParsed ReasoningFormat = "parsed"
	ReasoningFormatRaw    ReasoningFormat = "raw"
	ReasoningFormatHidden ReasoningFormat = "hidden"
)

// ReasoningEffort is a Groq reasoning effort.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortDefault ReasoningEffort = "default"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
)

// ServiceTier selects the Groq serving tier.
type ServiceTier string

// Service tiers.
const (
	// ServiceTierOnDemand has consistent performance and fairness. It is the
	// default.
	ServiceTierOnDemand ServiceTier = "on_demand"

	// ServiceTierPerformance prioritizes latency-sensitive workloads.
	ServiceTierPerformance ServiceTier = "performance"

	// ServiceTierFlex has higher throughput for workloads that tolerate
	// occasional failures.
	ServiceTierFlex ServiceTier = "flex"

	// ServiceTierAuto uses on-demand rate limits and then falls back to the
	// flex tier.
	ServiceTierAuto ServiceTier = "auto"
)

// ChatOptions are the provider options of chat models, read from the "groq"
// namespace.
type ChatOptions struct {
	// ReasoningFormat selects how reasoning is returned.
	ReasoningFormat ReasoningFormat `json:"reasoningFormat,omitempty"`

	// ReasoningEffort selects the reasoning effort. It takes precedence over
	// model.CallOptions.Reasoning.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty"`

	// ParallelToolCalls allows several tool calls per response. It defaults
	// to true.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`

	// User identifies the end user, to help Groq detect abuse.
	User *string `json:"user,omitempty"`

	// StructuredOutputs sends JSON schemas as response formats. It defaults
	// to true.
	StructuredOutputs *bool `json:"structuredOutputs,omitempty"`

	// StrictJSONSchema requests constrained decoding for JSON schema response
	// formats. It defaults to true.
	StrictJSONSchema *bool `json:"strictJsonSchema,omitempty"`

	// ServiceTier selects the serving tier.
	ServiceTier ServiceTier `json:"serviceTier,omitempty"`
}

// readChatOptions reads and validates the call's Groq options.
func readChatOptions(options model.ProviderOptions) (ChatOptions, error) {
	chatOptions, _, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return ChatOptions{}, err
	}

	switch chatOptions.ReasoningFormat {
	case "", ReasoningFormatParsed, ReasoningFormatRaw, ReasoningFormatHidden:
	default:
		return ChatOptions{}, fmt.Errorf("%w: groq reasoningFormat %q", model.ErrInvalidArgument, chatOptions.ReasoningFormat)
	}

	switch chatOptions.ReasoningEffort {
	case "", ReasoningEffortNone, ReasoningEffortDefault, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh:
	default:
		return ChatOptions{}, fmt.Errorf("%w: groq reasoningEffort %q", model.ErrInvalidArgument, chatOptions.ReasoningEffort)
	}

	switch chatOptions.ServiceTier {
	case "", ServiceTierOnDemand, ServiceTierPerformance, ServiceTierFlex, ServiceTierAuto:
		return chatOptions, nil
	default:
		return ChatOptions{}, fmt.Errorf("%w: groq serviceTier %q", model.ErrInvalidArgument, chatOptions.ServiceTier)
	}
}

// TranscriptionOptions are the provider options of transcription models,
// read from the "groq" namespace.
type TranscriptionOptions struct {
	// Language of the audio in ISO-639-1 format, such as "en".
	Language *string `json:"language,omitempty"`

	// Prompt guides the style or the spelling of unfamiliar words. It is
	// limited to 224 tokens.
	Prompt *string `json:"prompt,omitempty"`

	// ResponseFormat selects the output format, such as "json",
	// "verbose_json" for timestamps, or "text".
	ResponseFormat *string `json:"responseFormat,omitempty"`

	// Temperature is the sampling temperature, from 0 to 1.
	Temperature *float64 `json:"temperature,omitempty"`

	// TimestampGranularities selects "word" and "segment" timestamps. They
	// require the "verbose_json" response format.
	TimestampGranularities []string `json:"timestampGranularities,omitempty"`
}

// readTranscriptionOptions reads and validates the call's Groq options.
func readTranscriptionOptions(options model.ProviderOptions) (TranscriptionOptions, error) {
	transcriptionOptions, _, err := model.ProviderValue[TranscriptionOptions](options, Namespace)
	if err != nil {
		return TranscriptionOptions{}, err
	}

	temperature := transcriptionOptions.Temperature
	if temperature != nil && (*temperature < 0 || *temperature > 1) {
		return TranscriptionOptions{}, fmt.Errorf("%w: groq temperature must be between 0 and 1", model.ErrInvalidArgument)
	}

	return transcriptionOptions, nil
}
