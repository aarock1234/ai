package mistral

import (
	"encoding/json"
	"fmt"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// ReasoningEffort is a Mistral reasoning effort.
type ReasoningEffort string

// Reasoning efforts.
const (
	// ReasoningEffortHigh enables reasoning.
	ReasoningEffortHigh ReasoningEffort = "high"

	// ReasoningEffortNone disables reasoning.
	ReasoningEffortNone ReasoningEffort = "none"
)

// ChatOptions are the provider options of chat and conversation models, read
// from the "mistral" namespace. The Conversations API does not support
// SafePrompt, the document limits, PromptCacheKey or ParallelToolCalls.
type ChatOptions struct {
	// SafePrompt injects a safety prompt before the conversation. It defaults
	// to false.
	SafePrompt *bool `json:"safePrompt,omitempty"`

	// DocumentImageLimit limits the images read from documents.
	DocumentImageLimit *int `json:"documentImageLimit,omitempty"`

	// DocumentPageLimit limits the pages read from documents.
	DocumentPageLimit *int `json:"documentPageLimit,omitempty"`

	// StructuredOutputs sends JSON schemas as response formats. It defaults
	// to true; otherwise the schema is added to the system prompt.
	StructuredOutputs *bool `json:"structuredOutputs,omitempty"`

	// StrictJSONSchema requests strict schema adherence. It defaults to
	// false.
	StrictJSONSchema *bool `json:"strictJsonSchema,omitempty"`

	// ParallelToolCalls allows several tool calls per response. It defaults
	// to true.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`

	// PromptCacheKey routes requests with a shared prompt prefix to the same
	// cache.
	PromptCacheKey *string `json:"promptCacheKey,omitempty"`

	// ReasoningEffort enables or disables reasoning on models that support
	// adjustable reasoning.
	ReasoningEffort *ReasoningEffort `json:"reasoningEffort,omitempty"`
}

// readChatOptions reads and validates the call's Mistral options.
func readChatOptions(options model.ProviderOptions) (ChatOptions, error) {
	chatOptions, _, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return ChatOptions{}, err
	}

	if chatOptions.ReasoningEffort == nil {
		return chatOptions, nil
	}

	switch *chatOptions.ReasoningEffort {
	case ReasoningEffortHigh, ReasoningEffortNone:
		return chatOptions, nil
	default:
		return ChatOptions{}, fmt.Errorf("%w: mistral reasoningEffort %q", model.ErrInvalidArgument, *chatOptions.ReasoningEffort)
	}
}

// EmbeddingDataType is the data type of output embeddings.
type EmbeddingDataType string

// Embedding data types.
const (
	EmbeddingFloat   EmbeddingDataType = "float"
	EmbeddingInt8    EmbeddingDataType = "int8"
	EmbeddingUint8   EmbeddingDataType = "uint8"
	EmbeddingBinary  EmbeddingDataType = "binary"
	EmbeddingUbinary EmbeddingDataType = "ubinary"
)

// EmbeddingOptions are the provider options of embedding models.
type EmbeddingOptions struct {
	// Metadata is attached to the embedding request.
	Metadata map[string]json.RawMessage `json:"metadata,omitempty"`

	// OutputDimension requests a vector size from models that support it.
	// It takes precedence over model.EmbedOptions.Dimensions.
	OutputDimension *int `json:"outputDimension,omitempty"`

	// OutputDtype requests a data type from models that support it.
	OutputDtype EmbeddingDataType `json:"outputDtype,omitempty"`
}

func readEmbeddingOptions(options model.ProviderOptions) (EmbeddingOptions, error) {
	embeddingOptions, _, err := model.ProviderValue[EmbeddingOptions](options, Namespace)
	if err != nil {
		return EmbeddingOptions{}, err
	}

	if embeddingOptions.OutputDimension != nil && *embeddingOptions.OutputDimension <= 0 {
		return EmbeddingOptions{}, fmt.Errorf("%w: mistral outputDimension must be positive", model.ErrInvalidArgument)
	}

	switch embeddingOptions.OutputDtype {
	case "", EmbeddingFloat, EmbeddingInt8, EmbeddingUint8, EmbeddingBinary, EmbeddingUbinary:
		return embeddingOptions, nil
	default:
		return EmbeddingOptions{}, fmt.Errorf("%w: mistral outputDtype %q", model.ErrInvalidArgument, embeddingOptions.OutputDtype)
	}
}

// SpeechOptions are the provider options of speech models.
type SpeechOptions struct {
	// RefAudio is base64 reference audio for one-off voice cloning. It takes
	// precedence over the voice.
	RefAudio *string `json:"refAudio,omitempty"`
}

func readSpeechOptions(options model.ProviderOptions) (SpeechOptions, error) {
	speechOptions, _, err := model.ProviderValue[SpeechOptions](options, Namespace)
	if err != nil {
		return SpeechOptions{}, err
	}

	if speechOptions.RefAudio != nil && *speechOptions.RefAudio == "" {
		return SpeechOptions{}, fmt.Errorf("%w: mistral refAudio must not be empty", model.ErrInvalidArgument)
	}

	return speechOptions, nil
}

// TimestampGranularity selects the timestamps of a transcription.
type TimestampGranularity string

// Timestamp granularities.
const (
	TimestampSegment TimestampGranularity = "segment"
	TimestampWord    TimestampGranularity = "word"
)

// maxContextBiasItems is the limit of TranscriptionOptions.ContextBias.
const maxContextBiasItems = 100

// contextBiasItem matches one context bias term: no commas or whitespace.
var contextBiasItem = regexp.MustCompile(`^[^\s,]+$`)

// TranscriptionOptions are the provider options of transcription models.
type TranscriptionOptions struct {
	// Language of the audio, such as "en". It cannot be combined with
	// TimestampGranularities.
	Language *string `json:"language,omitempty"`

	// Temperature is the sampling temperature.
	Temperature *float64 `json:"temperature,omitempty"`

	// TimestampGranularities selects the timestamps to include.
	TimestampGranularities []TimestampGranularity `json:"timestampGranularities,omitempty"`

	// Diarize identifies speakers.
	Diarize *bool `json:"diarize,omitempty"`

	// ContextBias lists up to 100 terms that guide spelling. Terms must not
	// contain commas or whitespace; use underscores for phrases.
	ContextBias []string `json:"contextBias,omitempty"`
}

func readTranscriptionOptions(options model.ProviderOptions) (TranscriptionOptions, error) {
	transcriptionOptions, _, err := model.ProviderValue[TranscriptionOptions](options, Namespace)
	if err != nil {
		return TranscriptionOptions{}, err
	}

	if err := transcriptionOptions.validate(); err != nil {
		return TranscriptionOptions{}, err
	}

	return transcriptionOptions, nil
}

func (o TranscriptionOptions) validate() error {
	if o.Language != nil && *o.Language == "" {
		return fmt.Errorf("%w: mistral language must not be empty", model.ErrInvalidArgument)
	}

	if o.TimestampGranularities != nil && len(o.TimestampGranularities) == 0 {
		return fmt.Errorf("%w: mistral timestampGranularities must not be empty", model.ErrInvalidArgument)
	}

	for _, granularity := range o.TimestampGranularities {
		if granularity != TimestampSegment && granularity != TimestampWord {
			return fmt.Errorf("%w: mistral timestamp granularity %q", model.ErrInvalidArgument, granularity)
		}
	}

	if len(o.ContextBias) > maxContextBiasItems {
		return fmt.Errorf("%w: mistral contextBias accepts at most %d items", model.ErrInvalidArgument, maxContextBiasItems)
	}

	invalid := slices.IndexFunc(o.ContextBias, func(item string) bool {
		return !contextBiasItem.MatchString(item)
	})
	if invalid >= 0 {
		return fmt.Errorf("%w: context bias items must not contain commas or whitespace", model.ErrInvalidArgument)
	}

	// Mistral documents these options as incompatible. Rejecting them here
	// gives a stable error instead of an API failure.
	if o.Language != nil && o.TimestampGranularities != nil {
		return fmt.Errorf(
			"%w: providerOptions.mistral.language cannot be combined with providerOptions.mistral.timestampGranularities",
			model.ErrInvalidArgument,
		)
	}

	return nil
}
