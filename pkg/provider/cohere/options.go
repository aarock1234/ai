package cohere

import (
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// ThinkingType enables or disables reasoning.
type ThinkingType string

// Thinking types.
const (
	ThinkingEnabled  ThinkingType = "enabled"
	ThinkingDisabled ThinkingType = "disabled"
)

// ChatOptions are the provider options of chat models.
type ChatOptions struct {
	// Thinking configures reasoning. It takes precedence over
	// model.CallOptions.Reasoning.
	Thinking *ThinkingOptions `json:"thinking,omitempty"`
}

// ThinkingOptions configure reasoning.
type ThinkingOptions struct {
	// Type defaults to ThinkingEnabled.
	Type ThinkingType `json:"type,omitempty"`

	// TokenBudget limits the reasoning tokens. When the budget is reached,
	// the model stops thinking and answers.
	TokenBudget *int `json:"tokenBudget,omitempty"`
}

func readChatOptions(options model.ProviderOptions) (ChatOptions, error) {
	chatOptions, _, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return ChatOptions{}, err
	}

	if chatOptions.Thinking == nil {
		return chatOptions, nil
	}

	switch chatOptions.Thinking.Type {
	case "", ThinkingEnabled, ThinkingDisabled:
		return chatOptions, nil
	default:
		return ChatOptions{}, fmt.Errorf("%w: cohere thinking type %q", model.ErrInvalidArgument, chatOptions.Thinking.Type)
	}
}

// ImageDetail is the fidelity of an image input.
type ImageDetail string

// Image details.
const (
	ImageDetailAuto ImageDetail = "auto"
	ImageDetailLow  ImageDetail = "low"
	ImageDetailHigh ImageDetail = "high"
)

// ImagePartOptions are the provider options of image file parts.
type ImagePartOptions struct {
	// Detail is the image fidelity.
	Detail ImageDetail `json:"detail,omitempty"`
}

func readImagePartOptions(options model.ProviderOptions) (ImagePartOptions, error) {
	partOptions, _, err := model.ProviderValue[ImagePartOptions](options, Namespace)
	if err != nil {
		return ImagePartOptions{}, err
	}

	switch partOptions.Detail {
	case "", ImageDetailAuto, ImageDetailLow, ImageDetailHigh:
		return partOptions, nil
	default:
		return ImagePartOptions{}, fmt.Errorf("%w: cohere image detail %q", model.ErrInvalidArgument, partOptions.Detail)
	}
}

// EmbeddingType is the format of returned embeddings. Binary formats contain
// packed bytes rather than vector dimensions.
type EmbeddingType string

// Embedding types.
const (
	EmbeddingFloat   EmbeddingType = "float"
	EmbeddingInt8    EmbeddingType = "int8"
	EmbeddingUint8   EmbeddingType = "uint8"
	EmbeddingBinary  EmbeddingType = "binary"
	EmbeddingUbinary EmbeddingType = "ubinary"
)

// InputType describes what the embedded values are used for.
type InputType string

// Input types.
const (
	InputSearchDocument InputType = "search_document"
	InputSearchQuery    InputType = "search_query"
	InputClassification InputType = "classification"
	InputClustering     InputType = "clustering"
)

// Truncate selects how inputs above the token limit are shortened.
type Truncate string

// Truncation modes.
const (
	// TruncateNone rejects inputs above the limit.
	TruncateNone Truncate = "NONE"

	// TruncateStart discards the start of the input.
	TruncateStart Truncate = "START"

	// TruncateEnd discards the end of the input.
	TruncateEnd Truncate = "END"
)

// outputDimensions are the vector sizes Cohere models accept.
var outputDimensions = []int{
	256,
	512,
	768,
	1024,
	1536,
	2048,
}

// EmbeddingOptions are the provider options of embedding models.
type EmbeddingOptions struct {
	// EmbeddingType defaults to EmbeddingFloat. Other types require Embed
	// v3.0 or newer.
	EmbeddingType EmbeddingType `json:"embeddingType,omitempty"`

	// InputType defaults to InputSearchQuery.
	InputType InputType `json:"inputType,omitempty"`

	// Truncate defaults to TruncateEnd.
	Truncate Truncate `json:"truncate,omitempty"`

	// OutputDimension requests a vector size from Embed v4.0 and newer: 256,
	// 512, 768, 1024, 1536 or 2048. It takes precedence over
	// model.EmbedOptions.Dimensions.
	OutputDimension *int `json:"outputDimension,omitempty"`
}

func readEmbeddingOptions(options model.ProviderOptions) (EmbeddingOptions, error) {
	embeddingOptions, _, err := model.ProviderValue[EmbeddingOptions](options, Namespace)
	if err != nil {
		return EmbeddingOptions{}, err
	}

	if err := embeddingOptions.validate(); err != nil {
		return EmbeddingOptions{}, err
	}

	return embeddingOptions, nil
}

func (o EmbeddingOptions) validate() error {
	switch o.EmbeddingType {
	case "", EmbeddingFloat, EmbeddingInt8, EmbeddingUint8, EmbeddingBinary, EmbeddingUbinary:
	default:
		return fmt.Errorf("%w: cohere embeddingType %q", model.ErrInvalidArgument, o.EmbeddingType)
	}

	switch o.InputType {
	case "", InputSearchDocument, InputSearchQuery, InputClassification, InputClustering:
	default:
		return fmt.Errorf("%w: cohere inputType %q", model.ErrInvalidArgument, o.InputType)
	}

	switch o.Truncate {
	case "", TruncateNone, TruncateStart, TruncateEnd:
	default:
		return fmt.Errorf("%w: cohere truncate %q", model.ErrInvalidArgument, o.Truncate)
	}

	if o.OutputDimension != nil && !slices.Contains(outputDimensions, *o.OutputDimension) {
		return fmt.Errorf("%w: cohere outputDimension %d", model.ErrInvalidArgument, *o.OutputDimension)
	}

	return nil
}

// RerankingOptions are the provider options of reranking models.
type RerankingOptions struct {
	// MaxTokensPerDoc truncates long documents. It defaults to 4096.
	MaxTokensPerDoc *int `json:"maxTokensPerDoc,omitempty"`

	// Priority is the priority of the request. It defaults to 0.
	Priority *int `json:"priority,omitempty"`
}
