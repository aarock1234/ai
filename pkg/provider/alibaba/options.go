package alibaba

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// ChatOptions are the provider options of chat models, read from the
// "alibaba" namespace.
type ChatOptions struct {
	// EnableThinking enables reasoning on models that support it. It takes
	// precedence over model.CallOptions.Reasoning.
	EnableThinking *bool `json:"enableThinking,omitempty"`

	// ThinkingBudget is the most reasoning tokens to generate. It must be
	// positive.
	ThinkingBudget *int `json:"thinkingBudget,omitempty"`

	// PreserveThinking replays reasoning from earlier assistant turns. It
	// defaults to true on models that support preserved thinking.
	PreserveThinking *bool `json:"preserveThinking,omitempty"`

	// ParallelToolCalls allows several tool calls per response. It defaults
	// to true.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`
}

// readChatOptions reads and validates the call's Alibaba options.
func readChatOptions(options model.ProviderOptions) (ChatOptions, error) {
	chatOptions, _, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return ChatOptions{}, err
	}

	if chatOptions.ThinkingBudget != nil && *chatOptions.ThinkingBudget <= 0 {
		return ChatOptions{}, fmt.Errorf("%w: alibaba thinkingBudget must be positive", model.ErrInvalidArgument)
	}

	return chatOptions, nil
}

// CacheControl marks a prompt part as an explicit context cache
// breakpoint. It is read from the "cacheControl" or "cache_control" key of a
// message's or part's "alibaba" provider options, and Alibaba validates the
// value.
type CacheControl struct {
	// Type is the cache type, such as "ephemeral".
	Type string `json:"type"`
}

// cacheControlOptions are the provider options that carry a cache control,
// in either spelling.
type cacheControlOptions struct {
	CacheControl      *CacheControl `json:"cacheControl"`
	SnakeCacheControl *CacheControl `json:"cache_control"`
}

// TextType selects how text is embedded for asymmetric retrieval.
type TextType string

// Text types.
const (
	TextTypeQuery    TextType = "query"
	TextTypeDocument TextType = "document"
)

// EmbeddingOutputType selects the embedding vectors to return.
type EmbeddingOutputType string

// Embedding output types. Sparse-only output is not supported, because
// embeddings are dense vectors.
const (
	EmbeddingOutputDense          EmbeddingOutputType = "dense"
	EmbeddingOutputSparse         EmbeddingOutputType = "sparse"
	EmbeddingOutputDenseAndSparse EmbeddingOutputType = "dense&sparse"
)

// EmbeddingOptions are the provider options of embedding models, read from
// the "alibaba" namespace.
type EmbeddingOptions struct {
	// TextType distinguishes queries from documents. It defaults to
	// documents.
	TextType TextType `json:"textType,omitempty"`

	// Dimension is the vector size, 1024 by default. It takes precedence
	// over model.EmbedOptions.Dimensions.
	Dimension *int `json:"dimension,omitempty"`

	// OutputType selects dense vectors, or dense and sparse vectors, whose
	// sparse part is reported in the provider metadata.
	OutputType EmbeddingOutputType `json:"outputType,omitempty"`
}

// readEmbeddingOptions reads and validates the call's Alibaba options.
func readEmbeddingOptions(options model.ProviderOptions) (EmbeddingOptions, error) {
	embeddingOptions, _, err := model.ProviderValue[EmbeddingOptions](options, Namespace)
	if err != nil {
		return EmbeddingOptions{}, err
	}

	switch embeddingOptions.TextType {
	case "", TextTypeQuery, TextTypeDocument:
	default:
		return EmbeddingOptions{}, fmt.Errorf("%w: alibaba textType %q", model.ErrInvalidArgument, embeddingOptions.TextType)
	}

	switch embeddingOptions.OutputType {
	case "", EmbeddingOutputDense, EmbeddingOutputDenseAndSparse:
		return embeddingOptions, nil

	case EmbeddingOutputSparse:
		return EmbeddingOptions{}, fmt.Errorf(
			"%w: alibaba embedding outputType 'sparse' because embeddings require dense number arrays, use 'dense' or 'dense&sparse' instead",
			model.ErrUnsupported,
		)

	default:
		return EmbeddingOptions{}, fmt.Errorf("%w: alibaba outputType %q", model.ErrInvalidArgument, embeddingOptions.OutputType)
	}
}

// ShotType selects single-shot or multi-shot video on models before
// wan2.7.
type ShotType string

// Shot types.
const (
	ShotTypeSingle ShotType = "single"
	ShotTypeMulti  ShotType = "multi"
)

// VideoMediaType is the role of an explicit media item.
type VideoMediaType string

// Media item types. Reference audio, files and links are wan3 only.
const (
	VideoMediaReferenceImage VideoMediaType = "reference_image"
	VideoMediaReferenceVideo VideoMediaType = "reference_video"
	VideoMediaReferenceAudio VideoMediaType = "reference_audio"
	VideoMediaFirstFrame     VideoMediaType = "first_frame"
	VideoMediaLastFrame      VideoMediaType = "last_frame"
	VideoMediaFile           VideoMediaType = "file"
	VideoMediaLink           VideoMediaType = "link"
)

// VideoRatio is an output aspect ratio of wan2.7 text- and
// reference-to-video models and wan3 models.
type VideoRatio string

// Video aspect ratios. VideoRatioAdaptive is wan3 only and its default.
const (
	VideoRatioAdaptive VideoRatio = "adaptive"
	VideoRatio16x9     VideoRatio = "16:9"
	VideoRatio9x16     VideoRatio = "9:16"
	VideoRatio1x1      VideoRatio = "1:1"
	VideoRatio4x3      VideoRatio = "4:3"
	VideoRatio3x4      VideoRatio = "3:4"
)

// VideoOptions are the provider options of video models, read from the
// "alibaba" namespace.
type VideoOptions struct {
	// NegativePrompt describes what to avoid, in at most 500 characters.
	NegativePrompt *string `json:"negativePrompt,omitempty"`

	// AudioURL is a WAV or MP3 file of 3 to 30 seconds for audio-video sync.
	AudioURL *string `json:"audioUrl,omitempty"`

	// PromptExtend lets Alibaba rewrite the prompt. It defaults to true.
	PromptExtend *bool `json:"promptExtend,omitempty"`

	// ShotType selects single-shot or multi-shot video on models before
	// wan2.7, which describe the shots in the prompt instead.
	ShotType ShotType `json:"shotType,omitempty"`

	// Watermark adds a watermark. It defaults to false.
	Watermark *bool `json:"watermark,omitempty"`

	// Audio enables audio generation on wan2.6 image- and reference-to-video
	// models and wan3 models. wan2.7 models always generate audio.
	Audio *bool `json:"audio,omitempty"`

	// ReferenceURLs are up to 5 image and video URLs for wan2.6
	// reference-to-video models. Prompts refer to them as character1,
	// character2 and so on.
	ReferenceURLs []string `json:"referenceUrls,omitempty"`

	// Media is the explicit media of wan2.7 and wan3 models. It replaces the
	// media derived from the input references and frame images.
	Media []VideoMedia `json:"media,omitempty"`

	// Ratio is the aspect ratio of wan2.7 text- and reference-to-video models
	// and wan3 models.
	Ratio VideoRatio `json:"ratio,omitempty"`

	// PollIntervalMs and PollTimeoutMs are accepted for compatibility and
	// ignored; ai.VideoPollOptions controls polling.
	PollIntervalMs *float64 `json:"pollIntervalMs,omitempty"`
	PollTimeoutMs  *float64 `json:"pollTimeoutMs,omitempty"`
}

// VideoMedia is one explicit media item.
type VideoMedia struct {
	Type VideoMediaType `json:"type"`

	// URL is a public URL, or a data URI for images.
	URL string `json:"url"`

	// ReferenceVoice is an audio URL used as the voice of this item.
	ReferenceVoice *string `json:"referenceVoice,omitempty"`
}

// readVideoOptions reads and validates the call's Alibaba video options.
func readVideoOptions(options model.ProviderOptions) (VideoOptions, error) {
	videoOptions, _, err := model.ProviderValue[VideoOptions](options, Namespace)
	if err != nil {
		return VideoOptions{}, err
	}

	if err := videoOptions.validate(); err != nil {
		return VideoOptions{}, err
	}

	return videoOptions, nil
}

func (o VideoOptions) validate() error {
	switch o.ShotType {
	case "", ShotTypeSingle, ShotTypeMulti:
	default:
		return fmt.Errorf("%w: alibaba shotType %q", model.ErrInvalidArgument, o.ShotType)
	}

	switch o.Ratio {
	case "", VideoRatioAdaptive, VideoRatio16x9, VideoRatio9x16, VideoRatio1x1, VideoRatio4x3, VideoRatio3x4:
	default:
		return fmt.Errorf("%w: alibaba ratio %q", model.ErrInvalidArgument, o.Ratio)
	}

	for _, item := range o.Media {
		switch item.Type {
		case VideoMediaReferenceImage, VideoMediaReferenceVideo, VideoMediaReferenceAudio,
			VideoMediaFirstFrame, VideoMediaLastFrame, VideoMediaFile, VideoMediaLink:
		default:
			return fmt.Errorf("%w: alibaba media type %q", model.ErrInvalidArgument, item.Type)
		}
	}

	if o.PollIntervalMs != nil && *o.PollIntervalMs <= 0 {
		return fmt.Errorf("%w: alibaba pollIntervalMs must be positive", model.ErrInvalidArgument)
	}

	if o.PollTimeoutMs != nil && *o.PollTimeoutMs <= 0 {
		return fmt.Errorf("%w: alibaba pollTimeoutMs must be positive", model.ErrInvalidArgument)
	}

	return nil
}
