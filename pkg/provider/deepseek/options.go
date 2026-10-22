package deepseek

import (
	"fmt"
	"regexp"

	"github.com/aarock1234/ai/pkg/model"
)

// ThinkingType enables or disables thinking.
type ThinkingType string

// Thinking types. ThinkingAdaptive is accepted for compatibility and sent as
// ThinkingEnabled.
const (
	ThinkingEnabled  ThinkingType = "enabled"
	ThinkingDisabled ThinkingType = "disabled"
	ThinkingAdaptive ThinkingType = "adaptive"
)

// ReasoningEffort is a DeepSeek thinking strength. ReasoningEffortMedium and
// ReasoningEffortXHigh are accepted for compatibility and mapped to high and
// max.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
	ReasoningEffortXHigh  ReasoningEffort = "xhigh"
	ReasoningEffortMax    ReasoningEffort = "max"
)

// Limits of the chat options.
const (
	maxTopLogprobs = 20
	maxUserIDLen   = 512
)

// userIDPattern matches valid user IDs.
var userIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// ChatOptions are the provider options of chat models, read from the
// "deepseek" namespace.
type ChatOptions struct {
	// Logprobs returns the log probabilities of the output tokens.
	Logprobs *bool `json:"logprobs,omitempty"`

	// TopLogprobs is the number, from 0 to 20, of most likely tokens to
	// return at each position. It enables Logprobs.
	TopLogprobs *int `json:"topLogprobs,omitempty"`

	// UserID is an opaque end-user identifier of at most 512 ASCII letters,
	// digits, underscores and hyphens.
	UserID *string `json:"userId,omitempty"`

	// Thinking enables or disables thinking, which is enabled by default.
	Thinking *Thinking `json:"thinking,omitempty"`

	// ReasoningEffort controls the thinking strength of V4 reasoning models.
	// It takes precedence over model.CallOptions.Reasoning.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty"`

	// StrictJSONSchema requests strict JSON schema validation on endpoints
	// that support JSON schema response formats. It defaults to true.
	StrictJSONSchema *bool `json:"strictJsonSchema,omitempty"`
}

// Thinking configures thinking.
type Thinking struct {
	Type ThinkingType `json:"type,omitempty"`
}

// readChatOptions reads and validates the call's DeepSeek options.
func readChatOptions(options model.ProviderOptions, namespace string) (ChatOptions, error) {
	chatOptions, _, err := model.ProviderValue[ChatOptions](options, namespace)
	if err != nil {
		return ChatOptions{}, err
	}

	if err := chatOptions.validate(); err != nil {
		return ChatOptions{}, err
	}

	return chatOptions, nil
}

func (o ChatOptions) validate() error {
	if o.TopLogprobs != nil && (*o.TopLogprobs < 0 || *o.TopLogprobs > maxTopLogprobs) {
		return fmt.Errorf("%w: deepseek topLogprobs must be between 0 and %d", model.ErrInvalidArgument, maxTopLogprobs)
	}

	if o.UserID != nil {
		if !userIDPattern.MatchString(*o.UserID) {
			return fmt.Errorf("%w: deepseek userId must match /^[a-zA-Z0-9_-]+$/", model.ErrInvalidArgument)
		}

		if len(*o.UserID) > maxUserIDLen {
			return fmt.Errorf("%w: deepseek userId must be at most %d characters long", model.ErrInvalidArgument, maxUserIDLen)
		}
	}

	if o.Thinking != nil {
		switch o.Thinking.Type {
		case "", ThinkingEnabled, ThinkingDisabled, ThinkingAdaptive:
		default:
			return fmt.Errorf("%w: deepseek thinking type %q", model.ErrInvalidArgument, o.Thinking.Type)
		}
	}

	switch o.ReasoningEffort {
	case "", ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax:
		return nil
	default:
		return fmt.Errorf("%w: deepseek reasoningEffort %q", model.ErrInvalidArgument, o.ReasoningEffort)
	}
}

// MessageOptions are the provider options of system, user and tool
// messages, read from the "deepseek" namespace.
type MessageOptions struct {
	// Name is the name of the participant represented by the message. Tool
	// messages do not support it.
	Name *string `json:"name,omitempty"`
}

// AssistantMessageOptions are the provider options of assistant messages,
// read from the "deepseek" namespace.
type AssistantMessageOptions struct {
	// Name is the name of the participant represented by the message.
	Name *string `json:"name,omitempty"`

	// Prefix marks the final assistant message as a prefix that DeepSeek
	// continues. It must be true when set and requires a "/beta" base URL.
	Prefix *bool `json:"prefix,omitempty"`
}

// readMessageOptions reads and validates the options of one prompt message.
// Every role is read as assistant options, so that a prefix on another role
// can be rejected.
func readMessageOptions(options model.ProviderOptions, namespace string) (AssistantMessageOptions, error) {
	messageOptions, _, err := model.ProviderValue[AssistantMessageOptions](options, namespace)
	if err != nil {
		return AssistantMessageOptions{}, err
	}

	if messageOptions.Prefix != nil && !*messageOptions.Prefix {
		return AssistantMessageOptions{}, fmt.Errorf("%w: deepseek prefix must be true when set", model.ErrInvalidArgument)
	}

	return messageOptions, nil
}

// hasPrefix reports whether the message is an assistant prefix.
func (o AssistantMessageOptions) hasPrefix() bool {
	return o.Prefix != nil && *o.Prefix
}

// ImageDetail controls how DeepSeek processes an image.
type ImageDetail string

// Image detail levels.
const (
	ImageDetailLow      ImageDetail = "low"
	ImageDetailHigh     ImageDetail = "high"
	ImageDetailOriginal ImageDetail = "original"
	ImageDetailAuto     ImageDetail = "auto"
)

// FilePartOptions are the provider options of image file parts, read from
// the "deepseek" namespace.
type FilePartOptions struct {
	// ImageDetail controls how DeepSeek processes the image.
	ImageDetail ImageDetail `json:"imageDetail,omitempty"`

	// FileData sends inline image data as a file part with file_data, which
	// keeps the filename, instead of an image_url data URL. It must be true
	// when set, applies only to inline data and cannot be combined with
	// ImageDetail.
	FileData *bool `json:"fileData,omitempty"`
}

// readFilePartOptions reads and validates the options of one file part.
func readFilePartOptions(options model.ProviderOptions, namespace string) (FilePartOptions, error) {
	partOptions, _, err := model.ProviderValue[FilePartOptions](options, namespace)
	if err != nil {
		return FilePartOptions{}, err
	}

	switch partOptions.ImageDetail {
	case "", ImageDetailLow, ImageDetailHigh, ImageDetailOriginal, ImageDetailAuto:
	default:
		return FilePartOptions{}, fmt.Errorf("%w: deepseek imageDetail %q", model.ErrInvalidArgument, partOptions.ImageDetail)
	}

	if partOptions.FileData != nil && !*partOptions.FileData {
		return FilePartOptions{}, fmt.Errorf("%w: deepseek fileData must be true when set", model.ErrInvalidArgument)
	}

	return partOptions, nil
}

// sendsFileData reports whether inline data is sent as a file part.
func (o FilePartOptions) sendsFileData() bool {
	return o.FileData != nil && *o.FileData
}

// File expiry limits, in seconds.
const (
	minFileExpiry = 3600
	maxFileExpiry = 2592000
)

// FilesOptions are the provider options of file uploads, read from the
// "deepseek" namespace.
type FilesOptions struct {
	// ExpiresAfter is the number of seconds, from one hour to 30 days, after
	// creation before the file expires.
	ExpiresAfter *int `json:"expiresAfter,omitempty"`
}

// readFilesOptions reads and validates the options of one upload.
func readFilesOptions(options model.ProviderOptions) (FilesOptions, error) {
	filesOptions, _, err := model.ProviderValue[FilesOptions](options, Namespace)
	if err != nil {
		return FilesOptions{}, err
	}

	expiry := filesOptions.ExpiresAfter
	if expiry != nil && (*expiry < minFileExpiry || *expiry > maxFileExpiry) {
		return FilesOptions{}, fmt.Errorf(
			"%w: deepseek expiresAfter must be between %d and %d seconds",
			model.ErrInvalidArgument,
			minFileExpiry,
			maxFileExpiry,
		)
	}

	return filesOptions, nil
}
