package fireworks

import (
	"fmt"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// ServiceTier selects the Fireworks serving path.
type ServiceTier string

// ServiceTierPriority uses the priority serving path, for higher reliability
// during peak traffic.
const ServiceTierPriority ServiceTier = "priority"

// ThinkingType enables or disables thinking.
type ThinkingType string

// Thinking types.
const (
	ThinkingEnabled  ThinkingType = "enabled"
	ThinkingDisabled ThinkingType = "disabled"
)

// ReasoningHistory selects how reasoning from earlier turns is kept.
type ReasoningHistory string

// Reasoning history modes.
const (
	ReasoningHistoryDisabled    ReasoningHistory = "disabled"
	ReasoningHistoryInterleaved ReasoningHistory = "interleaved"
	ReasoningHistoryPreserved   ReasoningHistory = "preserved"
)

// minThinkingBudget is the smallest thinking budget Fireworks accepts.
const minThinkingBudget = 1024

// ChatOptions are the provider options of chat models, read from the
// "fireworks" namespace.
type ChatOptions struct {
	// PromptCacheKey routes requests with shared prompt prefixes to the same
	// prompt cache.
	PromptCacheKey *string `json:"promptCacheKey,omitempty"`

	// ServiceTier selects the serving path.
	ServiceTier ServiceTier `json:"serviceTier,omitempty"`

	// Thinking configures thinking.
	Thinking *Thinking `json:"thinking,omitempty"`

	// ReasoningHistory selects how reasoning from earlier turns is kept.
	ReasoningHistory ReasoningHistory `json:"reasoningHistory,omitempty"`
}

// Thinking configures thinking.
type Thinking struct {
	// Type enables or disables thinking.
	Type ThinkingType `json:"type,omitempty"`

	// BudgetTokens is the thinking budget, at least 1024 tokens.
	BudgetTokens *int `json:"budgetTokens,omitempty"`
}

// readChatOptions reads and validates the call's Fireworks options.
func readChatOptions(options model.ProviderOptions) (ChatOptions, error) {
	chatOptions, _, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return ChatOptions{}, err
	}

	if err := chatOptions.validate(); err != nil {
		return ChatOptions{}, err
	}

	return chatOptions, nil
}

func (o ChatOptions) validate() error {
	if o.ServiceTier != "" && o.ServiceTier != ServiceTierPriority {
		return fmt.Errorf("%w: fireworks serviceTier %q", model.ErrInvalidArgument, o.ServiceTier)
	}

	switch o.ReasoningHistory {
	case "", ReasoningHistoryDisabled, ReasoningHistoryInterleaved, ReasoningHistoryPreserved:
	default:
		return fmt.Errorf("%w: fireworks reasoningHistory %q", model.ErrInvalidArgument, o.ReasoningHistory)
	}

	if o.Thinking == nil {
		return nil
	}

	switch o.Thinking.Type {
	case "", ThinkingEnabled, ThinkingDisabled:
	default:
		return fmt.Errorf("%w: fireworks thinking type %q", model.ErrInvalidArgument, o.Thinking.Type)
	}

	if o.Thinking.BudgetTokens != nil && *o.Thinking.BudgetTokens < minThinkingBudget {
		return fmt.Errorf("%w: fireworks thinking budgetTokens must be at least %d", model.ErrInvalidArgument, minThinkingBudget)
	}

	return nil
}

// OutputFormat is the file format of generated images.
type OutputFormat string

// Image output formats.
const (
	OutputFormatJPEG OutputFormat = "jpeg"
	OutputFormatPNG  OutputFormat = "png"
)

// maxSafetyTolerance is the most permissive moderation level.
const maxSafetyTolerance = 6

// ImageOptions are the common provider options of image models, read from
// the "fireworks" namespace. Every key of the namespace, including keys not
// listed here, is sent in the request unchanged, so the JSON names follow the
// wire format.
type ImageOptions struct {
	// GuidanceScale is the classifier-free guidance scale.
	GuidanceScale *float64 `json:"guidance_scale,omitempty"`

	// NumInferenceSteps is the number of denoising steps.
	NumInferenceSteps *int `json:"num_inference_steps,omitempty"`

	// OutputFormat is the file format of the generated image.
	OutputFormat OutputFormat `json:"output_format,omitempty"`

	// WebhookURL receives notifications for asynchronous Kontext requests,
	// signed with WebhookSecret.
	WebhookURL    *string `json:"webhook_url,omitempty"`
	WebhookSecret *string `json:"webhook_secret,omitempty"`

	// PromptUpsampling lets Fireworks rewrite the prompt for more creative
	// generation.
	PromptUpsampling *bool `json:"prompt_upsampling,omitempty"`

	// SafetyTolerance is the moderation tolerance, from 0 to 6.
	SafetyTolerance *int `json:"safety_tolerance,omitempty"`

	// CfgScale and Steps are the guidance scale and step count of legacy
	// image generation models.
	CfgScale *float64 `json:"cfg_scale,omitempty"`
	Steps    *int     `json:"steps,omitempty"`
}

// readImageOptions validates the call's Fireworks image options and returns
// every member of the namespace, in order, for the request.
func readImageOptions(options model.ProviderOptions) (jsonobject.Object, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options, Namespace)
	if err != nil {
		return nil, err
	}

	switch imageOptions.OutputFormat {
	case "", OutputFormatJPEG, OutputFormatPNG:
	default:
		return nil, fmt.Errorf("%w: fireworks output_format %q", model.ErrInvalidArgument, imageOptions.OutputFormat)
	}

	tolerance := imageOptions.SafetyTolerance
	if tolerance != nil && (*tolerance < 0 || *tolerance > maxSafetyTolerance) {
		return nil, fmt.Errorf("%w: fireworks safety_tolerance must be between 0 and %d", model.ErrInvalidArgument, maxSafetyTolerance)
	}

	members, _, err := model.ProviderValue[jsonobject.Object](options, Namespace)
	if err != nil {
		return nil, err
	}

	return members, nil
}
