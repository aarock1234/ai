package zai

import (
	"fmt"
	"slices"
	"unicode/utf8"

	"github.com/aarock1234/ai/pkg/model"
)

// ThinkingType enables or disables model thinking.
type ThinkingType string

// Thinking types.
const (
	ThinkingEnabled  ThinkingType = "enabled"
	ThinkingDisabled ThinkingType = "disabled"
)

// ReasoningEffort is the reasoning effort of GLM-5.2 and later models.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortMax     ReasoningEffort = "max"
)

// reasoningEfforts are the accepted reasoning efforts.
var reasoningEfforts = []ReasoningEffort{
	ReasoningEffortNone,
	ReasoningEffortMinimal,
	ReasoningEffortLow,
	ReasoningEffortMedium,
	ReasoningEffortHigh,
	ReasoningEffortXHigh,
	ReasoningEffortMax,
}

// Identifier length limits.
const (
	minRequestIDLength = 6
	maxRequestIDLength = 64
	minUserIDLength    = 6
	maxUserIDLength    = 128
)

// ChatOptions are the provider options of chat models, read from the "zai"
// namespace. Unknown keys are ignored.
type ChatOptions struct {
	// DoSample enables or disables sampling. When disabled, temperature and
	// topP do not take effect.
	DoSample *bool `json:"doSample,omitempty"`

	// Thinking controls model thinking and whether reasoning from earlier
	// turns is kept.
	Thinking *Thinking `json:"thinking,omitempty"`

	// ReasoningEffort controls the reasoning effort of GLM-5.2 and later
	// models.
	ReasoningEffort *ReasoningEffort `json:"reasoningEffort,omitempty"`

	// ToolStream enables incremental function call argument streaming on
	// supported models.
	ToolStream *bool `json:"toolStream,omitempty"`

	// RequestID is a caller-provided request identifier between 6 and 64
	// characters.
	RequestID *string `json:"requestId,omitempty"`

	// UserID is a non-sensitive end-user identifier between 6 and 128
	// characters.
	UserID *string `json:"userId,omitempty"`
}

// Thinking configures model thinking.
type Thinking struct {
	// Type enables or disables thinking.
	Type *ThinkingType `json:"type,omitempty"`

	// ClearThinking drops reasoning from earlier turns.
	ClearThinking *bool `json:"clearThinking,omitempty"`
}

// readChatOptions decodes and validates the "zai" provider options.
func readChatOptions(options model.ProviderOptions) (*ChatOptions, error) {
	chatOptions, ok, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return nil, fmt.Errorf("invalid zai provider options: %w", err)
	}

	if !ok {
		return nil, nil
	}

	if err := chatOptions.validate(); err != nil {
		return nil, fmt.Errorf("%w: invalid zai provider options: %w", model.ErrInvalidArgument, err)
	}

	return &chatOptions, nil
}

func (o ChatOptions) validate() error {
	if o.Thinking != nil && o.Thinking.Type != nil {
		thinkingType := *o.Thinking.Type
		if thinkingType != ThinkingEnabled && thinkingType != ThinkingDisabled {
			return fmt.Errorf("thinking.type must be %q or %q, got %q", ThinkingEnabled, ThinkingDisabled, thinkingType)
		}
	}

	if o.ReasoningEffort != nil && !slices.Contains(reasoningEfforts, *o.ReasoningEffort) {
		return fmt.Errorf("unsupported reasoningEffort %q", *o.ReasoningEffort)
	}

	if err := validateLength("requestId", o.RequestID, minRequestIDLength, maxRequestIDLength); err != nil {
		return err
	}

	return validateLength("userId", o.UserID, minUserIDLength, maxUserIDLength)
}

// validateLength checks that an optional string has between minimum and
// maximum characters.
func validateLength(field string, value *string, minimum int, maximum int) error {
	if value == nil {
		return nil
	}

	length := utf8.RuneCountInString(*value)
	if length < minimum || length > maximum {
		return fmt.Errorf("%s must have between %d and %d characters, got %d", field, minimum, maximum, length)
	}

	return nil
}
