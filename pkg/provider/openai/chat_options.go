package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"unicode/utf8"

	"github.com/aarock1234/ai/pkg/model"
)

// ChatOptions are the Chat Completions settings read from the "openai"
// provider options namespace.
type ChatOptions struct {
	// LogitBias maps token IDs to a bias from -100 to 100 that is added to
	// the token's logit before sampling.
	LogitBias map[int]float64 `json:"logitBias,omitempty"`

	// Logprobs requests the log probabilities of the generated tokens.
	Logprobs *LogprobsSetting `json:"logprobs,omitempty"`

	// ParallelToolCalls enables parallel function calling. The API default
	// is true.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`

	// User identifies the end user for abuse monitoring.
	User string `json:"user,omitempty"`

	// ReasoningEffort sets the effort of reasoning models. It takes
	// precedence over CallOptions.Reasoning. "provider-default" is not
	// accepted.
	ReasoningEffort model.ReasoningEffort `json:"reasoningEffort,omitempty"`

	// MaxCompletionTokens limits generated tokens, including reasoning
	// tokens.
	MaxCompletionTokens *int `json:"maxCompletionTokens,omitempty"`

	// Store enables persistence of the completion.
	Store *bool `json:"store,omitempty"`

	// Metadata is attached to the stored completion. Keys are at most 64
	// characters and values at most 512 characters.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Prediction is the JSON object of predicted output parameters, such as
	// {"type": "content", "content": "..."}.
	Prediction json.RawMessage `json:"prediction,omitempty"`

	// ServiceTier selects the processing tier. The API default is "auto".
	ServiceTier ChatServiceTier `json:"serviceTier,omitempty"`

	// StrictJSONSchema enables strict JSON schema validation of structured
	// output. The default is true.
	StrictJSONSchema *bool `json:"strictJsonSchema,omitempty"`

	// TextVerbosity controls the length of responses.
	TextVerbosity ChatTextVerbosity `json:"textVerbosity,omitempty"`

	// PromptCacheKey groups requests for prompt caching.
	PromptCacheKey string `json:"promptCacheKey,omitempty"`

	// PromptCacheOptions configures prompt caching for GPT-5.6 and later
	// models.
	PromptCacheOptions *ChatPromptCacheOptions `json:"promptCacheOptions,omitempty"`

	// PromptCacheRetention sets the prompt cache retention policy of models
	// before GPT-5.6. The API default is "in_memory".
	//
	// Deprecated: Use PromptCacheOptions.TTL with GPT-5.6 and later models.
	PromptCacheRetention ChatPromptCacheRetention `json:"promptCacheRetention,omitempty"`

	// SafetyIdentifier is a stable, preferably hashed, identifier of the end
	// user for policy enforcement.
	SafetyIdentifier string `json:"safetyIdentifier,omitempty"`

	// SystemMessageMode overrides how system messages are sent. The default
	// depends on the model.
	SystemMessageMode ChatSystemMessageMode `json:"systemMessageMode,omitempty"`

	// ForceReasoning treats the model as a reasoning model even when its ID
	// is not recognized, for example behind a custom base URL.
	ForceReasoning *bool `json:"forceReasoning,omitempty"`
}

// ChatServiceTier selects how a request is processed.
type ChatServiceTier string

// Chat service tiers.
const (
	ChatServiceTierAuto      ChatServiceTier = "auto"
	ChatServiceTierFlex      ChatServiceTier = "flex"
	ChatServiceTierPriority  ChatServiceTier = "priority"
	ChatServiceTierFast      ChatServiceTier = "fast"
	ChatServiceTierUltrafast ChatServiceTier = "ultrafast"
	ChatServiceTierDefault   ChatServiceTier = "default"
)

// ChatTextVerbosity controls response length.
type ChatTextVerbosity string

// Chat text verbosity levels.
const (
	ChatTextVerbosityLow    ChatTextVerbosity = "low"
	ChatTextVerbosityMedium ChatTextVerbosity = "medium"
	ChatTextVerbosityHigh   ChatTextVerbosity = "high"
)

// ChatPromptCacheOptions configures prompt caching.
type ChatPromptCacheOptions struct {
	// Mode controls whether OpenAI also places an implicit breakpoint.
	Mode ChatPromptCacheMode `json:"mode,omitempty"`

	// TTL is the minimum cache lifetime. Only "30m" is supported.
	TTL string `json:"ttl,omitempty"`
}

// ChatPromptCacheMode selects implicit or explicit cache breakpoints.
type ChatPromptCacheMode string

// Chat prompt cache modes.
const (
	ChatPromptCacheImplicit ChatPromptCacheMode = "implicit"
	ChatPromptCacheExplicit ChatPromptCacheMode = "explicit"
)

// ChatPromptCacheRetention is a prompt cache retention policy.
type ChatPromptCacheRetention string

// Chat prompt cache retention policies.
const (
	ChatPromptCacheInMemory ChatPromptCacheRetention = "in_memory"
	ChatPromptCache24h      ChatPromptCacheRetention = "24h"
)

// ChatSystemMessageMode selects how system messages are sent.
type ChatSystemMessageMode string

// Chat system message modes.
const (
	// ChatSystemMessageSystem sends system messages with the "system" role.
	ChatSystemMessageSystem ChatSystemMessageMode = "system"

	// ChatSystemMessageDeveloper sends system messages with the "developer"
	// role, as reasoning models expect.
	ChatSystemMessageDeveloper ChatSystemMessageMode = "developer"

	// ChatSystemMessageRemove drops system messages with a warning.
	ChatSystemMessageRemove ChatSystemMessageMode = "remove"
)

// ChatPartOptions are the "openai" provider options of a prompt message,
// part or tool result output.
type ChatPartOptions struct {
	// ImageDetail sets the detail level of an image file part, such as
	// "low", "high" or "auto".
	ImageDetail string `json:"imageDetail,omitempty"`

	// PromptCacheBreakpoint places an explicit prompt cache breakpoint after
	// the content.
	PromptCacheBreakpoint *ChatPromptCacheBreakpoint `json:"promptCacheBreakpoint,omitempty"`
}

// ChatPromptCacheBreakpoint marks an explicit prompt cache breakpoint.
type ChatPromptCacheBreakpoint struct {
	// Mode is "explicit".
	Mode ChatPromptCacheMode `json:"mode"`
}

// ChatMetadata is the "openai" provider metadata of a Chat Completions
// result.
type ChatMetadata struct {
	// AcceptedPredictionTokens counts predicted tokens that appeared in the
	// completion.
	AcceptedPredictionTokens *int `json:"acceptedPredictionTokens,omitempty"`

	// RejectedPredictionTokens counts predicted tokens that did not appear
	// in the completion.
	RejectedPredictionTokens *int `json:"rejectedPredictionTokens,omitempty"`

	// Logprobs holds the token log probabilities, when requested.
	Logprobs []ChatLogprob `json:"logprobs,omitempty"`
}

// ChatLogprob is the log probability of one generated token.
type ChatLogprob struct {
	Token       string           `json:"token"`
	Logprob     float64          `json:"logprob"`
	TopLogprobs []ChatTopLogprob `json:"top_logprobs"`
}

// ChatTopLogprob is the log probability of one alternative token.
type ChatTopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
}

// LogprobsSetting requests token log probabilities. It encodes as true when
// only the generated tokens are requested, or as the number of top
// alternatives per token.
type LogprobsSetting struct {
	// Enabled requests log probabilities.
	Enabled bool

	// Top is the number of most likely alternatives to return per token.
	// Zero returns only the generated tokens.
	Top int
}

// MarshalJSON encodes the setting as a boolean or a number.
func (s LogprobsSetting) MarshalJSON() ([]byte, error) {
	if s.Top > 0 {
		return json.Marshal(s.Top)
	}

	return json.Marshal(s.Enabled)
}

// UnmarshalJSON accepts a boolean or a number.
func (s *LogprobsSetting) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)

	var enabled bool
	if err := json.Unmarshal(trimmed, &enabled); err == nil {
		*s = LogprobsSetting{
			Enabled: enabled,
		}

		return nil
	}

	var top int
	if err := json.Unmarshal(trimmed, &top); err != nil {
		return fmt.Errorf("logprobs must be a boolean or an integer: %w", err)
	}

	*s = LogprobsSetting{
		Enabled: true,
		Top:     top,
	}

	return nil
}

// requested reports whether log probabilities are requested.
func (s *LogprobsSetting) requested() bool {
	return s != nil && (s.Enabled || s.Top > 0)
}

var chatReasoningEfforts = []model.ReasoningEffort{
	model.ReasoningNone,
	model.ReasoningMinimal,
	model.ReasoningLow,
	model.ReasoningMedium,
	model.ReasoningHigh,
	model.ReasoningXHigh,
	model.ReasoningMax,
}

// readChatOptions reads and validates the Chat Completions provider options.
func readChatOptions(options model.ProviderOptions) (ChatOptions, error) {
	values, _, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return ChatOptions{}, err
	}

	if err := values.validate(); err != nil {
		return ChatOptions{}, fmt.Errorf("%w: openai chat options: %w", model.ErrInvalidArgument, err)
	}

	return values, nil
}

func (o ChatOptions) validate() error {
	if o.ReasoningEffort != "" && !slices.Contains(chatReasoningEfforts, o.ReasoningEffort) {
		return fmt.Errorf("unknown reasoningEffort %q", o.ReasoningEffort)
	}

	if err := o.validateEnums(); err != nil {
		return err
	}

	if err := o.validateMetadata(); err != nil {
		return err
	}

	return o.validatePrediction()
}

func (o ChatOptions) validateEnums() error {
	switch o.ServiceTier {
	case "",
		ChatServiceTierAuto,
		ChatServiceTierFlex,
		ChatServiceTierPriority,
		ChatServiceTierFast,
		ChatServiceTierUltrafast,
		ChatServiceTierDefault:
	default:
		return fmt.Errorf("unknown serviceTier %q", o.ServiceTier)
	}

	switch o.TextVerbosity {
	case "", ChatTextVerbosityLow, ChatTextVerbosityMedium, ChatTextVerbosityHigh:
	default:
		return fmt.Errorf("unknown textVerbosity %q", o.TextVerbosity)
	}

	switch o.PromptCacheRetention {
	case "", ChatPromptCacheInMemory, ChatPromptCache24h:
	default:
		return fmt.Errorf("unknown promptCacheRetention %q", o.PromptCacheRetention)
	}

	switch o.SystemMessageMode {
	case "", ChatSystemMessageSystem, ChatSystemMessageDeveloper, ChatSystemMessageRemove:
	default:
		return fmt.Errorf("unknown systemMessageMode %q", o.SystemMessageMode)
	}

	return o.PromptCacheOptions.validate()
}

func (o *ChatPromptCacheOptions) validate() error {
	if o == nil {
		return nil
	}

	switch o.Mode {
	case "", ChatPromptCacheImplicit, ChatPromptCacheExplicit:
	default:
		return fmt.Errorf("unknown promptCacheOptions.mode %q", o.Mode)
	}

	if o.TTL != "" && o.TTL != "30m" {
		return fmt.Errorf("promptCacheOptions.ttl must be \"30m\", got %q", o.TTL)
	}

	return nil
}

func (o ChatOptions) validateMetadata() error {
	for key, value := range o.Metadata {
		if utf8.RuneCountInString(key) > 64 {
			return fmt.Errorf("metadata key %q exceeds 64 characters", key)
		}

		if utf8.RuneCountInString(value) > 512 {
			return fmt.Errorf("metadata value of %q exceeds 512 characters", key)
		}
	}

	return nil
}

func (o ChatOptions) validatePrediction() error {
	if len(o.Prediction) == 0 {
		return nil
	}

	if !chatIsJSONObject(o.Prediction) {
		return fmt.Errorf("prediction must be a JSON object")
	}

	return nil
}

// chatIsJSONObject reports whether raw is a valid JSON object.
func chatIsJSONObject(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return false
	}

	return json.Valid(trimmed)
}

// readChatPartOptions reads the "openai" options of a message, part or
// output.
func readChatPartOptions(options model.ProviderOptions) (ChatPartOptions, error) {
	values, _, err := model.ProviderValue[ChatPartOptions](options, Namespace)

	return values, err
}

// chatResponsesOnlyOptions holds options that only the Responses API accepts,
// so that Chat Completions can warn instead of silently ignoring them.
type chatResponsesOnlyOptions struct {
	ReasoningSummary json.RawMessage `json:"reasoningSummary"`
}

// chatHasReasoningSummary reports whether the call sets reasoningSummary.
func chatHasReasoningSummary(options model.ProviderOptions) bool {
	values, _, err := model.ProviderValue[chatResponsesOnlyOptions](options, Namespace)
	if err != nil {
		return false
	}

	summary := bytes.TrimSpace(values.ReasoningSummary)

	return len(summary) > 0 && !bytes.Equal(summary, []byte("null"))
}
