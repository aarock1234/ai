package cerebras

import (
	"bytes"
	"encoding/json"
	"fmt"
	"unicode/utf8"

	"github.com/aarock1234/ai/pkg/model"
)

// ServiceTier selects the Cerebras serving tier.
type ServiceTier string

// Service tiers.
const (
	ServiceTierAuto     ServiceTier = "auto"
	ServiceTierDefault  ServiceTier = "default"
	ServiceTierFlex     ServiceTier = "flex"
	ServiceTierPriority ServiceTier = "priority"
)

// ReasoningEffort is a Cerebras reasoning effort.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningEffortNone   ReasoningEffort = "none"
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
)

// ReasoningFormat selects how reasoning is returned.
type ReasoningFormat string

// Reasoning formats.
const (
	ReasoningFormatNone       ReasoningFormat = "none"
	ReasoningFormatParsed     ReasoningFormat = "parsed"
	ReasoningFormatTextParsed ReasoningFormat = "text_parsed"
	ReasoningFormatRaw        ReasoningFormat = "raw"
	ReasoningFormatHidden     ReasoningFormat = "hidden"
)

// Limits of the chat options.
const (
	maxTopLogprobs       = 20
	maxLogitBias         = 100
	maxPromptCacheKeyLen = 1024
)

// ChatOptions are the provider options of chat models, read from the
// "cerebras" namespace.
type ChatOptions struct {
	// User identifies the end user.
	User *string `json:"user,omitempty"`

	// StrictJSONSchema requests constrained decoding for JSON schema response
	// formats. It defaults to true.
	StrictJSONSchema *bool `json:"strictJsonSchema,omitempty"`

	// ParallelToolCalls allows several tool calls per response.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`

	// Logprobs returns the log probabilities of the output tokens.
	Logprobs *bool `json:"logprobs,omitempty"`

	// TopLogprobs is the number, from 0 to 20, of most likely tokens to
	// return at each position.
	TopLogprobs *int `json:"topLogprobs,omitempty"`

	// LogitBias maps token IDs to a bias from -100 to 100.
	LogitBias map[string]float64 `json:"logitBias,omitempty"`

	// ServiceTier selects the serving tier.
	ServiceTier ServiceTier `json:"serviceTier,omitempty"`

	// ReasoningEffort selects the reasoning effort of reasoning models. It
	// takes precedence over model.CallOptions.Reasoning.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty"`

	// ReasoningFormat selects how reasoning is returned.
	ReasoningFormat ReasoningFormat `json:"reasoningFormat,omitempty"`

	// Prediction is predicted output that speeds up generation when most of
	// the response is known in advance.
	Prediction *Prediction `json:"prediction,omitempty"`

	// PromptCacheKey routes requests with a shared prompt prefix to the same
	// cache. It has at most 1024 characters.
	PromptCacheKey *string `json:"promptCacheKey,omitempty"`
}

// Prediction is predicted output content.
type Prediction struct {
	// Type is always "content".
	Type string `json:"type"`

	Content PredictionContent `json:"content"`
}

// PredictionContent is predicted text, sent as a string or as text parts.
// Parts takes precedence when it is not nil.
type PredictionContent struct {
	Text  string
	Parts []PredictionPart
}

// PredictionPart is one text part of predicted content.
type PredictionPart struct {
	// Type is always "text".
	Type string `json:"type"`

	Text string `json:"text"`
}

// MarshalJSON encodes the parts when there are any and the text otherwise.
func (c PredictionContent) MarshalJSON() ([]byte, error) {
	if c.Parts != nil {
		return json.Marshal(c.Parts)
	}

	return json.Marshal(c.Text)
}

// UnmarshalJSON accepts a string or an array of text parts.
func (c *PredictionContent) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	*c = PredictionContent{}

	if bytes.HasPrefix(trimmed, []byte("[")) {
		return json.Unmarshal(trimmed, &c.Parts)
	}

	return json.Unmarshal(trimmed, &c.Text)
}

// readChatOptions reads and validates the call's Cerebras options.
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
	if o.TopLogprobs != nil && (*o.TopLogprobs < 0 || *o.TopLogprobs > maxTopLogprobs) {
		return fmt.Errorf("%w: cerebras topLogprobs must be between 0 and %d", model.ErrInvalidArgument, maxTopLogprobs)
	}

	for token, bias := range o.LogitBias {
		if bias < -maxLogitBias || bias > maxLogitBias {
			return fmt.Errorf("%w: cerebras logitBias for token %q must be between -100 and 100", model.ErrInvalidArgument, token)
		}
	}

	if err := o.validateEnums(); err != nil {
		return err
	}

	if o.Prediction != nil {
		if err := o.Prediction.validate(); err != nil {
			return err
		}
	}

	if o.PromptCacheKey != nil && utf8.RuneCountInString(*o.PromptCacheKey) > maxPromptCacheKeyLen {
		return fmt.Errorf("%w: cerebras promptCacheKey must have at most %d characters", model.ErrInvalidArgument, maxPromptCacheKeyLen)
	}

	return nil
}

func (o ChatOptions) validateEnums() error {
	switch o.ServiceTier {
	case "", ServiceTierAuto, ServiceTierDefault, ServiceTierFlex, ServiceTierPriority:
	default:
		return fmt.Errorf("%w: cerebras serviceTier %q", model.ErrInvalidArgument, o.ServiceTier)
	}

	switch o.ReasoningEffort {
	case "", ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh:
	default:
		return fmt.Errorf("%w: cerebras reasoningEffort %q", model.ErrInvalidArgument, o.ReasoningEffort)
	}

	switch o.ReasoningFormat {
	case "", ReasoningFormatNone, ReasoningFormatParsed, ReasoningFormatTextParsed, ReasoningFormatRaw, ReasoningFormatHidden:
		return nil
	default:
		return fmt.Errorf("%w: cerebras reasoningFormat %q", model.ErrInvalidArgument, o.ReasoningFormat)
	}
}

func (p Prediction) validate() error {
	if p.Type != "content" {
		return fmt.Errorf("%w: cerebras prediction type must be \"content\"", model.ErrInvalidArgument)
	}

	for _, part := range p.Content.Parts {
		if part.Type != "text" {
			return fmt.Errorf("%w: cerebras prediction content parts must have type \"text\"", model.ErrInvalidArgument)
		}
	}

	return nil
}
