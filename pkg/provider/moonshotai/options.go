package moonshotai

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// ReasoningEffort is the reasoning effort of Kimi K3.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningEffortLow  ReasoningEffort = "low"
	ReasoningEffortHigh ReasoningEffort = "high"
	ReasoningEffortMax  ReasoningEffort = "max"
)

// ThinkingType enables or disables thinking.
type ThinkingType string

// Thinking types.
const (
	ThinkingEnabled  ThinkingType = "enabled"
	ThinkingDisabled ThinkingType = "disabled"
)

// ReasoningHistory controls preserved reasoning in multi-turn
// conversations.
type ReasoningHistory string

// Reasoning history modes. Disabled and interleaved leave the request
// unchanged; preserved keeps all reasoning on Kimi K2.6. Kimi K2.7 and K3
// preserve reasoning by default.
const (
	ReasoningHistoryDisabled    ReasoningHistory = "disabled"
	ReasoningHistoryInterleaved ReasoningHistory = "interleaved"
	ReasoningHistoryPreserved   ReasoningHistory = "preserved"
)

// Option limits.
const (
	maxTopLogprobs    = 20
	minThinkingBudget = 1024
)

// Option types that the options must use.
const (
	predictionContent  = "content"
	predictionTextType = "text"
	dynamicToolType    = "function"
)

// ChatOptions are the provider options of chat models, read from the
// "moonshotai" namespace.
type ChatOptions struct {
	// StrictJSONSchema requests strict validation of JSON schema response
	// formats. Nil means true.
	StrictJSONSchema *bool `json:"strictJsonSchema,omitempty"`

	// Logprobs returns the log probabilities of the generated tokens.
	Logprobs *bool `json:"logprobs,omitempty"`

	// TopLogprobs is the number, from 0 to 20, of most likely tokens to
	// return at each position. Setting it enables Logprobs.
	TopLogprobs *int `json:"topLogprobs,omitempty"`

	// ReasoningEffort is the reasoning effort of Kimi K3, which defaults to
	// max.
	ReasoningEffort *ReasoningEffort `json:"reasoningEffort,omitempty"`

	// Prediction is static predicted content that can accelerate responses
	// when much of the output is known ahead of time.
	Prediction *Prediction `json:"prediction,omitempty"`

	// Thinking configures thinking on Kimi K2.x models. Kimi K2.5 and K2.6
	// can enable or disable it; Kimi K2.7 Code always thinks.
	Thinking *Thinking `json:"thinking,omitempty"`

	// ReasoningHistory controls preserved reasoning in multi-turn
	// conversations.
	ReasoningHistory *ReasoningHistory `json:"reasoningHistory,omitempty"`

	// PromptCacheKey improves cache hit rates of similar requests. It is
	// typically a session or task ID.
	PromptCacheKey *string `json:"promptCacheKey,omitempty"`

	// SafetyIdentifier is a stable identifier that helps Moonshot detect
	// users violating usage policies, such as a hashed username.
	SafetyIdentifier *string `json:"safetyIdentifier,omitempty"`
}

// Prediction is predicted output content.
type Prediction struct {
	// Type is always "content".
	Type string `json:"type"`

	// Content is a string or an array of text parts.
	Content PredictionContent `json:"content"`
}

// PredictionContent is predicted text, sent as a string unless Parts is
// set.
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

// MarshalJSON encodes the parts, or the text when there are no parts.
func (c PredictionContent) MarshalJSON() ([]byte, error) {
	if c.Parts != nil {
		return json.Marshal(c.Parts)
	}

	return json.Marshal(c.Text)
}

// UnmarshalJSON accepts a string or an array of text parts.
func (c *PredictionContent) UnmarshalJSON(data []byte) error {
	if strings.HasPrefix(strings.TrimSpace(string(data)), "[") {
		c.Text = ""

		return json.Unmarshal(data, &c.Parts)
	}

	c.Parts = nil

	return json.Unmarshal(data, &c.Text)
}

// Thinking configures thinking.
type Thinking struct {
	Type *ThinkingType `json:"type,omitempty"`

	// BudgetTokens is accepted for backwards compatibility and omitted with
	// a warning, because Moonshot does not support thinking budgets.
	//
	// Deprecated: Moonshot Chat Completions does not support thinking
	// budgets.
	BudgetTokens *int `json:"budgetTokens,omitempty"`
}

// MessageOptions are the provider options of prompt messages, read from the
// "moonshotai" namespace.
type MessageOptions struct {
	// Name is the participant represented by a system, user or assistant
	// message.
	Name *string `json:"name,omitempty"`

	// Partial marks the final assistant message as a partial response that
	// Moonshot continues. It cannot be combined with the JSON object
	// response format.
	Partial *bool `json:"partial,omitempty"`

	// Tools are function tools that a system message with empty content
	// loads at this point of a Kimi K3 conversation.
	Tools []DynamicTool `json:"tools,omitempty"`
}

// DynamicTool is a function tool loaded by a system message.
type DynamicTool struct {
	// Type is always "function".
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema"`
	Strict      *bool           `json:"strict,omitempty"`
}

// ChatMetadata is the result provider metadata of chat models, stored under
// "moonshotai".
type ChatMetadata struct {
	Logprobs       *Logprobs `json:"logprobs,omitempty"`
	ResponseObject string    `json:"responseObject,omitempty"`
	ChoiceIndex    *int      `json:"choiceIndex,omitempty"`
	MessageRole    string    `json:"messageRole,omitempty"`
	ToolCallTypes  []string  `json:"toolCallTypes,omitempty"`
}

// Logprobs are the log probabilities of the generated tokens.
type Logprobs struct {
	Content []Logprob `json:"content"`
}

// Logprob is the log probability of one token.
type Logprob struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes"`
	TopLogprobs []TopLogprob `json:"top_logprobs"`
}

// TopLogprob is one of the most likely tokens at a position.
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}

// readChatOptions decodes and validates the "moonshotai" provider options.
func readChatOptions(options model.ProviderOptions) (ChatOptions, error) {
	chatOptions, _, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return ChatOptions{}, err
	}

	if err := chatOptions.validate(); err != nil {
		return ChatOptions{}, fmt.Errorf("%w: moonshotai provider options: %w", model.ErrInvalidArgument, err)
	}

	return chatOptions, nil
}

func (o ChatOptions) validate() error {
	if o.TopLogprobs != nil && (*o.TopLogprobs < 0 || *o.TopLogprobs > maxTopLogprobs) {
		return fmt.Errorf("topLogprobs must be between 0 and %d", maxTopLogprobs)
	}

	if o.ReasoningEffort != nil {
		switch *o.ReasoningEffort {
		case ReasoningEffortLow, ReasoningEffortHigh, ReasoningEffortMax:
		default:
			return fmt.Errorf("unsupported reasoningEffort %q", *o.ReasoningEffort)
		}
	}

	if err := o.Prediction.validate(); err != nil {
		return err
	}

	if err := o.Thinking.validate(); err != nil {
		return err
	}

	if o.ReasoningHistory != nil {
		switch *o.ReasoningHistory {
		case ReasoningHistoryDisabled, ReasoningHistoryInterleaved, ReasoningHistoryPreserved:
		default:
			return fmt.Errorf("unsupported reasoningHistory %q", *o.ReasoningHistory)
		}
	}

	return nil
}

func (p *Prediction) validate() error {
	if p == nil {
		return nil
	}

	if p.Type != predictionContent {
		return fmt.Errorf("prediction.type must be %q", predictionContent)
	}

	for _, part := range p.Content.Parts {
		if part.Type != predictionTextType {
			return fmt.Errorf("prediction content parts must have type %q", predictionTextType)
		}
	}

	return nil
}

func (t *Thinking) validate() error {
	if t == nil {
		return nil
	}

	if t.Type != nil && *t.Type != ThinkingEnabled && *t.Type != ThinkingDisabled {
		return fmt.Errorf("thinking.type must be %q or %q", ThinkingEnabled, ThinkingDisabled)
	}

	if t.BudgetTokens != nil && *t.BudgetTokens < minThinkingBudget {
		return fmt.Errorf("thinking.budgetTokens must be at least %d", minThinkingBudget)
	}

	return nil
}

// errPartialValue reports a partial flag other than true.
var errPartialValue = errors.New("partial must be true when set")

// readMessageOptions decodes and validates the options of one message.
func readMessageOptions(options model.ProviderOptions) (MessageOptions, error) {
	messageOptions, _, err := model.ProviderValue[MessageOptions](options, Namespace)
	if err != nil {
		return MessageOptions{}, err
	}

	if messageOptions.Partial != nil && !*messageOptions.Partial {
		return MessageOptions{}, fmt.Errorf("%w: moonshotai message options: %w", model.ErrInvalidArgument, errPartialValue)
	}

	for _, tool := range messageOptions.Tools {
		if tool.Type != dynamicToolType {
			return MessageOptions{}, fmt.Errorf("%w: moonshotai message options: tool type must be %q", model.ErrInvalidArgument, dynamicToolType)
		}
	}

	return messageOptions, nil
}

// isPartial reports whether the message is a partial response.
func (o MessageOptions) isPartial() bool {
	return o.Partial != nil && *o.Partial
}

// Family is a Moonshot AI model family.
type Family string

// Model families.
const (
	FamilyKimiK25   Family = "kimi-k2.5"
	FamilyKimiK26   Family = "kimi-k2.6"
	FamilyKimiK27   Family = "kimi-k2.7"
	FamilyKimiK3    Family = "kimi-k3"
	FamilyMoonshotV Family = "moonshot-v1"
	FamilyUnknown   Family = "unknown"
)

// ModelFamily returns the family of a model ID. Custom model IDs belong to
// FamilyUnknown.
func ModelFamily(modelID string) Family {
	switch {
	case modelID == "kimi-k2.5":
		return FamilyKimiK25
	case modelID == "kimi-k2.6":
		return FamilyKimiK26
	case modelID == "kimi-k2.7-code", modelID == "kimi-k2.7-code-highspeed":
		return FamilyKimiK27
	case modelID == "kimi-k3":
		return FamilyKimiK3
	case strings.HasPrefix(modelID, "moonshot-v1-"):
		return FamilyMoonshotV
	default:
		return FamilyUnknown
	}
}

// isKimi reports whether a family is a Kimi family, whose sampling settings
// are fixed by the model.
func (f Family) isKimi() bool {
	return strings.HasPrefix(string(f), "kimi-")
}
