package openaicompatible

import (
	"bytes"
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

type completionRequest struct {
	Model string `json:"model"`

	Echo      *bool              `json:"echo,omitempty"`
	LogitBias map[string]float64 `json:"logit_bias,omitempty"`
	Suffix    *string            `json:"suffix,omitempty"`
	User      *string            `json:"user,omitempty"`

	MaxTokens        *int     `json:"max_tokens,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	Seed             *int     `json:"seed,omitempty"`
}

type completionResponse struct {
	ID      string             `json:"id"`
	Created float64            `json:"created"`
	Model   string             `json:"model"`
	Choices []completionChoice `json:"choices"`
	Usage   *completionUsage   `json:"usage"`

	// Error is set on stream error events.
	Error json.RawMessage `json:"error"`
}

type completionChoice struct {
	Text         *string `json:"text"`
	FinishReason *string `json:"finish_reason"`
}

type completionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	Raw json.RawMessage `json:"-"`
}

// UnmarshalJSON decodes the usage and keeps the raw object.
func (u *completionUsage) UnmarshalJSON(data []byte) error {
	type plain completionUsage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = completionUsage(decoded)
	u.Raw = bytes.Clone(data)

	return nil
}

// convertCompletionUsage converts completion usage. Completions report no
// cache or reasoning breakdown.
func convertCompletionUsage(usage *completionUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:   new(usage.PromptTokens),
			NoCache: new(usage.PromptTokens),
		},
		OutputTokens: model.OutputTokens{
			Total: new(usage.CompletionTokens),
			Text:  new(usage.CompletionTokens),
		},
		Raw: usage.Raw,
	}
}
