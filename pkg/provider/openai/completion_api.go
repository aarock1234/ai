package openai

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// completionRequest is the body of a legacy Completions request.
type completionRequest struct {
	Model string `json:"model"`

	Echo      *bool              `json:"echo,omitempty"`
	LogitBias map[string]float64 `json:"logit_bias,omitempty"`
	Logprobs  *int               `json:"logprobs,omitempty"`
	Suffix    string             `json:"suffix,omitempty"`
	User      string             `json:"user,omitempty"`

	MaxTokens        *int     `json:"max_tokens,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	Seed             *int     `json:"seed,omitempty"`

	Prompt string   `json:"prompt"`
	Stop   []string `json:"stop,omitempty"`

	Stream        bool               `json:"stream,omitempty"`
	StreamOptions *chatStreamOptions `json:"stream_options,omitempty"`
}

// completionResponse is a legacy Completions response.
type completionResponse struct {
	ID      string             `json:"id"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []completionChoice `json:"choices"`
	Usage   *completionUsage   `json:"usage"`
}

// completionChoice is one generated choice.
type completionChoice struct {
	Text         *string             `json:"text"`
	FinishReason *string             `json:"finish_reason"`
	Index        int                 `json:"index"`
	Logprobs     *CompletionLogprobs `json:"logprobs"`
}

// completionUsage is the token usage of a Completions call. It keeps the
// original JSON for model.Usage.Raw.
type completionUsage struct {
	PromptTokens     *int `json:"prompt_tokens"`
	CompletionTokens *int `json:"completion_tokens"`
	TotalTokens      *int `json:"total_tokens"`

	raw json.RawMessage
}

// UnmarshalJSON decodes the usage and keeps a copy of the original JSON.
func (u *completionUsage) UnmarshalJSON(data []byte) error {
	type plain completionUsage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = completionUsage(decoded)
	u.raw = bytes.Clone(data)

	return nil
}

// completionChunk is one streamed Completions event: a completion chunk or
// an error frame.
type completionChunk struct {
	ID      string             `json:"id"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []completionChoice `json:"choices"`
	Usage   *completionUsage   `json:"usage"`

	Error *chatErrorFrame `json:"error"`
}

// isErrorFrame reports whether the chunk is an error frame rather than a
// completion chunk.
func (c completionChunk) isErrorFrame() bool {
	return c.Choices == nil && c.Error != nil
}

// hasOutput reports whether the chunk carries generated text.
func (c completionChunk) hasOutput() bool {
	for _, choice := range c.Choices {
		if choice.Text != nil && *choice.Text != "" {
			return true
		}
	}

	return false
}

// validate checks the fields that the response requires.
func (r completionResponse) validate() error {
	if len(r.Choices) == 0 {
		return fmt.Errorf("%w: response did not contain any choices", model.ErrInvalidResponse)
	}

	for _, choice := range r.Choices {
		if choice.Text == nil || choice.FinishReason == nil {
			return fmt.Errorf("%w: choice requires text and finish_reason", model.ErrInvalidResponse)
		}
	}

	return r.Usage.validate()
}

// validate checks the fields that a chunk requires.
func (c completionChunk) validate() error {
	if c.Choices == nil {
		return fmt.Errorf("%w: chunk has no choices", model.ErrInvalidResponse)
	}

	for _, choice := range c.Choices {
		if choice.Text == nil {
			return fmt.Errorf("%w: choice requires text", model.ErrInvalidResponse)
		}
	}

	return c.Usage.validate()
}

func (u *completionUsage) validate() error {
	if u == nil {
		return nil
	}

	if u.PromptTokens == nil || u.CompletionTokens == nil || u.TotalTokens == nil {
		return fmt.Errorf("%w: usage requires prompt, completion and total tokens", model.ErrInvalidResponse)
	}

	return nil
}
