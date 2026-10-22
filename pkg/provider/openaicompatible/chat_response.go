package openaicompatible

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// chatResponse is the subset of a chat completion that the model reads.
type chatResponse struct {
	ID      string       `json:"id"`
	Created float64      `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *ChatUsage   `json:"usage"`
}

type chatChoice struct {
	Message      chatResponseMessage `json:"message"`
	FinishReason *string             `json:"finish_reason"`
}

type chatResponseMessage struct {
	Content responseContent `json:"content"`

	// ReasoningContent is the usual reasoning field. Some providers serving
	// gpt-oss use Reasoning instead.
	ReasoningContent *string `json:"reasoning_content"`
	Reasoning        *string `json:"reasoning"`

	ToolCalls []chatResponseToolCall `json:"tool_calls"`
}

type chatResponseToolCall struct {
	ID           string        `json:"id"`
	Function     functionCall  `json:"function"`
	ExtraContent *extraContent `json:"extra_content"`
}

// chatChunk is one streamed chat completion event: a delta or an error.
type chatChunk struct {
	ID      string            `json:"id"`
	Created float64           `json:"created"`
	Model   string            `json:"model"`
	Choices []chatChunkChoice `json:"choices"`
	Usage   *ChatUsage        `json:"usage"`

	// Error is set on error events.
	Error json.RawMessage `json:"error"`
}

type chatChunkChoice struct {
	Delta        *chatDelta `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

type chatDelta struct {
	Content          responseContent     `json:"content"`
	ReasoningContent *string             `json:"reasoning_content"`
	Reasoning        *string             `json:"reasoning"`
	ToolCalls        []chatDeltaToolCall `json:"tool_calls"`
}

type chatDeltaToolCall struct {
	// Index is absent from Google's deltas.
	Index        *int               `json:"index"`
	ID           *string            `json:"id"`
	Function     *chatDeltaFunction `json:"function"`
	ExtraContent *extraContent      `json:"extra_content"`
}

type chatDeltaFunction struct {
	Name      *string `json:"name"`
	Arguments *string `json:"arguments"`
}

// reasoningText returns the reasoning of a message or delta.
func reasoningText(reasoningContent *string, reasoning *string) string {
	if reasoningContent != nil {
		return *reasoningContent
	}

	if reasoning != nil {
		return *reasoning
	}

	return ""
}

// responseContent is message content: absent, a string, or a list of typed
// parts. Only text and thinking parts are read.
type responseContent struct {
	Text  string
	Parts []responseContentPart
}

// responseContentPart is a loosely typed content part. Its text and thinking
// members are kept raw because providers send parts of other shapes.
type responseContentPart struct {
	Type     string          `json:"type"`
	Text     json.RawMessage `json:"text"`
	Thinking json.RawMessage `json:"thinking"`
}

// thinkingChunk is one item of a thinking part.
type thinkingChunk struct {
	Type string  `json:"type"`
	Text *string `json:"text"`
}

// UnmarshalJSON accepts null, a string or an array of parts.
func (c *responseContent) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)

	switch {
	case bytes.Equal(trimmed, []byte("null")):
		*c = responseContent{}

		return nil

	case bytes.HasPrefix(trimmed, []byte(`"`)):
		*c = responseContent{}

		return json.Unmarshal(trimmed, &c.Text)

	case bytes.HasPrefix(trimmed, []byte("[")):
		*c = responseContent{}

		return json.Unmarshal(trimmed, &c.Parts)

	default:
		return fmt.Errorf("%w: message content must be a string or an array", model.ErrInvalidResponse)
	}
}

// contentPiece is generated text or reasoning text.
type contentPiece struct {
	Text      string
	Reasoning bool
}

// pieces returns the non-empty text and reasoning of the content in order.
func (c responseContent) pieces() []contentPiece {
	if c.Parts == nil {
		if c.Text == "" {
			return nil
		}

		return []contentPiece{
			{
				Text: c.Text,
			},
		}
	}

	var pieces []contentPiece
	for _, part := range c.Parts {
		piece, ok := part.piece()
		if ok {
			pieces = append(pieces, piece)
		}
	}

	return pieces
}

// piece returns the text of a text part or the joined text of a thinking
// part. Parts of other types or shapes are skipped.
func (p responseContentPart) piece() (contentPiece, bool) {
	switch p.Type {
	case "text":
		var text string
		if err := json.Unmarshal(p.Text, &text); err != nil || text == "" {
			return contentPiece{}, false
		}

		return contentPiece{
			Text: text,
		}, true

	case "thinking":
		text := joinThinking(p.Thinking)
		if text == "" {
			return contentPiece{}, false
		}

		return contentPiece{
			Text:      text,
			Reasoning: true,
		}, true

	default:
		return contentPiece{}, false
	}
}

// joinThinking concatenates the text items of a thinking array. Items that
// are not text objects are skipped, as are thinking values that are not
// arrays.
func joinThinking(thinking json.RawMessage) string {
	var items []json.RawMessage
	if err := json.Unmarshal(thinking, &items); err != nil {
		return ""
	}

	var text strings.Builder
	for _, item := range items {
		var chunk thinkingChunk
		if err := json.Unmarshal(item, &chunk); err != nil {
			continue
		}

		if chunk.Type == "text" && chunk.Text != nil {
			text.WriteString(*chunk.Text)
		}
	}

	return text.String()
}

// ChatUsage is the token usage of a chat completion.
type ChatUsage struct {
	PromptTokens            *int                     `json:"prompt_tokens"`
	CompletionTokens        *int                     `json:"completion_tokens"`
	TotalTokens             *int                     `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details"`

	// Raw is the usage object as the provider sent it.
	Raw json.RawMessage `json:"-"`
}

// PromptTokensDetails breaks down prompt tokens.
type PromptTokensDetails struct {
	CachedTokens *int `json:"cached_tokens"`
}

// CompletionTokensDetails breaks down completion tokens.
type CompletionTokensDetails struct {
	ReasoningTokens          *int `json:"reasoning_tokens"`
	AcceptedPredictionTokens *int `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens *int `json:"rejected_prediction_tokens"`
}

// UnmarshalJSON decodes the usage and keeps the raw object.
func (u *ChatUsage) UnmarshalJSON(data []byte) error {
	type plain ChatUsage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = ChatUsage(decoded)
	u.Raw = bytes.Clone(data)

	return nil
}

// UsageConverter converts chat usage into normalized usage. The usage is nil
// when the provider did not report it.
type UsageConverter func(usage *ChatUsage) model.Usage

// ConvertChatUsage is the default UsageConverter. Missing counts are treated
// as zero once the provider reports usage at all.
func ConvertChatUsage(usage *ChatUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	promptTokens := valueOrZero(usage.PromptTokens)
	completionTokens := valueOrZero(usage.CompletionTokens)

	cacheReadTokens := 0
	if usage.PromptTokensDetails != nil {
		cacheReadTokens = valueOrZero(usage.PromptTokensDetails.CachedTokens)
	}

	reasoningTokens := 0
	if usage.CompletionTokensDetails != nil {
		reasoningTokens = valueOrZero(usage.CompletionTokensDetails.ReasoningTokens)
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:     new(promptTokens),
			NoCache:   new(promptTokens - cacheReadTokens),
			CacheRead: new(cacheReadTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(completionTokens),
			Text:      new(max(0, completionTokens-reasoningTokens)),
			Reasoning: new(reasoningTokens),
		},
		Raw: usage.Raw,
	}
}

// MapFinishReason converts a chat completion finish reason.
func MapFinishReason(finishReason string) model.FinishReason {
	return model.FinishReason{
		Unified: unifiedFinishReason(finishReason),
		Raw:     finishReason,
	}
}

func unifiedFinishReason(finishReason string) model.UnifiedFinishReason {
	switch finishReason {
	case "stop":
		return model.FinishStop
	case "length":
		return model.FinishLength
	case "content_filter":
		return model.FinishContentFilter
	case "function_call", "tool_calls":
		return model.FinishToolCalls
	default:
		return model.FinishOther
	}
}

// responseMetadata converts the common response identification fields. Some
// providers send zero as a placeholder timestamp, which is ignored.
func responseMetadata(id string, modelID string, created float64) model.ResponseMetadata {
	metadata := model.ResponseMetadata{
		ID:      id,
		ModelID: modelID,
	}

	if created != 0 {
		metadata.Timestamp = time.Unix(int64(created), 0)
	}

	return metadata
}

// isEmptyMetadata reports whether no identification field is set.
func isEmptyMetadata(metadata model.ResponseMetadata) bool {
	return metadata.ID == "" && metadata.ModelID == "" && metadata.Timestamp.IsZero()
}

func valueOrZero(value *int) int {
	if value == nil {
		return 0
	}

	return *value
}
