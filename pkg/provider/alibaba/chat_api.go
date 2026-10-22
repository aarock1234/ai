package alibaba

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// chatRequest is an Alibaba chat completions request.
type chatRequest struct {
	Model string `json:"model"`

	MaxTokens       *int     `json:"max_tokens,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"top_p,omitempty"`
	TopK            *int     `json:"top_k,omitempty"`
	PresencePenalty *float64 `json:"presence_penalty,omitempty"`
	Stop            []string `json:"stop,omitempty"`
	Seed            *int     `json:"seed,omitempty"`

	ResponseFormat *responseFormat `json:"response_format,omitempty"`

	EnableThinking   *bool `json:"enable_thinking,omitempty"`
	ThinkingBudget   *int  `json:"thinking_budget,omitempty"`
	PreserveThinking *bool `json:"preserve_thinking,omitempty"`

	Messages []chatMessage `json:"messages"`

	// Tools is sent when the call has tools, even when none is supported.
	Tools             []tool      `json:"tools,omitzero"`
	ToolChoice        *toolChoice `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool       `json:"parallel_tool_calls,omitempty"`

	Stream        bool           `json:"stream,omitempty"`
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// responseFormatType is the wire response format type.
type responseFormatType string

const (
	responseFormatJSONSchema responseFormatType = "json_schema"
	responseFormatJSONObject responseFormatType = "json_object"
)

type responseFormat struct {
	Type       responseFormatType `json:"type"`
	JSONSchema *jsonSchemaFormat  `json:"json_schema,omitempty"`
}

type jsonSchemaFormat struct {
	Schema      json.RawMessage `json:"schema"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
}

// chatMessage is a systemMessage, userMessage, assistantMessage or
// toolMessage.
type chatMessage interface {
	isChatMessage()
}

type systemMessage struct {
	Role    string      `json:"role"`
	Content textContent `json:"content"`
}

type userMessage struct {
	Role    string     `json:"role"`
	Content []userPart `json:"content"`
}

type assistantMessage struct {
	Role             string            `json:"role"`
	Content          assistantContent  `json:"content"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []messageToolCall `json:"tool_calls,omitempty"`
}

type toolMessage struct {
	Role       string      `json:"role"`
	ToolCallID string      `json:"tool_call_id"`
	Content    textContent `json:"content"`
}

func (systemMessage) isChatMessage()    {}
func (userMessage) isChatMessage()      {}
func (assistantMessage) isChatMessage() {}
func (toolMessage) isChatMessage()      {}

// textContent is plain text, or text parts when a part carries a cache
// control.
type textContent struct {
	Text  string
	Parts []textPart
}

// MarshalJSON encodes the parts when there are any and the text otherwise.
func (c textContent) MarshalJSON() ([]byte, error) {
	if c.Parts != nil {
		return json.Marshal(c.Parts)
	}

	return json.Marshal(c.Text)
}

// assistantContent is absent, plain text, or a cached text part.
type assistantContent struct {
	Text  *string
	Parts []textPart
}

// MarshalJSON encodes the parts, the text, or null.
func (c assistantContent) MarshalJSON() ([]byte, error) {
	if c.Parts != nil {
		return json.Marshal(c.Parts)
	}

	return json.Marshal(c.Text)
}

type textPart struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// userPartType is the wire type of a user content part.
type userPartType string

const (
	userPartText     userPartType = "text"
	userPartImageURL userPartType = "image_url"
)

// userPart is a text or image part. The field matching Type is set.
type userPart struct {
	Type         userPartType  `json:"type"`
	Text         *string       `json:"text,omitempty"`
	ImageURL     *imageURL     `json:"image_url,omitempty"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

type imageURL struct {
	URL string `json:"url"`
}

type messageToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function functionCall `json:"function"`
}

type functionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type tool struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict,omitempty"`
}

// toolChoice is a mode such as "auto" or a specific function.
type toolChoice struct {
	Mode         string
	FunctionName string
}

// MarshalJSON encodes a mode as a string and a function as an object.
func (c toolChoice) MarshalJSON() ([]byte, error) {
	if c.FunctionName == "" {
		return json.Marshal(c.Mode)
	}

	return json.Marshal(namedToolChoice{
		Type: functionType,
		Function: namedFunction{
			Name: c.FunctionName,
		},
	})
}

type namedToolChoice struct {
	Type     string        `json:"type"`
	Function namedFunction `json:"function"`
}

type namedFunction struct {
	Name string `json:"name"`
}

// Literal values that Alibaba responses must carry.
const (
	assistantRole = "assistant"
	functionType  = "function"
)

// chatResponse is the subset of a chat completion that the model reads.
type chatResponse struct {
	ID      *string      `json:"id"`
	Created *float64     `json:"created"`
	Model   *string      `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *Usage       `json:"usage"`
}

type chatChoice struct {
	Message      responseMessage `json:"message"`
	FinishReason *string         `json:"finish_reason"`
	Index        int             `json:"index"`
}

type responseMessage struct {
	Role             *string            `json:"role"`
	Content          *string            `json:"content"`
	ReasoningContent *string            `json:"reasoning_content"`
	ToolCalls        []responseToolCall `json:"tool_calls"`
}

type responseToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function functionCall `json:"function"`
}

// validate checks the literal fields of the response.
func (r chatResponse) validate() error {
	for _, choice := range r.Choices {
		if err := expectLiteral("message role", choice.Message.Role, assistantRole); err != nil {
			return err
		}

		for _, toolCall := range choice.Message.ToolCalls {
			if err := expectLiteral("tool call type", &toolCall.Type, functionType); err != nil {
				return err
			}
		}
	}

	return nil
}

// chatChunk is one streamed chat completion event.
type chatChunk struct {
	ID      *string       `json:"id"`
	Created *float64      `json:"created"`
	Model   *string       `json:"model"`
	Choices []chunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage"`
}

type chunkChoice struct {
	Delta        *chunkDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
	Index        int         `json:"index"`
}

type chunkDelta struct {
	Role             *string         `json:"role"`
	Content          *string         `json:"content"`
	ReasoningContent *string         `json:"reasoning_content"`
	ToolCalls        []chunkToolCall `json:"tool_calls"`
}

type chunkToolCall struct {
	Index    *int           `json:"index"`
	ID       *string        `json:"id"`
	Type     *string        `json:"type"`
	Function *chunkFunction `json:"function"`
}

type chunkFunction struct {
	Name      *string `json:"name"`
	Arguments *string `json:"arguments"`
}

// validate checks the required and literal fields of a chunk.
func (c chatChunk) validate() error {
	if c.Choices == nil {
		return fmt.Errorf("%w: chunk has no choices", model.ErrInvalidResponse)
	}

	for _, choice := range c.Choices {
		if choice.Delta == nil {
			return fmt.Errorf("%w: chunk choice has no delta", model.ErrInvalidResponse)
		}

		if err := expectLiteral("delta role", choice.Delta.Role, assistantRole); err != nil {
			return err
		}

		for _, toolCall := range choice.Delta.ToolCalls {
			if err := expectLiteral("tool call type", toolCall.Type, functionType); err != nil {
				return err
			}
		}
	}

	return nil
}

// expectLiteral checks that an optional field, when present, has the
// expected value.
func expectLiteral(field string, value *string, expected string) error {
	if value == nil || *value == expected {
		return nil
	}

	return fmt.Errorf("%w: %s %q, expected %q", model.ErrInvalidResponse, field, *value, expected)
}

// Usage is the token usage of an Alibaba chat completion. Alibaba counts
// cache reads and cache writes inside the prompt tokens.
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details"`

	// Raw is the usage object as Alibaba sent it, including fields not
	// listed here.
	Raw json.RawMessage `json:"-"`
}

// PromptTokensDetails breaks down prompt tokens.
type PromptTokensDetails struct {
	CachedTokens             *int `json:"cached_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`

	// CacheType is the caching mode that served the request. It is reported
	// only for explicit context caching, which cache controls select.
	CacheType *string `json:"cache_type"`
}

// CompletionTokensDetails breaks down completion tokens.
type CompletionTokensDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

// UnmarshalJSON decodes the usage and keeps the raw object.
func (u *Usage) UnmarshalJSON(data []byte) error {
	type plain Usage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = Usage(decoded)
	u.Raw = bytes.Clone(data)

	return nil
}
