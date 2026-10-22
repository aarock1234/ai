package deepseek

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// chatRequest is a DeepSeek chat completions request.
type chatRequest struct {
	Model string `json:"model"`

	Logprobs    *bool `json:"logprobs,omitempty"`
	TopLogprobs *int  `json:"top_logprobs,omitempty"`

	MaxTokens        *int     `json:"max_tokens,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`

	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	Messages       []chatMessage   `json:"messages"`

	// Tools is sent when the call has tools, even when none is supported.
	Tools      []tool      `json:"tools,omitzero"`
	ToolChoice *toolChoice `json:"tool_choice,omitempty"`

	Thinking        *thinking       `json:"thinking,omitempty"`
	UserID          *string         `json:"user_id,omitempty"`
	ReasoningEffort ReasoningEffort `json:"reasoning_effort,omitempty"`

	Stream        bool           `json:"stream,omitempty"`
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

type thinking struct {
	Type ThinkingType `json:"type"`
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
	Strict      bool            `json:"strict"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
}

// chatMessage is a systemMessage, userMessage, assistantMessage or
// toolMessage.
type chatMessage interface {
	isChatMessage()
}

type systemMessage struct {
	Role    string  `json:"role"`
	Content string  `json:"content"`
	Name    *string `json:"name,omitempty"`
}

type userMessage struct {
	Role    string         `json:"role"`
	Content messageContent `json:"content"`
	Name    *string        `json:"name,omitempty"`
}

type assistantMessage struct {
	Role    string  `json:"role"`
	Content string  `json:"content"`
	Name    *string `json:"name,omitempty"`
	Prefix  *bool   `json:"prefix,omitempty"`

	// ReasoningContent is sent as an empty string to V4 models, which
	// require it on every assistant turn.
	ReasoningContent *string           `json:"reasoning_content,omitempty"`
	ToolCalls        []messageToolCall `json:"tool_calls,omitempty"`
}

type toolMessage struct {
	Role       string         `json:"role"`
	Content    messageContent `json:"content"`
	ToolCallID string         `json:"tool_call_id"`
}

func (systemMessage) isChatMessage()    {}
func (userMessage) isChatMessage()      {}
func (assistantMessage) isChatMessage() {}
func (toolMessage) isChatMessage()      {}

// messageContent is plain text or a list of content parts.
type messageContent struct {
	Text  string
	Parts []contentPart
}

// MarshalJSON encodes the parts when there are any and the text otherwise.
func (c messageContent) MarshalJSON() ([]byte, error) {
	if c.Parts != nil {
		return json.Marshal(c.Parts)
	}

	return json.Marshal(c.Text)
}

// contentPartType is the wire type of a content part.
type contentPartType string

const (
	contentPartText     contentPartType = "text"
	contentPartImageURL contentPartType = "image_url"
	contentPartFile     contentPartType = "file"
)

// contentPart is a text, image or file part. File parts carry either an
// uploaded file ID or inline file data.
type contentPart struct {
	Type     contentPartType `json:"type"`
	Text     *string         `json:"text,omitempty"`
	ImageURL *imageURL       `json:"image_url,omitempty"`
	FileID   string          `json:"file_id,omitempty"`
	FileData string          `json:"file_data,omitempty"`
	Filename string          `json:"filename,omitempty"`
}

type imageURL struct {
	URL    string      `json:"url"`
	Detail ImageDetail `json:"detail,omitempty"`
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
		Type: "function",
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

// Literal values that DeepSeek responses must carry when they report them.
const (
	completionObject      = "chat.completion"
	completionChunkObject = "chat.completion.chunk"
	assistantRole         = "assistant"
	functionType          = "function"
)

// chatResponse is the subset of a chat completion that the model reads.
type chatResponse struct {
	ID                *string      `json:"id"`
	Created           *float64     `json:"created"`
	Model             *string      `json:"model"`
	Object            *string      `json:"object"`
	SystemFingerprint *string      `json:"system_fingerprint"`
	Choices           []chatChoice `json:"choices"`
	Usage             *chatUsage   `json:"usage"`
}

type chatChoice struct {
	Index        *int            `json:"index"`
	Message      responseMessage `json:"message"`
	Logprobs     *Logprobs       `json:"logprobs"`
	FinishReason *string         `json:"finish_reason"`
}

type responseMessage struct {
	Role             *string            `json:"role"`
	Content          *string            `json:"content"`
	ReasoningContent *string            `json:"reasoning_content"`
	ToolCalls        []responseToolCall `json:"tool_calls"`
}

type responseToolCall struct {
	ID       *string      `json:"id"`
	Type     *string      `json:"type"`
	Function functionCall `json:"function"`
}

// validate checks the literal fields of the response.
func (r chatResponse) validate() error {
	if err := expectLiteral("object", r.Object, completionObject); err != nil {
		return err
	}

	for _, choice := range r.Choices {
		if err := expectLiteral("message role", choice.Message.Role, assistantRole); err != nil {
			return err
		}

		for _, toolCall := range choice.Message.ToolCalls {
			if err := expectLiteral("tool call type", toolCall.Type, functionType); err != nil {
				return err
			}
		}
	}

	return nil
}

// chatChunk is one streamed chat completion event: a delta or an error.
type chatChunk struct {
	ID                *string       `json:"id"`
	Created           *float64      `json:"created"`
	Model             *string       `json:"model"`
	Object            *string       `json:"object"`
	SystemFingerprint *string       `json:"system_fingerprint"`
	Choices           []chunkChoice `json:"choices"`
	Usage             *chatUsage    `json:"usage"`

	// Error is set on error events.
	Error *ErrorDetail `json:"error"`
}

type chunkChoice struct {
	Index        *int        `json:"index"`
	Delta        *chunkDelta `json:"delta"`
	Logprobs     *Logprobs   `json:"logprobs"`
	FinishReason *string     `json:"finish_reason"`
}

type chunkDelta struct {
	Role             *string         `json:"role"`
	Content          *string         `json:"content"`
	ReasoningContent *string         `json:"reasoning_content"`
	ToolCalls        []chunkToolCall `json:"tool_calls"`
}

type chunkToolCall struct {
	Index    *int          `json:"index"`
	ID       *string       `json:"id"`
	Type     *string       `json:"type"`
	Function chunkFunction `json:"function"`
}

type chunkFunction struct {
	Name      *string `json:"name"`
	Arguments *string `json:"arguments"`
}

// validate checks the literal and required fields of a delta chunk.
func (c chatChunk) validate() error {
	if c.Choices == nil {
		return fmt.Errorf("%w: chunk has no choices", model.ErrInvalidResponse)
	}

	if err := expectLiteral("object", c.Object, completionChunkObject); err != nil {
		return err
	}

	for _, choice := range c.Choices {
		if choice.Delta == nil {
			continue
		}

		if err := expectLiteral("delta role", choice.Delta.Role, assistantRole); err != nil {
			return err
		}

		for _, toolCall := range choice.Delta.ToolCalls {
			if toolCall.Index == nil {
				return fmt.Errorf("%w: tool call delta has no index", model.ErrInvalidResponse)
			}

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

// Logprobs are the log probabilities of the content and reasoning tokens.
type Logprobs struct {
	Content          []Logprob `json:"content,omitempty"`
	ReasoningContent []Logprob `json:"reasoning_content,omitempty"`
}

// Logprob is the log probability of one token and its most likely
// alternatives.
type Logprob struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes"`
	TopLogprobs []TopLogprob `json:"top_logprobs"`
}

// TopLogprob is one likely alternative token.
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}

// chatUsage is the token usage of a chat completion.
type chatUsage struct {
	PromptTokens          *int                     `json:"prompt_tokens"`
	CompletionTokens      *int                     `json:"completion_tokens"`
	PromptCacheHitTokens  *int                     `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens *int                     `json:"prompt_cache_miss_tokens"`
	TotalTokens           *int                     `json:"total_tokens"`
	PromptTokensDetails   *promptTokensDetails     `json:"prompt_tokens_details"`
	CompletionDetails     *completionTokensDetails `json:"completion_tokens_details"`

	// Raw is the usage object as DeepSeek sent it.
	Raw json.RawMessage `json:"-"`
}

type promptTokensDetails struct {
	CachedTokens *int `json:"cached_tokens"`
}

type completionTokensDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

// UnmarshalJSON decodes the usage and keeps the raw object.
func (u *chatUsage) UnmarshalJSON(data []byte) error {
	type plain chatUsage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = chatUsage(decoded)
	u.Raw = bytes.Clone(data)

	return nil
}
