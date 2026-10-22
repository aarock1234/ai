package moonshotai

import (
	"bytes"
	"encoding/json"
)

// Message roles and part types of the chat completions API.
const (
	roleSystem    = "system"
	roleUser      = "user"
	roleAssistant = "assistant"
	roleTool      = "tool"

	partText  = "text"
	partImage = "image_url"
	partVideo = "video_url"

	functionType = "function"
)

type chatRequest struct {
	Model               string           `json:"model"`
	Logprobs            *bool            `json:"logprobs,omitempty"`
	TopLogprobs         *int             `json:"top_logprobs,omitempty"`
	MaxCompletionTokens *int             `json:"max_completion_tokens,omitempty"`
	Temperature         *float64         `json:"temperature,omitempty"`
	TopP                *float64         `json:"top_p,omitempty"`
	FrequencyPenalty    *float64         `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64         `json:"presence_penalty,omitempty"`
	ResponseFormat      *responseFormat  `json:"response_format,omitempty"`
	Stop                []string         `json:"stop,omitempty"`
	Messages            []chatMessage    `json:"messages"`
	Tools               []functionTool   `json:"tools,omitempty"`
	ToolChoice          *toolChoice      `json:"tool_choice,omitempty"`
	Prediction          *Prediction      `json:"prediction,omitempty"`
	Thinking            *thinkingRequest `json:"thinking,omitempty"`
	ReasoningEffort     *ReasoningEffort `json:"reasoning_effort,omitempty"`
	PromptCacheKey      *string          `json:"prompt_cache_key,omitempty"`
	SafetyIdentifier    *string          `json:"safety_identifier,omitempty"`
	Stream              bool             `json:"stream,omitempty"`
	StreamOptions       *streamOptions   `json:"stream_options,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// Response format types.
const (
	responseFormatJSONObject = "json_object"
	responseFormatJSONSchema = "json_schema"
)

type responseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *jsonSchema `json:"json_schema,omitempty"`
}

type jsonSchema struct {
	Name   string          `json:"name"`
	Strict bool            `json:"strict"`
	Schema json.RawMessage `json:"schema"`
}

// thinkingKeepAll preserves all earlier reasoning.
const thinkingKeepAll = "all"

type thinkingRequest struct {
	Type ThinkingType `json:"type"`
	Keep string       `json:"keep,omitempty"`
}

// chatMessage is one request message: systemMessage, systemToolsMessage,
// userMessage, assistantMessage or toolMessage.
type chatMessage interface {
	isChatMessage()
}

type systemMessage struct {
	Role    string  `json:"role"`
	Content string  `json:"content"`
	Name    *string `json:"name,omitempty"`
}

// systemToolsMessage loads dynamic tools. The API forbids content alongside
// the tools.
type systemToolsMessage struct {
	Role  string         `json:"role"`
	Tools []functionTool `json:"tools"`
}

type userMessage struct {
	Role    string      `json:"role"`
	Content userContent `json:"content"`
	Name    *string     `json:"name,omitempty"`
}

// userContent is a single text, sent as a string, or a list of parts.
type userContent struct {
	text  string
	parts []contentPart
}

// MarshalJSON encodes the parts, or the text when there are no parts.
func (c userContent) MarshalJSON() ([]byte, error) {
	if c.parts != nil {
		return json.Marshal(c.parts)
	}

	return json.Marshal(c.text)
}

// contentPart is a textPart, imagePart or videoPart.
type contentPart interface {
	isContentPart()
}

type textPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type imagePart struct {
	Type     string   `json:"type"`
	ImageURL mediaURL `json:"image_url"`
}

type videoPart struct {
	Type     string   `json:"type"`
	VideoURL mediaURL `json:"video_url"`
}

type mediaURL struct {
	URL string `json:"url"`
}

// assistantMessage content is null when the message has tool calls and no
// text.
type assistantMessage struct {
	Role             string            `json:"role"`
	Content          *string           `json:"content"`
	Name             *string           `json:"name,omitempty"`
	Partial          *bool             `json:"partial,omitempty"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []messageToolCall `json:"tool_calls,omitempty"`
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

type toolMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
}

func (systemMessage) isChatMessage()      {}
func (systemToolsMessage) isChatMessage() {}
func (userMessage) isChatMessage()        {}
func (assistantMessage) isChatMessage()   {}
func (toolMessage) isChatMessage()        {}

func (textPart) isContentPart()  {}
func (imagePart) isContentPart() {}
func (videoPart) isContentPart() {}

type functionTool struct {
	Type     string             `json:"type"`
	Function functionDefinition `json:"function"`
}

type functionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict,omitempty"`
}

// toolChoice is "auto", "none", "required" or a named function.
type toolChoice struct {
	mode     string
	function string
}

type namedToolChoice struct {
	Type     string            `json:"type"`
	Function namedToolFunction `json:"function"`
}

type namedToolFunction struct {
	Name string `json:"name"`
}

// MarshalJSON encodes a mode as a string and a function as an object.
func (c toolChoice) MarshalJSON() ([]byte, error) {
	if c.function == "" {
		return json.Marshal(c.mode)
	}

	return json.Marshal(namedToolChoice{
		Type: functionType,
		Function: namedToolFunction{
			Name: c.function,
		},
	})
}

type chatResponse struct {
	ID      *string      `json:"id"`
	Created *float64     `json:"created"`
	Model   *string      `json:"model"`
	Object  *string      `json:"object"`
	Choices []chatChoice `json:"choices"`
	Usage   *chatUsage   `json:"usage"`
}

type chatChoice struct {
	Index        *int                `json:"index"`
	Message      chatResponseMessage `json:"message"`
	Logprobs     *Logprobs           `json:"logprobs"`
	FinishReason *string             `json:"finish_reason"`
}

type chatResponseMessage struct {
	Role             *string            `json:"role"`
	Content          *string            `json:"content"`
	ReasoningContent *string            `json:"reasoning_content"`
	ToolCalls        []responseToolCall `json:"tool_calls"`
}

type responseToolCall struct {
	ID       *string              `json:"id"`
	Type     *string              `json:"type"`
	Function responseFunctionCall `json:"function"`
}

type responseFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// chatChunk is a streamed chunk, or an error chunk when Error is set.
type chatChunk struct {
	ID      *string       `json:"id"`
	Created *float64      `json:"created"`
	Model   *string       `json:"model"`
	Object  *string       `json:"object"`
	Choices []chunkChoice `json:"choices"`
	Usage   *chatUsage    `json:"usage"`
	Error   *ErrorData    `json:"error"`
}

type chunkChoice struct {
	Index        *int        `json:"index"`
	Delta        *chunkDelta `json:"delta"`
	Logprobs     *Logprobs   `json:"logprobs"`
	FinishReason *string     `json:"finish_reason"`
	Usage        *chatUsage  `json:"usage"`
}

type chunkDelta struct {
	Role             *string         `json:"role"`
	Content          *string         `json:"content"`
	ReasoningContent *string         `json:"reasoning_content"`
	ToolCalls        []toolCallDelta `json:"tool_calls"`
}

type toolCallDelta struct {
	Index    *int              `json:"index"`
	ID       *string           `json:"id"`
	Type     *string           `json:"type"`
	Function functionCallDelta `json:"function"`
}

type functionCallDelta struct {
	Name      *string `json:"name"`
	Arguments *string `json:"arguments"`
}

// chatUsage is Moonshot's token usage. The raw object is kept as the usage
// Raw value.
type chatUsage struct {
	PromptTokens            *int                     `json:"prompt_tokens"`
	CompletionTokens        *int                     `json:"completion_tokens"`
	CachedTokens            *int                     `json:"cached_tokens"`
	TotalTokens             *int                     `json:"total_tokens"`
	PromptTokensDetails     *promptTokensDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *completionTokensDetails `json:"completion_tokens_details"`

	raw json.RawMessage
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
	u.raw = bytes.Clone(data)

	return nil
}
