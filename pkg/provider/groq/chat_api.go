package groq

import (
	"bytes"
	"encoding/json"
)

// chatRequest is a Groq chat completions request.
type chatRequest struct {
	Model string `json:"model"`

	User              *string `json:"user,omitempty"`
	ParallelToolCalls *bool   `json:"parallel_tool_calls,omitempty"`

	MaxTokens        *int     `json:"max_tokens,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	Seed             *int     `json:"seed,omitempty"`

	ResponseFormat *responseFormat `json:"response_format,omitempty"`

	ReasoningFormat ReasoningFormat `json:"reasoning_format,omitempty"`
	ReasoningEffort ReasoningEffort `json:"reasoning_effort,omitempty"`
	ServiceTier     ServiceTier     `json:"service_tier,omitempty"`

	Messages []chatMessage `json:"messages"`

	// Tools is sent when the call has tools, even when none is supported.
	Tools      []tool      `json:"tools,omitzero"`
	ToolChoice *toolChoice `json:"tool_choice,omitempty"`

	Stream bool `json:"stream,omitempty"`
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
	Role    string `json:"role"`
	Content string `json:"content"`
}

type userMessage struct {
	Role    string      `json:"role"`
	Content userContent `json:"content"`
}

type assistantMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	Reasoning string            `json:"reasoning,omitempty"`
	ToolCalls []messageToolCall `json:"tool_calls,omitempty"`
}

type toolMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
}

func (systemMessage) isChatMessage()    {}
func (userMessage) isChatMessage()      {}
func (assistantMessage) isChatMessage() {}
func (toolMessage) isChatMessage()      {}

// userContent is a single text or a list of content parts.
type userContent struct {
	Text  string
	Parts []contentPart
}

// MarshalJSON encodes the parts when there are any and the text otherwise.
func (c userContent) MarshalJSON() ([]byte, error) {
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
)

// contentPart is a text or image part. The field matching Type is set.
type contentPart struct {
	Type     contentPartType `json:"type"`
	Text     *string         `json:"text,omitempty"`
	ImageURL *imageURL       `json:"image_url,omitempty"`
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

// tool is a function tool or the browser search tool.
type tool struct {
	Type     string        `json:"type"`
	Function *toolFunction `json:"function,omitempty"`
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

// chatResponse is the subset of a chat completion that the model reads.
type chatResponse struct {
	ID      *string      `json:"id"`
	Created *float64     `json:"created"`
	Model   *string      `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *chatUsage   `json:"usage"`
}

type chatChoice struct {
	Message      responseMessage `json:"message"`
	Index        int             `json:"index"`
	FinishReason *string         `json:"finish_reason"`
}

type responseMessage struct {
	Content   *string            `json:"content"`
	Reasoning *string            `json:"reasoning"`
	ToolCalls []responseToolCall `json:"tool_calls"`
}

type responseToolCall struct {
	ID       *string      `json:"id"`
	Type     string       `json:"type"`
	Function functionCall `json:"function"`
}

// chatChunk is one streamed chat completion event: a delta or an error.
type chatChunk struct {
	ID      *string       `json:"id"`
	Created *float64      `json:"created"`
	Model   *string       `json:"model"`
	Choices []chunkChoice `json:"choices"`
	XGroq   *xGroq        `json:"x_groq"`

	// Error is set on error events.
	Error *ErrorDetail `json:"error"`
}

type chunkChoice struct {
	Delta        *chunkDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
	Index        int         `json:"index"`
}

type chunkDelta struct {
	Content   *string         `json:"content"`
	Reasoning *string         `json:"reasoning"`
	ToolCalls []chunkToolCall `json:"tool_calls"`
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

// xGroq carries Groq's extensions of a streamed chunk.
type xGroq struct {
	Usage *chatUsage `json:"usage"`
}

// chatUsage is the token usage of a chat completion.
type chatUsage struct {
	PromptTokens            *int                     `json:"prompt_tokens"`
	CompletionTokens        *int                     `json:"completion_tokens"`
	TotalTokens             *int                     `json:"total_tokens"`
	PromptTokensDetails     *promptTokensDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *completionTokensDetails `json:"completion_tokens_details"`

	// Raw is the usage object as Groq sent it.
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
