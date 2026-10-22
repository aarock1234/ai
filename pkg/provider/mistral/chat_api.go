package mistral

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// chatRequest is a chat completions request body.
type chatRequest struct {
	Model      string `json:"model"`
	SafePrompt *bool  `json:"safe_prompt,omitempty"`

	completionSettings

	DocumentImageLimit *int    `json:"document_image_limit,omitempty"`
	DocumentPageLimit  *int    `json:"document_page_limit,omitempty"`
	PromptCacheKey     *string `json:"prompt_cache_key,omitempty"`

	Messages          []chatMessage `json:"messages"`
	Tools             []tool        `json:"tools,omitempty"`
	ToolChoice        *toolChoice   `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool         `json:"parallel_tool_calls,omitempty"`

	Stream bool `json:"stream,omitempty"`
}

// completionSettings are the sampling and output settings shared by chat
// completions and the Conversations API.
type completionSettings struct {
	MaxTokens        *int            `json:"max_tokens,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
	RandomSeed       *int            `json:"random_seed,omitempty"`
	ReasoningEffort  *string         `json:"reasoning_effort,omitempty"`
	ResponseFormat   *responseFormat `json:"response_format,omitempty"`
}

type responseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *jsonSchema `json:"json_schema,omitempty"`
}

type jsonSchema struct {
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
	Role    string        `json:"role"`
	Content []contentPart `json:"content"`
}

type assistantMessage struct {
	Role    string           `json:"role"`
	Content assistantContent `json:"content"`

	// Prefix marks a trailing assistant message that the model continues.
	Prefix *bool `json:"prefix,omitempty"`

	ToolCalls []messageToolCall `json:"tool_calls,omitempty"`
}

type toolMessage struct {
	Role       string `json:"role"`
	Name       string `json:"name"`
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
}

func (systemMessage) isChatMessage()    {}
func (userMessage) isChatMessage()      {}
func (assistantMessage) isChatMessage() {}
func (toolMessage) isChatMessage()      {}

// contentPartType is the type of a message content part.
type contentPartType string

const (
	contentText        contentPartType = "text"
	contentImageURL    contentPartType = "image_url"
	contentDocumentURL contentPartType = "document_url"
	contentThinking    contentPartType = "thinking"
)

// contentPart is a text, image, document or thinking part. The field
// matching Type is set.
type contentPart struct {
	Type contentPartType `json:"type"`

	Text        *string        `json:"text,omitempty"`
	ImageURL    string         `json:"image_url,omitempty"`
	DocumentURL string         `json:"document_url,omitempty"`
	Thinking    []thinkingText `json:"thinking,omitempty"`

	// Closed marks a complete thinking part.
	Closed bool `json:"closed,omitempty"`
}

type thinkingText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func textPart(text string) contentPart {
	return contentPart{
		Type: contentText,
		Text: &text,
	}
}

func thinkingPart(text string) contentPart {
	return contentPart{
		Type: contentThinking,
		Thinking: []thinkingText{
			{
				Type: "text",
				Text: text,
			},
		},
		Closed: true,
	}
}

// assistantContent is plain text, or parts when the message has reasoning.
type assistantContent struct {
	Text  string
	Parts []contentPart
}

// MarshalJSON encodes the parts when there are any and the text otherwise.
func (c assistantContent) MarshalJSON() ([]byte, error) {
	if c.Parts != nil {
		return json.Marshal(c.Parts)
	}

	return json.Marshal(c.Text)
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

// chatResponse is the subset of a chat completion that the model reads.
type chatResponse struct {
	ID      string       `json:"id"`
	Created float64      `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *Usage       `json:"usage"`
}

type chatChoice struct {
	Message      responseMessage `json:"message"`
	FinishReason *string         `json:"finish_reason"`
}

type responseMessage struct {
	Content   responseContent    `json:"content"`
	ToolCalls []responseToolCall `json:"tool_calls"`
}

type responseToolCall struct {
	ID       string       `json:"id"`
	Function functionCall `json:"function"`
}

// chatChunk is one streamed chat completion event.
type chatChunk struct {
	ID      string            `json:"id"`
	Created float64           `json:"created"`
	Model   string            `json:"model"`
	Choices []chatChunkChoice `json:"choices"`
	Usage   *Usage            `json:"usage"`
}

type chatChunkChoice struct {
	Delta        chatDelta `json:"delta"`
	FinishReason *string   `json:"finish_reason"`
}

type chatDelta struct {
	Content   responseContent `json:"content"`
	ToolCalls []deltaToolCall `json:"tool_calls"`
}

type deltaToolCall struct {
	Index    *int          `json:"index"`
	ID       *string       `json:"id"`
	Function deltaFunction `json:"function"`
}

type deltaFunction struct {
	Name      *string `json:"name"`
	Arguments *string `json:"arguments"`
}

// responseContent is message content: absent, a string, or typed parts.
type responseContent struct {
	Text  string
	Parts []responsePart
}

// UnmarshalJSON accepts null, a string or an array of parts.
func (c *responseContent) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	*c = responseContent{}

	switch {
	case bytes.Equal(trimmed, []byte("null")):
		return nil

	case bytes.HasPrefix(trimmed, []byte(`"`)):
		return json.Unmarshal(trimmed, &c.Text)

	case bytes.HasPrefix(trimmed, []byte("[")):
		return json.Unmarshal(trimmed, &c.Parts)

	default:
		return fmt.Errorf("%w: message content must be a string or an array", model.ErrInvalidResponse)
	}
}

// responsePart is a text, image, reference or thinking part. Image and
// reference parts are not read.
type responsePart struct {
	Type     contentPartType `json:"type"`
	Text     string          `json:"text"`
	Thinking []thinkingText  `json:"thinking"`
}

// UnmarshalJSON rejects part types that Mistral does not document.
func (p *responsePart) UnmarshalJSON(data []byte) error {
	type plain responsePart

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	switch decoded.Type {
	case contentText, contentImageURL, contentThinking, "reference":
		*p = responsePart(decoded)

		return nil

	default:
		return fmt.Errorf("%w: unsupported content part type %q", model.ErrInvalidResponse, decoded.Type)
	}
}

// joinThinking concatenates the text items of a thinking part.
func joinThinking(thinking []thinkingText) string {
	var text strings.Builder
	for _, item := range thinking {
		if item.Type == "text" {
			text.WriteString(item.Text)
		}
	}

	return text.String()
}

// text returns the joined text of the content, or an empty string.
func (c responseContent) text() string {
	if c.Parts == nil {
		return c.Text
	}

	var text strings.Builder
	for _, part := range c.Parts {
		if part.Type == contentText {
			text.WriteString(part.Text)
		}
	}

	return text.String()
}
