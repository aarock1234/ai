package cohere

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// chatRequest is a chat request body.
type chatRequest struct {
	Model string `json:"model"`

	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	MaxTokens        *int            `json:"max_tokens,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	P                *float64        `json:"p,omitempty"`
	K                *int            `json:"k,omitempty"`
	Seed             *int            `json:"seed,omitempty"`
	StopSequences    []string        `json:"stop_sequences,omitempty"`
	ResponseFormat   *responseFormat `json:"response_format,omitempty"`

	Messages   []chatMessage `json:"messages"`
	Tools      []tool        `json:"tools,omitempty"`
	ToolChoice toolChoice    `json:"tool_choice,omitempty"`

	// Documents are sent for retrieval-augmented generation.
	Documents []document `json:"documents,omitempty"`

	Thinking *thinking `json:"thinking,omitempty"`

	Stream bool `json:"stream,omitempty"`
}

type responseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

type thinking struct {
	Type        ThinkingType `json:"type"`
	TokenBudget *int         `json:"token_budget,omitempty"`
}

type document struct {
	Data documentData `json:"data"`
}

type documentData struct {
	Text  string `json:"text"`
	Title string `json:"title,omitempty"`
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
	Role string `json:"role"`

	// Content is absent when the message has tool calls.
	Content *string `json:"content,omitempty"`

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

// userContent is plain text, or parts when the message has images.
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

// contentPart is a text or image part.
type contentPart struct {
	Type     string    `json:"type"`
	Text     *string   `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
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

// chatResponse is a chat response.
type chatResponse struct {
	GenerationID *string         `json:"generation_id"`
	Message      responseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
	Usage        *Usage          `json:"usage"`
}

type responseMessage struct {
	Content   []responseContent  `json:"content"`
	ToolCalls []messageToolCall  `json:"tool_calls"`
	Citations []responseCitation `json:"citations"`
}

// responseContent is a text or thinking item.
type responseContent struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Thinking string `json:"thinking"`
}

// UnmarshalJSON rejects undocumented content types.
func (c *responseContent) UnmarshalJSON(data []byte) error {
	type plain responseContent

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	if decoded.Type != "text" && decoded.Type != "thinking" {
		return fmt.Errorf("%w: unsupported content type %q", model.ErrInvalidResponse, decoded.Type)
	}

	*c = responseContent(decoded)

	return nil
}

type responseCitation struct {
	Start   int              `json:"start"`
	End     int              `json:"end"`
	Text    string           `json:"text"`
	Sources []CitationSource `json:"sources"`
	Type    string           `json:"type"`
}

// CitationSource is a source of a citation.
type CitationSource struct {
	Type     string           `json:"type,omitempty"`
	ID       string           `json:"id,omitempty"`
	Document CitationDocument `json:"document"`
}

// CitationDocument is the cited document.
type CitationDocument struct {
	ID    string `json:"id,omitempty"`
	Text  string `json:"text"`
	Title string `json:"title"`
}

// Stream event types.
const (
	eventCitationStart = "citation-start"
	eventCitationEnd   = "citation-end"
	eventContentStart  = "content-start"
	eventContentDelta  = "content-delta"
	eventContentEnd    = "content-end"
	eventMessageStart  = "message-start"
	eventMessageEnd    = "message-end"
	eventToolPlanDelta = "tool-plan-delta"
	eventToolCallStart = "tool-call-start"
	eventToolCallDelta = "tool-call-delta"
	eventToolCallEnd   = "tool-call-end"
)

// chatEvent is one stream event. The fields of its type are set.
type chatEvent struct {
	Type  string      `json:"type"`
	Index int         `json:"index"`
	ID    *string     `json:"id"`
	Delta *eventDelta `json:"delta"`
}

type eventDelta struct {
	Message      *deltaMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
	Usage        *Usage        `json:"usage"`
}

type deltaMessage struct {
	Content   *deltaContent  `json:"content"`
	ToolCalls *deltaToolCall `json:"tool_calls"`
	ToolPlan  *string        `json:"tool_plan"`
}

// deltaContent is streamed text or thinking. Thinking is set only for
// reasoning.
type deltaContent struct {
	Type     string  `json:"type"`
	Text     string  `json:"text"`
	Thinking *string `json:"thinking"`
}

type deltaToolCall struct {
	ID       string        `json:"id"`
	Function deltaFunction `json:"function"`
}

type deltaFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}
