package openresponses

import (
	"bytes"
	"encoding/json"
	"fmt"
)

type responsesRequest struct {
	Model            string           `json:"model"`
	Input            []inputItem      `json:"input"`
	Instructions     *string          `json:"instructions,omitempty"`
	MaxOutputTokens  *int             `json:"max_output_tokens,omitempty"`
	Temperature      *float64         `json:"temperature,omitempty"`
	TopP             *float64         `json:"top_p,omitempty"`
	PresencePenalty  *float64         `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64         `json:"frequency_penalty,omitempty"`
	Reasoning        *reasoningParam  `json:"reasoning,omitempty"`
	Tools            []requestTool    `json:"tools,omitempty"`
	ToolChoice       *toolChoiceParam `json:"tool_choice,omitempty"`
	Text             *textParam       `json:"text,omitempty"`
	Stream           bool             `json:"stream,omitempty"`
}

type reasoningParam struct {
	Effort  *string           `json:"effort,omitempty"`
	Summary *ReasoningSummary `json:"summary,omitempty"`
}

type textParam struct {
	Format textFormat `json:"format"`
}

type textFormat struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// requestTool is a function tool or a custom tool. Function tools set
// Parameters; custom tools may set Format.
type requestTool struct {
	Type        string            `json:"type"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Parameters  json.RawMessage   `json:"parameters,omitempty"`
	Strict      *bool             `json:"strict,omitempty"`
	Format      *CustomToolFormat `json:"format,omitempty"`
}

// toolChoiceParam is "auto", "none", "required" or a named function or
// custom tool.
type toolChoiceParam struct {
	mode     string
	toolType string
	name     string
}

type namedToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

// MarshalJSON encodes a mode as a string and a named tool as an object.
func (c toolChoiceParam) MarshalJSON() ([]byte, error) {
	if c.mode != "" {
		return json.Marshal(c.mode)
	}

	return json.Marshal(namedToolChoice{
		Type: c.toolType,
		Name: c.name,
	})
}

// inputItem is one input item: a message, reasoning item, function or
// custom tool call, or a tool output.
type inputItem interface {
	isInputItem()
}

type userMessage struct {
	Type    string      `json:"type"`
	Role    string      `json:"role"`
	Content []inputPart `json:"content"`
}

// inputPart is a text, image or file part. Each type sets its own fields.
type inputPart struct {
	Type     string      `json:"type"`
	Text     *string     `json:"text,omitempty"`
	ImageURL string      `json:"image_url,omitempty"`
	Detail   ImageDetail `json:"detail,omitempty"`
	Filename string      `json:"filename,omitempty"`
	FileData string      `json:"file_data,omitempty"`
	FileURL  string      `json:"file_url,omitempty"`
}

// assistantMessage content is a string for easy input messages and output
// parts otherwise.
type assistantMessage struct {
	ID      string           `json:"id,omitempty"`
	Type    string           `json:"type"`
	Status  string           `json:"status,omitempty"`
	Role    string           `json:"role"`
	Content assistantContent `json:"content"`
}

type assistantContent struct {
	text  string
	parts []outputTextParam
}

// MarshalJSON encodes the parts, or the text when there are no parts.
func (c assistantContent) MarshalJSON() ([]byte, error) {
	if c.parts != nil {
		return json.Marshal(c.parts)
	}

	return json.Marshal(c.text)
}

// outputTextParam is replayed output text. Strict output items always carry
// annotations and logprobs.
type outputTextParam struct {
	Type        string             `json:"type"`
	Text        string             `json:"text"`
	Annotations *[]URLCitation     `json:"annotations,omitempty"`
	Logprobs    *[]json.RawMessage `json:"logprobs,omitempty"`
}

type reasoningItem struct {
	ID               *string         `json:"id,omitempty"`
	Type             string          `json:"type"`
	Summary          []SummaryText   `json:"summary"`
	Content          []ReasoningText `json:"content,omitempty"`
	EncryptedContent *string         `json:"encrypted_content,omitempty"`
}

type functionCallItem struct {
	Type      string  `json:"type"`
	ID        *string `json:"id,omitempty"`
	CallID    string  `json:"call_id"`
	Name      string  `json:"name"`
	Arguments string  `json:"arguments"`
}

type customToolCallItem struct {
	Type   string  `json:"type"`
	ID     *string `json:"id,omitempty"`
	CallID string  `json:"call_id"`
	Name   string  `json:"name"`
	Input  string  `json:"input"`
}

type toolOutputItem struct {
	Type   string     `json:"type"`
	CallID string     `json:"call_id"`
	Output toolOutput `json:"output"`
}

// toolOutput is text, or content parts for multipart results.
type toolOutput struct {
	text  string
	parts []inputPart
}

// MarshalJSON encodes the parts, or the text when there are no parts.
func (o toolOutput) MarshalJSON() ([]byte, error) {
	if o.parts != nil {
		return json.Marshal(o.parts)
	}

	return json.Marshal(o.text)
}

func (userMessage) isInputItem()        {}
func (assistantMessage) isInputItem()   {}
func (*reasoningItem) isInputItem()     {}
func (functionCallItem) isInputItem()   {}
func (customToolCallItem) isInputItem() {}
func (toolOutputItem) isInputItem()     {}

type responsesResponse struct {
	ID                string             `json:"id"`
	CreatedAt         *float64           `json:"created_at"`
	Model             string             `json:"model"`
	Status            string             `json:"status"`
	Error             *ResponseError     `json:"error"`
	IncompleteDetails *incompleteDetails `json:"incomplete_details"`
	Output            []outputItem       `json:"output"`
	Usage             *responsesUsage    `json:"usage"`
}

type incompleteDetails struct {
	Reason *string `json:"reason"`
}

// UnmarshalJSON decodes the error and keeps the raw object.
func (e *ResponseError) UnmarshalJSON(data []byte) error {
	type plain ResponseError

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*e = ResponseError(decoded)
	e.Raw = bytes.Clone(data)

	return nil
}

// responsesUsage is the token usage of a response. The raw object is kept as
// the usage Raw value.
type responsesUsage struct {
	InputTokens         *int                 `json:"input_tokens"`
	OutputTokens        *int                 `json:"output_tokens"`
	TotalTokens         *int                 `json:"total_tokens"`
	InputTokensDetails  *inputTokensDetails  `json:"input_tokens_details"`
	OutputTokensDetails *outputTokensDetails `json:"output_tokens_details"`

	raw json.RawMessage
}

type inputTokensDetails struct {
	CachedTokens     *int `json:"cached_tokens"`
	CacheWriteTokens *int `json:"cache_write_tokens"`
}

type outputTokensDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

// UnmarshalJSON decodes the usage and keeps the raw object.
func (u *responsesUsage) UnmarshalJSON(data []byte) error {
	type plain responsesUsage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = responsesUsage(decoded)
	u.raw = bytes.Clone(data)

	return nil
}

// Output item types.
const (
	itemMessage        = "message"
	itemReasoning      = "reasoning"
	itemFunctionCall   = "function_call"
	itemCustomToolCall = "custom_tool_call"
)

// outputItem is one response output item. The fields of handled item types
// are decoded according to Type; other items keep only their type.
type outputItem struct {
	Type string

	Message        *messageOutput
	Reasoning      *reasoningOutput
	FunctionCall   *functionCallOutput
	CustomToolCall *customToolCallOutput
}

type itemHeader struct {
	Type string `json:"type"`
}

// UnmarshalJSON decodes the item variant named by its type.
func (i *outputItem) UnmarshalJSON(data []byte) error {
	var header itemHeader
	if err := json.Unmarshal(data, &header); err != nil {
		return err
	}

	*i = outputItem{
		Type: header.Type,
	}

	switch header.Type {
	case itemMessage:
		return decodeItem(data, &i.Message)
	case itemReasoning:
		return decodeItem(data, &i.Reasoning)
	case itemFunctionCall:
		return decodeItem(data, &i.FunctionCall)
	case itemCustomToolCall:
		return decodeItem(data, &i.CustomToolCall)
	default:
		return nil
	}
}

func decodeItem[T any](data []byte, target **T) error {
	var item T
	if err := json.Unmarshal(data, &item); err != nil {
		return fmt.Errorf("decode output item: %w", err)
	}

	*target = &item

	return nil
}

type messageOutput struct {
	ID      string              `json:"id"`
	Content []outputTextContent `json:"content"`
}

type outputTextContent struct {
	Type        string          `json:"type"`
	Text        string          `json:"text"`
	Annotations json.RawMessage `json:"annotations"`
}

type reasoningOutput struct {
	ID               string        `json:"id"`
	Content          []textContent `json:"content"`
	Summary          []textContent `json:"summary"`
	EncryptedContent *string       `json:"encrypted_content"`
}

type textContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type functionCallOutput struct {
	ID        string `json:"id"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type customToolCallOutput struct {
	ID     string `json:"id"`
	CallID string `json:"call_id"`
	Name   string `json:"name"`
	Input  string `json:"input"`
}

// streamChunk is one streamed event. The fields of each event type share
// their JSON names.
type streamChunk struct {
	Type      string             `json:"type"`
	Item      *outputItem        `json:"item"`
	ItemID    string             `json:"item_id"`
	Delta     string             `json:"delta"`
	Arguments string             `json:"arguments"`
	Input     string             `json:"input"`
	Response  *responsesResponse `json:"response"`
	Error     *ResponseError     `json:"error"`
}

// errorEnvelope is the default error body.
type errorEnvelope struct {
	Error *errorMessage `json:"error"`
}

type errorMessage struct {
	Message string `json:"message"`
}

// parseError reads the message of a {"error": {"message": ...}} body.
func parseError(body []byte) string {
	var envelope errorEnvelope
	if err := json.Unmarshal(body, &envelope); err != nil || envelope.Error == nil {
		return ""
	}

	return envelope.Error.Message
}
