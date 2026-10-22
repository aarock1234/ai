package huggingface

import (
	"bytes"
	"encoding/json"
	"fmt"
)

type responsesRequest struct {
	Model           string            `json:"model"`
	Input           []inputMessage    `json:"input"`
	Temperature     *float64          `json:"temperature,omitempty"`
	TopP            *float64          `json:"top_p,omitempty"`
	MaxOutputTokens *int              `json:"max_output_tokens,omitempty"`
	Text            *textConfig       `json:"text,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
	Instructions    *string           `json:"instructions,omitempty"`
	Tools           []functionTool    `json:"tools,omitempty"`
	ToolChoice      *toolChoice       `json:"tool_choice,omitempty"`
	Reasoning       *reasoningConfig  `json:"reasoning,omitempty"`
	Stream          bool              `json:"stream"`
}

type textConfig struct {
	Format textFormat `json:"format"`
}

type textFormat struct {
	Type        string          `json:"type"`
	Strict      bool            `json:"strict"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema"`
}

type reasoningConfig struct {
	Effort string `json:"effort"`
}

// inputMessage is a systemInput, userInput or assistantInput.
type inputMessage interface {
	isInputMessage()
}

type systemInput struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type userInput struct {
	Role    string      `json:"role"`
	Content []inputPart `json:"content"`
}

type assistantInput struct {
	Role    string       `json:"role"`
	Content []outputText `json:"content"`
}

// inputPart is an inputText or inputImage.
type inputPart interface {
	isInputPart()
}

type inputText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type inputImage struct {
	Type     string `json:"type"`
	ImageURL string `json:"image_url"`
}

type outputText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func (systemInput) isInputMessage()    {}
func (userInput) isInputMessage()      {}
func (assistantInput) isInputMessage() {}

func (inputText) isInputPart()  {}
func (inputImage) isInputPart() {}

type functionTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// toolChoice is "auto", "required" or a named function.
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
		Type: "function",
		Function: namedToolFunction{
			Name: c.function,
		},
	})
}

type responsesResponse struct {
	ID                string             `json:"id"`
	Model             string             `json:"model"`
	CreatedAt         float64            `json:"created_at"`
	Status            string             `json:"status"`
	Error             *responseError     `json:"error"`
	IncompleteDetails *incompleteDetails `json:"incomplete_details"`
	Usage             *responsesUsage    `json:"usage"`
	Output            []outputItem       `json:"output"`
}

type responseError struct {
	Message string          `json:"message"`
	Code    json.RawMessage `json:"code,omitempty"`
}

type incompleteDetails struct {
	Reason string `json:"reason"`
}

// responsesUsage is the token usage of a response. The raw object is kept as
// the usage Raw value.
type responsesUsage struct {
	InputTokens         int                  `json:"input_tokens"`
	InputTokensDetails  *inputTokensDetails  `json:"input_tokens_details"`
	OutputTokens        int                  `json:"output_tokens"`
	OutputTokensDetails *outputTokensDetails `json:"output_tokens_details"`
	TotalTokens         int                  `json:"total_tokens"`

	raw json.RawMessage
}

type inputTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type outputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
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
	itemMessage      = "message"
	itemReasoning    = "reasoning"
	itemFunctionCall = "function_call"
	itemMCPCall      = "mcp_call"
	itemMCPListTools = "mcp_list_tools"
)

// outputItem is one response output item. The fields of each item type are
// decoded according to Type; unknown types keep only their type.
type outputItem struct {
	Type string

	Message      *messageItem
	Reasoning    *reasoningItem
	FunctionCall *functionCallItem
	MCPCall      *mcpCallItem
	MCPListTools *mcpListToolsItem
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
	case itemMCPCall:
		return decodeItem(data, &i.MCPCall)
	case itemMCPListTools:
		return decodeItem(data, &i.MCPListTools)
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

type messageItem struct {
	ID      string           `json:"id"`
	Role    string           `json:"role"`
	Content []messageContent `json:"content"`
}

type messageContent struct {
	Type        string       `json:"type"`
	Text        string       `json:"text"`
	Annotations []annotation `json:"annotations"`
}

type annotation struct {
	URL   string `json:"url"`
	Title string `json:"title"`
}

type reasoningItem struct {
	ID      string             `json:"id"`
	Content []reasoningContent `json:"content"`
}

type reasoningContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type functionCallItem struct {
	ID        string  `json:"id"`
	CallID    string  `json:"call_id"`
	Name      string  `json:"name"`
	Arguments string  `json:"arguments"`
	Output    *string `json:"output"`
}

type mcpCallItem struct {
	ID        string  `json:"id"`
	Name      string  `json:"name"`
	Arguments string  `json:"arguments"`
	Output    *string `json:"output"`
}

type mcpListToolsItem struct {
	ID          string            `json:"id"`
	ServerLabel string            `json:"server_label"`
	Tools       []json.RawMessage `json:"tools"`
}

// streamChunk is one streamed event. The fields of each event type share
// their JSON names.
type streamChunk struct {
	Type     string             `json:"type"`
	Item     *outputItem        `json:"item"`
	ItemID   string             `json:"item_id"`
	Delta    string             `json:"delta"`
	Response *responsesResponse `json:"response"`

	// Error is the error object of "error" events, which may also carry
	// the message and code at the top level.
	Error   *responseError  `json:"error"`
	Message *string         `json:"message"`
	Code    json.RawMessage `json:"code"`
}
