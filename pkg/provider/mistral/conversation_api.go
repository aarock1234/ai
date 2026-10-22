package mistral

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// conversationRequest starts an unstored conversation.
type conversationRequest struct {
	Model          string              `json:"model"`
	Store          bool                `json:"store"`
	Inputs         []conversationInput `json:"inputs"`
	Instructions   string              `json:"instructions,omitempty"`
	Tools          []tool              `json:"tools,omitempty"`
	CompletionArgs completionArgs      `json:"completion_args"`
	Stream         bool                `json:"stream,omitempty"`
}

type completionArgs struct {
	completionSettings

	ToolChoice *toolChoice `json:"tool_choice,omitempty"`
}

// conversationInput is a messageInput, functionCallInput or
// functionResultInput.
type conversationInput interface {
	isConversationInput()
}

type messageInput struct {
	Type    string        `json:"type"`
	Role    string        `json:"role"`
	Content []contentPart `json:"content"`
}

type functionCallInput struct {
	Type       string `json:"type"`
	ToolCallID string `json:"tool_call_id"`
	Name       string `json:"name"`
	Arguments  string `json:"arguments"`
}

type functionResultInput struct {
	Type       string `json:"type"`
	ToolCallID string `json:"tool_call_id"`
	Result     string `json:"result"`
}

func (messageInput) isConversationInput()        {}
func (functionCallInput) isConversationInput()   {}
func (functionResultInput) isConversationInput() {}

// conversationResponse is a completed conversation turn.
type conversationResponse struct {
	ConversationID string               `json:"conversation_id"`
	Outputs        []conversationOutput `json:"outputs"`
	Usage          *Usage               `json:"usage"`
}

// conversationOutput is one output entry. Exactly one field is set.
type conversationOutput struct {
	ToolExecution *toolExecution
	FunctionCall  *functionCallOutput
	Message       *messageOutput
}

// typeHeader reads the discriminator of a tagged union member.
type typeHeader struct {
	Type string `json:"type"`
}

// UnmarshalJSON decodes the entry by its type.
func (o *conversationOutput) UnmarshalJSON(data []byte) error {
	var header typeHeader
	if err := json.Unmarshal(data, &header); err != nil {
		return err
	}

	*o = conversationOutput{}

	switch header.Type {
	case "tool.execution":
		o.ToolExecution = &toolExecution{}

		return json.Unmarshal(data, o.ToolExecution)

	case "function.call":
		o.FunctionCall = &functionCallOutput{}

		return json.Unmarshal(data, o.FunctionCall)

	case "message.output":
		o.Message = &messageOutput{}

		return json.Unmarshal(data, o.Message)

	default:
		return fmt.Errorf("%w: unsupported conversation output type %q", model.ErrInvalidResponse, header.Type)
	}
}

// toolExecution is a built-in tool call that Mistral ran.
type toolExecution struct {
	ID        string  `json:"id"`
	Name      string  `json:"name"`
	Arguments string  `json:"arguments"`
	Function  *string `json:"function"`

	// Info is the tool's output, an arbitrary JSON object.
	Info json.RawMessage `json:"info"`
}

// functionCallOutput is a call of a function tool.
type functionCallOutput struct {
	ToolCallID string        `json:"tool_call_id"`
	Name       string        `json:"name"`
	Arguments  callArguments `json:"arguments"`
}

// callArguments are function arguments sent either as JSON text or as a JSON
// object. Text keeps the JSON text in both cases.
type callArguments struct {
	Text string
}

// UnmarshalJSON accepts a string or an object.
func (a *callArguments) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)

	switch {
	case bytes.HasPrefix(trimmed, []byte(`"`)):
		return json.Unmarshal(trimmed, &a.Text)

	case bytes.HasPrefix(trimmed, []byte("{")):
		compacted, err := json.Marshal(json.RawMessage(trimmed))
		if err != nil {
			return err
		}

		a.Text = string(compacted)

		return nil

	default:
		return fmt.Errorf("%w: function call arguments must be a string or an object", model.ErrInvalidResponse)
	}
}

type messageOutput struct {
	Content conversationContent `json:"content"`
	Model   *string             `json:"model"`
}

// conversationContent is a string or a list of parts.
type conversationContent struct {
	Parts []conversationPart
}

// UnmarshalJSON accepts a string, which becomes one text part, or an array.
func (c *conversationContent) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)

	if bytes.HasPrefix(trimmed, []byte(`"`)) {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return err
		}

		c.Parts = []conversationPart{
			{
				Type: conversationText,
				Text: text,
			},
		}

		return nil
	}

	return json.Unmarshal(trimmed, &c.Parts)
}

// conversationPartType is the type of a conversation content part.
type conversationPartType string

const (
	conversationText          conversationPartType = "text"
	conversationThinking      conversationPartType = "thinking"
	conversationToolReference conversationPartType = "tool_reference"
)

// conversationPart is a text, thinking or tool reference part.
type conversationPart struct {
	Type     conversationPartType `json:"type"`
	Text     string               `json:"text"`
	Thinking []thinkingText       `json:"thinking"`

	// Title and URL describe a tool reference, such as a search result.
	Title *string `json:"title"`
	URL   *string `json:"url"`
}

// UnmarshalJSON rejects undocumented part types.
func (p *conversationPart) UnmarshalJSON(data []byte) error {
	type plain conversationPart

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	switch decoded.Type {
	case conversationText, conversationThinking, conversationToolReference:
		*p = conversationPart(decoded)

		return nil

	default:
		return fmt.Errorf("%w: unsupported conversation content type %q", model.ErrInvalidResponse, decoded.Type)
	}
}

// Conversation stream event types.
const (
	eventResponseStarted      = "conversation.response.started"
	eventResponseDone         = "conversation.response.done"
	eventResponseError        = "conversation.response.error"
	eventMessageOutputDelta   = "message.output.delta"
	eventFunctionCallDelta    = "function.call.delta"
	eventToolExecutionStarted = "tool.execution.started"
	eventToolExecutionDelta   = "tool.execution.delta"
	eventToolExecutionDone    = "tool.execution.done"
)

type responseStartedEvent struct {
	ConversationID string  `json:"conversation_id"`
	CreatedAt      *string `json:"created_at"`
}

type responseDoneEvent struct {
	Usage *Usage `json:"usage"`
}

// ConversationError is an error event of a conversation stream.
type ConversationError struct {
	Message string `json:"message"`
	Code    int    `json:"code"`
}

// Error returns the provider's message and code.
func (e *ConversationError) Error() string {
	return fmt.Sprintf("conversation error %d: %s", e.Code, e.Message)
}

type messageOutputDeltaEvent struct {
	ID           string       `json:"id"`
	ContentIndex *int         `json:"content_index"`
	Content      deltaContent `json:"content"`
	Model        *string      `json:"model"`
}

// deltaContent is a string or a single part.
type deltaContent struct {
	Part conversationPart
}

// UnmarshalJSON accepts a string, which becomes a text part, or a part.
func (c *deltaContent) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)

	if bytes.HasPrefix(trimmed, []byte(`"`)) {
		c.Part = conversationPart{
			Type: conversationText,
		}

		return json.Unmarshal(trimmed, &c.Part.Text)
	}

	return json.Unmarshal(trimmed, &c.Part)
}

type functionCallDeltaEvent struct {
	ToolCallID  string `json:"tool_call_id"`
	Name        string `json:"name"`
	Arguments   string `json:"arguments"`
	OutputIndex *int   `json:"output_index"`
}

type toolExecutionDeltaEvent struct {
	ID        string  `json:"id"`
	Name      string  `json:"name"`
	Arguments string  `json:"arguments"`
	Function  *string `json:"function"`
}

type toolExecutionDoneEvent struct {
	ID       string          `json:"id"`
	Name     string          `json:"name"`
	Function *string         `json:"function"`
	Info     json.RawMessage `json:"info"`
}
