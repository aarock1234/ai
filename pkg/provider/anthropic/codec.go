package anthropic

import "encoding/json"

// MessageRequest is the Anthropic Messages API request body.
type MessageRequest struct {
	Model         string            `json:"model"`
	System        string            `json:"system,omitempty"`
	Messages      []message         `json:"messages"`
	MaxTokens     int               `json:"max_tokens"`
	Temperature   *float64          `json:"temperature,omitempty"`
	TopP          *float64          `json:"top_p,omitempty"`
	StopSequences []string          `json:"stop_sequences,omitempty"`
	Metadata      map[string]string `json:"metadata,omitempty"`
	Tools         []toolDefinition  `json:"tools,omitempty"`
	ToolChoice    *toolChoice       `json:"tool_choice,omitempty"`
	Stream        bool              `json:"stream,omitempty"`
}

type message struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

type contentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Source    *imageSource    `json:"source,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

type imageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type toolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
	Strict      bool            `json:"strict,omitempty"`
}

type toolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

// MessageResponse is the Anthropic Messages API response body.
type MessageResponse struct {
	ID           string            `json:"id"`
	Type         string            `json:"type"`
	Role         string            `json:"role"`
	Content      []contentBlock    `json:"content"`
	Model        string            `json:"model"`
	StopReason   string            `json:"stop_reason"`
	StopSequence *string           `json:"stop_sequence,omitempty"`
	Usage        wireUsage         `json:"usage"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

type wireUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type streamMessageStartEvent struct {
	Type    string          `json:"type"`
	Message MessageResponse `json:"message"`
}

type streamContentBlockStartEvent struct {
	Type         string       `json:"type"`
	Index        int          `json:"index"`
	ContentBlock contentBlock `json:"content_block"`
}

type streamContentBlockDeltaEvent struct {
	Type  string            `json:"type"`
	Index int               `json:"index"`
	Delta contentBlockDelta `json:"delta"`
}

type streamContentBlockStopEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type streamMessageDeltaEvent struct {
	Type  string       `json:"type"`
	Delta messageDelta `json:"delta"`
	Usage *wireUsage   `json:"usage,omitempty"`
}

type messageDelta struct {
	StopReason   string  `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

type contentBlockDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
}

type apiErrorResponse struct {
	Error *apiErrorBody `json:"error,omitempty"`
}

type apiErrorBody struct {
	Type    string `json:"type,omitempty"`
	Message string `json:"message,omitempty"`
}
