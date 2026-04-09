// Package openai provides an OpenAI language model provider and shared
// codec for OpenAI-compatible chat completion APIs. Providers that use
// the OpenAI wire format (OpenAI, OpenRouter, Groq, Together, DeepSeek,
// etc.) import this package for request encoding, response decoding,
// and streaming.
package openai

import "encoding/json"

// ChatRequest is the OpenAI chat completions request body.
type ChatRequest struct {
	Model          string            `json:"model"`
	Messages       []chatMessage     `json:"messages"`
	Temperature    *float64          `json:"temperature,omitempty"`
	MaxTokens      *int              `json:"max_completion_tokens,omitempty"`
	TopP           *float64          `json:"top_p,omitempty"`
	Stop           []string          `json:"stop,omitempty"`
	Tools          []chatTool        `json:"tools,omitempty"`
	ToolChoice     any               `json:"tool_choice,omitempty"`
	ResponseFormat *responseFormat   `json:"response_format,omitempty"`
	Metadata       map[string]string `json:"metadata,omitempty"`
	Stream         bool              `json:"stream,omitempty"`
	StreamOptions  *streamOptions    `json:"stream_options,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// chatMessage is a single message in the OpenAI wire format.
type chatMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
}

// wireToolCall is a tool call in the OpenAI wire format.
type wireToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function wireFunction `json:"function"`
}

// wireFunction describes the function invocation in a tool call.
type wireFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// chatTool describes a tool available to the model.
type chatTool struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

// toolFunction describes a function tool's schema.
type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// responseFormat requests a specific output structure.
type responseFormat struct {
	Type       string          `json:"type"`
	JSONSchema *jsonSchemaSpec `json:"json_schema,omitempty"`
}

// jsonSchemaSpec is the schema specification for structured output.
type jsonSchemaSpec struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

// ChatResponse is the OpenAI chat completions response.
type ChatResponse struct {
	ID       string            `json:"id"`
	Choices  []chatChoice      `json:"choices"`
	Usage    wireUsage         `json:"usage"`
	Model    string            `json:"model"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// chatChoice is a single completion choice.
type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// wireUsage tracks token consumption.
type wireUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// chatStreamChunk is a single chunk in a streaming response.
type chatStreamChunk struct {
	ID       string              `json:"id"`
	Choices  []streamChunkChoice `json:"choices"`
	Usage    *wireUsage          `json:"usage,omitempty"`
	Model    string              `json:"model,omitempty"`
	Metadata map[string]string   `json:"metadata,omitempty"`
}

// streamChunkChoice is a single choice delta in a streaming chunk.
type streamChunkChoice struct {
	Index        int       `json:"index"`
	Delta        chatDelta `json:"delta"`
	FinishReason *string   `json:"finish_reason"`
}

// chatDelta is the incremental content in a streaming chunk.
type chatDelta struct {
	Role      string         `json:"role,omitempty"`
	Content   *string        `json:"content,omitempty"`
	ToolCalls []wireToolCall `json:"tool_calls,omitempty"`
}

// toolChoiceSpec is used when tool_choice references a specific tool.
type toolChoiceSpec struct {
	Type     string                 `json:"type"`
	Function toolChoiceSpecFunction `json:"function"`
}

// toolChoiceSpecFunction names the specific tool to call.
type toolChoiceSpecFunction struct {
	Name string `json:"name"`
}
