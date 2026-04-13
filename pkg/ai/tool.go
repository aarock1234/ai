package ai

import (
	"context"
	"encoding/json"
)

// Tool defines a tool that the model can invoke during generation.
// Provide an Execute function to enable the automatic tool loop in
// GenerateText and StreamText.
type Tool struct {
	// Name is the tool identifier the model uses to reference it.
	Name string

	// Description tells the model what the tool does and when to use it.
	Description string

	// InputSchema is the JSON Schema describing the tool's input
	// parameters. Use SchemaFor[T]() to generate this from a Go type.
	InputSchema json.RawMessage

	// Strict requests strict schema adherence when the provider supports it.
	Strict bool

	// Execute is called when the model invokes this tool. It receives
	// the raw JSON arguments and returns a string result. If nil, the
	// tool loop will not execute this tool automatically.
	Execute func(ctx context.Context, input json.RawMessage) (string, error)
}

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
	ToolCallID string
	ToolName   string
	Args       json.RawMessage
}

// ToolResult carries the outcome of a tool execution back to the model.
type ToolResult struct {
	ToolCallID string
	ToolName   string
	Content    string
	IsError    bool
}
