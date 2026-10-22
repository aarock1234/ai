package model

import "encoding/json"

// Tool is a tool the model may call: a FunctionTool or a ProviderTool.
type Tool interface {
	// ToolName returns the name the model uses to call the tool.
	ToolName() string
}

// FunctionTool is a tool described by a JSON Schema and executed by the
// application.
type FunctionTool struct {
	Name string

	// Description is optional guidance for the model.
	Description string

	// InputSchema is the JSON Schema of the arguments object.
	InputSchema json.RawMessage

	// InputExamples are example argument objects for providers that accept
	// them.
	InputExamples []json.RawMessage

	// Strict requests strict schema adherence. Nil leaves the provider
	// default.
	Strict *bool

	ProviderOptions ProviderOptions
}

// ToolName returns the function name.
func (t FunctionTool) ToolName() string {
	return t.Name
}

// ProviderTool is a tool implemented by the provider, such as web search.
type ProviderTool struct {
	// ID has the form "provider.tool_name", for example
	// "anthropic.web_search_20250305".
	ID string

	// Name is the name the application uses for the tool.
	Name string

	// Args is the JSON object of provider-specific configuration.
	Args json.RawMessage
}

// ToolName returns the application name.
func (t ProviderTool) ToolName() string {
	return t.Name
}

// ToolChoiceType selects how the model chooses tools.
type ToolChoiceType string

// Tool choice types.
const (
	// ToolChoiceAuto lets the model decide whether to call a tool.
	ToolChoiceAuto ToolChoiceType = "auto"

	// ToolChoiceNone prevents tool calls.
	ToolChoiceNone ToolChoiceType = "none"

	// ToolChoiceRequired requires at least one tool call.
	ToolChoiceRequired ToolChoiceType = "required"

	// ToolChoiceTool requires a call to the tool named by ToolName.
	ToolChoiceTool ToolChoiceType = "tool"
)

// ToolChoice controls tool selection.
type ToolChoice struct {
	Type ToolChoiceType

	// ToolName is set only when Type is ToolChoiceTool.
	ToolName string
}
