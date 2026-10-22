package model

import "encoding/json"

// ToolResultOutput is the model-visible outcome of a tool call: TextOutput,
// JSONOutput, ExecutionDeniedOutput, ErrorTextOutput, ErrorJSONOutput or
// ContentOutput.
type ToolResultOutput interface {
	isToolResultOutput()
}

// TextOutput is a successful text result.
type TextOutput struct {
	Value           string
	ProviderOptions ProviderOptions
}

// JSONOutput is a successful JSON result.
type JSONOutput struct {
	Value           json.RawMessage
	ProviderOptions ProviderOptions
}

// ExecutionDeniedOutput reports that the user denied the tool call.
type ExecutionDeniedOutput struct {
	// Reason optionally explains the denial to the model.
	Reason          string
	ProviderOptions ProviderOptions
}

// ErrorTextOutput is a failed result described as text.
type ErrorTextOutput struct {
	Value           string
	ProviderOptions ProviderOptions
}

// ErrorJSONOutput is a failed result described as JSON.
type ErrorJSONOutput struct {
	Value           json.RawMessage
	ProviderOptions ProviderOptions
}

// ContentOutput is a multipart result containing text, files or custom
// provider content.
type ContentOutput struct {
	Value []ToolResultContent
}

func (TextOutput) isToolResultOutput()            {}
func (JSONOutput) isToolResultOutput()            {}
func (ExecutionDeniedOutput) isToolResultOutput() {}
func (ErrorTextOutput) isToolResultOutput()       {}
func (ErrorJSONOutput) isToolResultOutput()       {}
func (ContentOutput) isToolResultOutput()         {}

// ToolResultContent is one item of a ContentOutput: ToolResultText,
// ToolResultFile or ToolResultCustom.
type ToolResultContent interface {
	isToolResultContent()
}

// ToolResultText is a text item.
type ToolResultText struct {
	Text            string
	ProviderOptions ProviderOptions
}

// ToolResultFile is a file item.
type ToolResultFile struct {
	// Filename is optional.
	Filename        string
	MediaType       string
	Data            FileData
	ProviderOptions ProviderOptions
}

// ToolResultCustom is provider-specific content described entirely by its
// provider options.
type ToolResultCustom struct {
	ProviderOptions ProviderOptions
}

func (ToolResultText) isToolResultContent()   {}
func (ToolResultFile) isToolResultContent()   {}
func (ToolResultCustom) isToolResultContent() {}
