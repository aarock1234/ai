package openai

import "encoding/json"

// ApplyPatchInput is the input of an apply patch tool call.
type ApplyPatchInput struct {
	// CallID is the ID of the apply patch call.
	CallID string `json:"callId"`

	// Operation is the file change to apply.
	Operation ApplyPatchOperation `json:"operation"`
}

// ApplyPatchOperationType identifies a file change.
type ApplyPatchOperationType string

// Apply patch operation types.
const (
	ApplyPatchCreateFile ApplyPatchOperationType = "create_file"
	ApplyPatchDeleteFile ApplyPatchOperationType = "delete_file"
	ApplyPatchUpdateFile ApplyPatchOperationType = "update_file"
)

// ApplyPatchOperation is a file change. Path is relative to the workspace
// root. Diff is the unified diff for create and update operations.
type ApplyPatchOperation struct {
	Type ApplyPatchOperationType `json:"type"`
	Path string                  `json:"path"`
	Diff string                  `json:"diff,omitempty"`
}

// ApplyPatchStatus reports whether a patch was applied.
type ApplyPatchStatus string

// Apply patch statuses.
const (
	ApplyPatchCompleted ApplyPatchStatus = "completed"
	ApplyPatchFailed    ApplyPatchStatus = "failed"
)

// ApplyPatchOutput is the result of an apply patch call, returned by the
// application as a JSON tool result.
type ApplyPatchOutput struct {
	Status ApplyPatchStatus `json:"status"`

	// Output is optional log text, such as patch results or errors.
	Output string `json:"output,omitempty"`
}

// CodeInterpreterInput is the input of a code interpreter call.
type CodeInterpreterInput struct {
	// Code is the code that ran, or nil when it is not available.
	Code *string `json:"code"`

	// ContainerID identifies the container that ran the code.
	ContainerID string `json:"containerId"`
}

// CodeInterpreterResult is the result of a code interpreter call. Outputs
// is nil when no outputs are available.
type CodeInterpreterResult struct {
	Outputs []CodeInterpreterOutput `json:"outputs"`
}

// CodeInterpreterOutput is one output: "logs" with Logs or "image" with URL.
type CodeInterpreterOutput struct {
	Type string `json:"type"`
	Logs string `json:"logs,omitempty"`
	URL  string `json:"url,omitempty"`
}

// ComputerInput is the input of a computer tool call.
type ComputerInput struct {
	// Actions are the UI actions to execute in order.
	Actions []ComputerAction `json:"actions"`

	// PendingSafetyChecks must be acknowledged before continuing.
	PendingSafetyChecks []ComputerSafetyCheck `json:"pendingSafetyChecks"`

	// Status is "in_progress", "completed" or "incomplete".
	Status string `json:"status"`
}

// ComputerActionType identifies a UI action.
type ComputerActionType string

// Computer action types.
const (
	ComputerClick       ComputerActionType = "click"
	ComputerDoubleClick ComputerActionType = "double_click"
	ComputerDrag        ComputerActionType = "drag"
	ComputerKeypress    ComputerActionType = "keypress"
	ComputerMove        ComputerActionType = "move"
	ComputerScreenshot  ComputerActionType = "screenshot"
	ComputerScroll      ComputerActionType = "scroll"
	ComputerType        ComputerActionType = "type"
	ComputerWait        ComputerActionType = "wait"
)

// ComputerAction is one UI action. Clicks set Button, X and Y; double
// clicks and moves set X and Y; drags set Path; scrolls set X, Y, ScrollX
// and ScrollY; key presses set Keys; typing sets Text. Pointer actions may
// set Keys to the modifier keys held down.
type ComputerAction struct {
	Type    ComputerActionType `json:"type"`
	Button  string             `json:"button,omitempty"`
	X       *float64           `json:"x,omitempty"`
	Y       *float64           `json:"y,omitempty"`
	Path    []ComputerPoint    `json:"path,omitempty"`
	Keys    []string           `json:"keys,omitempty"`
	ScrollX *float64           `json:"scrollX,omitempty"`
	ScrollY *float64           `json:"scrollY,omitempty"`
	Text    string             `json:"text,omitempty"`
}

// ComputerPoint is a screen coordinate.
type ComputerPoint struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// ComputerSafetyCheck is a safety check raised by a computer call.
type ComputerSafetyCheck struct {
	ID      string `json:"id"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

// ComputerOutput is the result of a computer call, returned by the
// application as a JSON tool result.
type ComputerOutput struct {
	// Output is the screenshot captured after the actions.
	Output ComputerScreenshotOutput `json:"output"`

	// AcknowledgedSafetyChecks are the safety checks the application
	// reviewed.
	AcknowledgedSafetyChecks []ComputerSafetyCheck `json:"acknowledgedSafetyChecks,omitempty"`
}

// ComputerScreenshotOutput is a screenshot given by image URL or file ID.
type ComputerScreenshotOutput struct {
	// Type is "computer_screenshot".
	Type     string `json:"type"`
	ImageURL string `json:"imageUrl,omitempty"`
	FileID   string `json:"fileId,omitempty"`

	// Detail is "auto", "low", "high" or "original".
	Detail string `json:"detail,omitempty"`
}

// ComputerUseResult is the result of a legacy computer use call.
type ComputerUseResult struct {
	// Type is "computer_use_tool_result".
	Type   string `json:"type"`
	Status string `json:"status"`
}

// FileSearchResult is the result of a file search call.
type FileSearchResult struct {
	Queries []string `json:"queries"`

	// Results is nil when results were not included.
	Results []FileSearchMatch `json:"results"`
}

// FileSearchMatch is one retrieved file chunk.
type FileSearchMatch struct {
	// Attributes are the file's attributes: strings, numbers or booleans.
	Attributes map[string]json.RawMessage `json:"attributes"`

	FileID   string `json:"fileId"`
	Filename string `json:"filename"`

	// Score is the relevance, between 0 and 1.
	Score float64 `json:"score"`

	Text string `json:"text"`
}

// ImageGenerationResult is the result of an image generation call: the
// base64-encoded image.
type ImageGenerationResult struct {
	Result string `json:"result"`
}

// LocalShellInput is the input of a local shell call.
type LocalShellInput struct {
	Action LocalShellAction `json:"action"`
}

// LocalShellAction is a command to execute.
type LocalShellAction struct {
	// Type is "exec".
	Type             string            `json:"type"`
	Command          []string          `json:"command"`
	TimeoutMs        *int              `json:"timeoutMs,omitempty"`
	User             string            `json:"user,omitempty"`
	WorkingDirectory string            `json:"workingDirectory,omitempty"`
	Env              map[string]string `json:"env,omitempty"`
}

// LocalShellOutput is the result of a local shell call, returned by the
// application as a JSON tool result.
type LocalShellOutput struct {
	Output string `json:"output"`
}

// MCPResult is the result of a remote MCP tool call.
type MCPResult struct {
	// Type is "call".
	Type        string  `json:"type"`
	ServerLabel string  `json:"serverLabel"`
	Name        string  `json:"name"`
	Arguments   string  `json:"arguments"`
	Output      *string `json:"output,omitempty"`

	// Error is the error string or object, when the call failed.
	Error json.RawMessage `json:"error,omitempty"`
}

// ProgrammaticToolCallingInput is the input of a hosted program.
type ProgrammaticToolCallingInput struct {
	// Code is the JavaScript source OpenAI generated and executed.
	Code string `json:"code"`

	// Fingerprint must be preserved across requests for replay.
	Fingerprint string `json:"fingerprint"`
}

// ProgrammaticToolCallingResult is the result of a hosted program.
type ProgrammaticToolCallingResult struct {
	Result string `json:"result"`

	// Status is "completed" or "incomplete".
	Status string `json:"status"`
}

// ShellInput is the input of a shell call.
type ShellInput struct {
	Action ShellAction `json:"action"`
}

// ShellAction lists commands to execute.
type ShellAction struct {
	Commands []string `json:"commands"`

	// TimeoutMs optionally limits the run time in milliseconds.
	TimeoutMs *int `json:"timeoutMs,omitempty"`

	// MaxOutputLength optionally limits the characters returned per command.
	MaxOutputLength *int `json:"maxOutputLength,omitempty"`
}

// ShellOutput is the result of a shell call. The application returns it as
// a JSON tool result; hosted shells report it as a tool result.
type ShellOutput struct {
	// MaxOutputLength is the limit applied to each command's output.
	MaxOutputLength *int `json:"maxOutputLength,omitempty"`

	Output []ShellCommandOutput `json:"output"`
}

// ShellCommandOutput is the output of one command.
type ShellCommandOutput struct {
	Stdout  string       `json:"stdout"`
	Stderr  string       `json:"stderr"`
	Outcome ShellOutcome `json:"outcome"`
}

// ShellOutcome is "timeout" or "exit" with ExitCode.
type ShellOutcome struct {
	Type     string `json:"type"`
	ExitCode *int   `json:"exitCode,omitempty"`
}

// ToolSearchInput is the input of a tool search call.
type ToolSearchInput struct {
	// Arguments are the search arguments, kept for replay.
	Arguments json.RawMessage `json:"arguments,omitempty"`

	// CallID is set for client-executed searches and nil for hosted ones.
	CallID *string `json:"call_id"`
}

// ToolSearchOutput lists the tools a tool search loaded. Client-executed
// searches return it as a JSON tool result.
type ToolSearchOutput struct {
	// Tools are tool definitions in the API's format.
	Tools []json.RawMessage `json:"tools"`
}

// WebSearchResult is the result of a web search call.
type WebSearchResult struct {
	// Action describes what the model did, when reported.
	Action *WebSearchAction `json:"action,omitempty"`

	// Sources are the sources the search used, when included.
	Sources []WebSearchSource `json:"sources,omitempty"`
}

// WebSearchActionType identifies a web search action.
type WebSearchActionType string

// Web search action types.
const (
	WebSearchActionSearch     WebSearchActionType = "search"
	WebSearchActionOpenPage   WebSearchActionType = "openPage"
	WebSearchActionFindInPage WebSearchActionType = "findInPage"
)

// WebSearchAction is a web search action. Searches set Query and Queries;
// page actions set URL, and find-in-page actions also set Pattern.
type WebSearchAction struct {
	Type    WebSearchActionType `json:"type"`
	Query   string              `json:"query,omitempty"`
	Queries []string            `json:"queries,omitempty"`
	URL     string              `json:"url,omitempty"`
	Pattern string              `json:"pattern,omitempty"`
}

// WebSearchSource is a source used by a web search: "url" with URL or "api"
// with Name.
type WebSearchSource struct {
	Type string `json:"type"`
	URL  string `json:"url,omitempty"`
	Name string `json:"name,omitempty"`
}

// WebSearchFailure is the error result of a failed web search.
type WebSearchFailure struct {
	// Status is "failed" or "incomplete".
	Status string `json:"status"`
}
