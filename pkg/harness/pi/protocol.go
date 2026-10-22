package pi

import (
	"encoding/json"
	"fmt"
	"slices"
)

// Record types of Pi's RPC protocol.
const (
	recordResponse            = "response"
	recordExtensionUIRequest  = "extension_ui_request"
	recordExtensionUIResponse = "extension_ui_response"
	recordExtensionError      = "extension_error"
)

// Names shared with the adapter's Pi extension (assets/extension.ts).
const (
	dialogTitle         = "ai-sdk-harness-pi"
	syncCommand         = "/ai-sdk-harness-sync"
	startupErrorCommand = "ai-sdk-harness-startup-error"
	fileChangeStatusKey = "ai-sdk-harness-pi:file-change"
)

// Dialog request kinds of the adapter's Pi extension.
const (
	dialogToolResult   = "tool-result"
	dialogToolApproval = "tool-approval"
)

// recordHeader is the part of every record the connection routes on.
type recordHeader struct {
	Type string `json:"type"`
	ID   string `json:"id"`
}

// response is Pi's reply to one command.
type response struct {
	ID      string          `json:"id"`
	Command string          `json:"command"`
	Success bool            `json:"success"`
	Data    json.RawMessage `json:"data"`
	Error   string          `json:"error"`
}

// err returns the failure of an unsuccessful response.
func (r response) err() error {
	if r.Success {
		return nil
	}

	return &CommandError{
		Command: r.Command,
		Message: r.Error,
	}
}

// CommandError is a Pi RPC command that Pi rejected.
type CommandError struct {
	Command string
	Message string
}

// Error describes the rejected command.
func (e *CommandError) Error() string {
	return fmt.Sprintf("pi rejected the %s command: %s", e.Command, e.Message)
}

// promptCommand sends a user prompt, or an extension command.
type promptCommand struct {
	Type    string       `json:"type"`
	Message string       `json:"message"`
	Images  []imageBlock `json:"images,omitempty"`
}

// imageBlock is Pi's inline image content.
type imageBlock struct {
	Type     string `json:"type"`
	Data     string `json:"data"`
	MIMEType string `json:"mimeType"`
}

// steerCommand queues a steering message.
type steerCommand struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// compactCommand compacts the context.
type compactCommand struct {
	Type               string `json:"type"`
	CustomInstructions string `json:"customInstructions,omitempty"`
}

// setModelCommand selects a model.
type setModelCommand struct {
	Type     string `json:"type"`
	Provider string `json:"provider"`
	ModelID  string `json:"modelId"`
}

// simpleCommand is a command without arguments.
type simpleCommand struct {
	Type string `json:"type"`
}

// promptResponseData is the data of a prompt response.
type promptResponseData struct {
	Disposition string `json:"disposition"`
}

// Prompt dispositions.
const (
	dispositionHandled = "handled"
	dispositionStarted = "started"
)

// stateResponseData is the part of the get_state data the adapter reads.
type stateResponseData struct {
	Model       *modelInfo `json:"model"`
	SessionFile string     `json:"sessionFile"`
	IsStreaming bool       `json:"isStreaming"`
}

// modelInfo is the part of Pi's model object the adapter reads.
type modelInfo struct {
	ID       string   `json:"id"`
	Name     string   `json:"name"`
	Provider string   `json:"provider"`
	Input    []string `json:"input"`
}

// supportsImages reports image input.
func (m modelInfo) supportsImages() bool {
	return slices.Contains(m.Input, "image")
}

// availableModelsData is the data of a get_available_models response.
type availableModelsData struct {
	Models []modelInfo `json:"models"`
}

// sessionStatsData is the part of the get_session_stats data the adapter
// reads.
type sessionStatsData struct {
	Tokens sessionTokens `json:"tokens"`
}

// sessionTokens are the cumulative token counts of a session.
type sessionTokens struct {
	Input      int `json:"input"`
	Output     int `json:"output"`
	CacheRead  int `json:"cacheRead"`
	CacheWrite int `json:"cacheWrite"`
}

// commandsData is the data of a get_commands response.
type commandsData struct {
	Commands []commandInfo `json:"commands"`
}

// commandInfo is one discoverable command.
type commandInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// extensionUIRequest is an extension UI record from Pi.
type extensionUIRequest struct {
	ID          string `json:"id"`
	Method      string `json:"method"`
	Title       string `json:"title"`
	Placeholder string `json:"placeholder"`
	StatusKey   string `json:"statusKey"`
	StatusText  string `json:"statusText"`
}

// isDialog reports a request that waits for a response.
func (r extensionUIRequest) isDialog() bool {
	switch r.Method {
	case "select", "confirm", "input", "editor":
		return true
	default:
		return false
	}
}

// dialogRequest is the request the adapter's extension encodes in an input
// dialog's placeholder.
type dialogRequest struct {
	Kind       string `json:"kind"`
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
}

// valueResponse answers a dialog with a value.
type valueResponse struct {
	Type  string `json:"type"`
	ID    string `json:"id"`
	Value string `json:"value"`
}

// cancelledResponse dismisses a dialog.
type cancelledResponse struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	Cancelled bool   `json:"cancelled"`
}

// fileChangeStatus is the status text the extension reports a workspace
// file change with.
type fileChangeStatus struct {
	Event string `json:"event"`
	Path  string `json:"path"`
}

// extensionErrorEvent reports an extension handler failure.
type extensionErrorEvent struct {
	ExtensionPath string `json:"extensionPath"`
	Event         string `json:"event"`
	Error         string `json:"error"`
}

// approvalDecision is the value answering a tool approval dialog.
type approvalDecision struct {
	Approved bool   `json:"approved"`
	Reason   string `json:"reason,omitempty"`
}

// hostToolResult is the value answering a tool result dialog.
type hostToolResult struct {
	Content []hostToolContent `json:"content"`

	// FromModelOutput reports content converted from a model-facing tool
	// output rather than a serialized raw output.
	FromModelOutput bool `json:"fromModelOutput"`

	IsError bool `json:"isError"`
}

// hostToolContent is one content block of a host tool result.
type hostToolContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Data     string `json:"data,omitempty"`
	MIMEType string `json:"mimeType,omitempty"`
}

// extensionConfig is the configuration of the adapter's extension.
type extensionConfig struct {
	CWD            string              `json:"cwd"`
	TurnFile       string              `json:"turnFile"`
	SkillsDir      string              `json:"skillsDir"`
	BuiltinTools   []string            `json:"builtinTools"`
	PermissionMode string              `json:"permissionMode"`
	PathPolicy     *FileToolPathPolicy `json:"pathPolicy,omitempty"`
	Providers      []namedConfig       `json:"providers"`
	MCPServers     []namedConfig       `json:"mcpServers"`
}

// turnFile is the per-turn file the extension reads.
type turnFile struct {
	Instructions string     `json:"instructions,omitempty"`
	Tools        []toolSpec `json:"tools"`
}

// toolSpec is an application tool as the extension registers it.
type toolSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}
