package bridge

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
)

// The commands below are the host-to-bridge vocabulary shared by every
// bridge adapter. Adapters define their own start command by embedding
// StartBase and send every command with Channel.Send.

// ToolWire is the part of an application tool that travels on the start
// command. Execution stays on the host.
type ToolWire struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

// ToolWiresOf converts tool specs to their wire form.
func ToolWiresOf(specs []harness.ToolSpec) []ToolWire {
	wires := make([]ToolWire, 0, len(specs))
	for _, spec := range specs {
		wires = append(wires, ToolWire{
			Name:        spec.Name,
			Description: spec.Description,
			InputSchema: spec.InputSchema,
		})
	}

	return wires
}

// StartBase holds the common fields of a start command. Adapters embed it in
// their start command with their runtime-specific configuration.
type StartBase struct {
	// Type is "start".
	Type string `json:"type"`

	Prompt               string                        `json:"prompt"`
	Tools                []ToolWire                    `json:"tools,omitempty"`
	Model                string                        `json:"model,omitempty"`
	Debug                *harness.DebugConfig          `json:"debug,omitempty"`
	PermissionMode       harness.PermissionMode        `json:"permissionMode,omitempty"`
	BuiltinToolFiltering *harness.BuiltinToolFiltering `json:"builtinToolFiltering,omitempty"`
	ResponseFormat       *ResponseFormatWire           `json:"responseFormat,omitempty"`
}

// ResponseFormatWire is the wire form of a turn's response format.
type ResponseFormatWire struct {
	// Type is "text" or "json".
	Type        string          `json:"type"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
}

// ToolResultCommand submits the result of an application tool call.
type ToolResultCommand struct {
	// Type is "tool-result".
	Type       string          `json:"type"`
	ToolCallID string          `json:"toolCallId"`
	Output     json.RawMessage `json:"output"`
	IsError    bool            `json:"isError,omitempty"`

	// ToolResult is the model-facing tool result part, in the upstream JSON
	// shape.
	ToolResult json.RawMessage `json:"toolResult,omitempty"`
}

// NewToolResultCommand encodes a tool result submission.
func NewToolResultCommand(submission harness.ToolResultSubmission) (ToolResultCommand, error) {
	output, err := json.Marshal(submission.Output)
	if err != nil {
		return ToolResultCommand{}, fmt.Errorf("encode tool output: %w", err)
	}

	command := ToolResultCommand{
		Type:       "tool-result",
		ToolCallID: submission.ToolCallID,
		Output:     output,
		IsError:    submission.IsError,
	}

	if submission.ToolResult != nil {
		toolResult, err := harness.MarshalToolResultPart(*submission.ToolResult)
		if err != nil {
			return ToolResultCommand{}, err
		}

		command.ToolResult = toolResult
	}

	return command, nil
}

// ToolApprovalResponseCommand answers a built-in tool approval request.
type ToolApprovalResponseCommand struct {
	// Type is "tool-approval-response".
	Type       string `json:"type"`
	ApprovalID string `json:"approvalId"`
	Approved   bool   `json:"approved"`
	Reason     string `json:"reason,omitempty"`
}

// NewToolApprovalResponseCommand encodes an approval submission.
func NewToolApprovalResponseCommand(submission harness.ToolApprovalSubmission) ToolApprovalResponseCommand {
	return ToolApprovalResponseCommand{
		Type:       "tool-approval-response",
		ApprovalID: submission.ApprovalID,
		Approved:   submission.Approved,
		Reason:     submission.Reason,
	}
}

// UserMessageCommand injects a user message into a running turn. The
// message ID correlates the bridge's UserMessageResponse.
type UserMessageCommand struct {
	// Type is "user-message".
	Type      string `json:"type"`
	MessageID string `json:"messageId,omitempty"`
	Text      string `json:"text"`
}

// SimpleCommand is a command with only a type: "abort", "destroy" or
// "stop". The bridge answers stop with a Stop frame and exits.
type SimpleCommand struct {
	Type string `json:"type"`
}

// ResumeCommand asks the bridge to replay every buffered event after
// LastSeenEventID.
type ResumeCommand struct {
	// Type is "resume".
	Type            string `json:"type"`
	LastSeenEventID int64  `json:"lastSeenEventId"`
}

// Ready is the line the bridge writes to standard output once its WebSocket
// server is listening.
type Ready struct {
	// Type is "bridge-ready".
	Type string `json:"type"`
	Port int    `json:"port"`
}

// ParseReady parses a bridge-ready line.
func ParseReady(line string) (Ready, bool) {
	var ready struct {
		Type string   `json:"type"`
		Port *float64 `json:"port"`
	}

	if err := json.Unmarshal([]byte(line), &ready); err != nil {
		return Ready{}, false
	}

	if ready.Type != "bridge-ready" || ready.Port == nil {
		return Ready{}, false
	}

	return Ready{
		Type: ready.Type,
		Port: int(*ready.Port),
	}, true
}
