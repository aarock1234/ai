package acp

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/harness/internal/jsonvalue"
	"github.com/aarock1234/ai/pkg/model"
)

// Frame types the ACP bridge adds to the shared bridge protocol.
const (
	frameToolCallCandidate = "acp-tool-call-candidate"
	frameQuestionRequest   = "acp-question-request"
	frameQuestionResolved  = "acp-question-resolved"
)

// customFrameTypes are the ACP frame types the channel keeps whole.
var customFrameTypes = []string{
	frameToolCallCandidate,
	frameQuestionRequest,
	frameQuestionResolved,
}

// Recovery modes of a start command.
const (
	recoveryLossyRerun  = "lossy-rerun"
	recoveryColdRestore = "cold-restore"
	recoveryDiskReplay  = "disk-replay"
)

// builtinToolMapping is the bridge's description of a built-in tool, used to
// map ACP tool calls to it.
type builtinToolMapping struct {
	// ToolName is the tool's name in Harness.BuiltinTools, emitted on tool
	// calls.
	ToolName string `json:"toolName"`

	NativeName string `json:"nativeName,omitempty"`

	// Title is the stable prefix of the display title, used when the
	// implementation omits a programmatic tool name.
	Title string `json:"title,omitempty"`

	ToolUseKind harness.BuiltinToolUseKind `json:"toolUseKind,omitempty"`

	// InputSchema identifies calls without a programmatic tool name.
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

// builtinToolMappings describes the built-in tools for the bridge.
func builtinToolMappings(tools []harness.BuiltinTool) []builtinToolMapping {
	mappings := make([]builtinToolMapping, 0, len(tools))

	for _, tool := range tools {
		mappings = append(mappings, builtinToolMapping{
			ToolName:    tool.Name,
			NativeName:  tool.NativeName,
			Title:       tool.Title,
			ToolUseKind: tool.UseKind,
			InputSchema: tool.InputSchema,
		})
	}

	return mappings
}

// serializableToolSpecs validates that application tools can be persisted:
// each input schema must be a JSON object.
func serializableToolSpecs(tools []harness.ToolSpec) ([]harness.ToolSpec, error) {
	specs := make([]harness.ToolSpec, 0, len(tools))

	for _, tool := range tools {
		if len(tool.InputSchema) > 0 && !jsonvalue.IsObject(tool.InputSchema) {
			return nil, fmt.Errorf("acp tool %q has an input schema that is not a JSON object", tool.Name)
		}

		specs = append(specs, tool)
	}

	return specs, nil
}

// startCommand starts or restores an ACP prompt turn.
type startCommand struct {
	// Type is "start".
	Type string `json:"type"`

	Prompt         []textBlock                `json:"prompt"`
	Tools          []harness.ToolSpec         `json:"tools,omitzero"`
	Model          string                     `json:"model,omitempty"`
	Debug          *harness.DebugConfig       `json:"debug,omitempty"`
	PermissionMode harness.PermissionMode     `json:"permissionMode,omitempty"`
	ResponseFormat *bridge.ResponseFormatWire `json:"responseFormat,omitempty"`

	Instructions        string               `json:"instructions,omitempty"`
	InstructionMapping  *InstructionMapping  `json:"instructionMapping,omitempty"`
	OutputSchemaMapping *OutputSchemaMapping `json:"outputSchemaMapping,omitempty"`
	MCPServers          map[string]any       `json:"mcpServers,omitzero"`

	BuiltinTools          []builtinToolMapping   `json:"builtinTools"`
	PermissionModeMapping *PermissionModeMapping `json:"permissionModeMapping,omitempty"`
	ModelMapping          *ModelMapping          `json:"modelMapping,omitempty"`

	TurnStartConfig turnStartConfig `json:"turnStartConfig"`
	RecoveryMode    *recoveryMode   `json:"recoveryMode,omitempty"`
}

// recoveryMode tells a respawned bridge how to restore the ACP session.
type recoveryMode struct {
	// Type is recoveryLossyRerun or recoveryColdRestore.
	Type         string `json:"type"`
	ACPSessionID string `json:"acpSessionId"`

	// Reason explains a lossy rerun.
	Reason string `json:"reason,omitempty"`
}

// toolCallCandidate asks the host to classify an ACP tool call before the
// bridge emits it.
type toolCallCandidate struct {
	RequestID string
	ToolCall  ToolCall
}

// questionRequest forwards a native question request of the
// implementation.
type questionRequest struct {
	RequestID      string
	NativeRequest  json.RawMessage
	NativeToolCall *ToolCall
}

// decodeToolCallCandidate decodes an acp-tool-call-candidate frame.
func decodeToolCallCandidate(data json.RawMessage) (toolCallCandidate, error) {
	var wire struct {
		RequestID *string         `json:"requestId"`
		ToolCall  json.RawMessage `json:"toolCall"`
	}

	if err := json.Unmarshal(data, &wire); err != nil || wire.RequestID == nil {
		return toolCallCandidate{}, invalidFrame(frameToolCallCandidate)
	}

	toolCall, ok := decodeToolCall(wire.ToolCall)
	if !ok {
		return toolCallCandidate{}, invalidFrame(frameToolCallCandidate)
	}

	return toolCallCandidate{
		RequestID: *wire.RequestID,
		ToolCall:  toolCall,
	}, nil
}

// decodeQuestionRequest decodes an acp-question-request frame.
func decodeQuestionRequest(data json.RawMessage) (questionRequest, error) {
	var wire struct {
		RequestID      *string         `json:"requestId"`
		NativeRequest  json.RawMessage `json:"nativeRequest"`
		NativeToolCall json.RawMessage `json:"nativeToolCall"`
	}

	if err := json.Unmarshal(data, &wire); err != nil || wire.RequestID == nil {
		return questionRequest{}, invalidFrame(frameQuestionRequest)
	}

	request := questionRequest{
		RequestID:     *wire.RequestID,
		NativeRequest: jsonOrNull(wire.NativeRequest),
	}

	if len(wire.NativeToolCall) > 0 && string(wire.NativeToolCall) != "null" {
		toolCall, ok := decodeToolCall(wire.NativeToolCall)
		if !ok {
			return questionRequest{}, invalidFrame(frameQuestionRequest)
		}

		request.NativeToolCall = &toolCall
	}

	return request, nil
}

// decodeQuestionResolved decodes the request ID of an acp-question-resolved
// frame.
func decodeQuestionResolved(data json.RawMessage) (string, error) {
	var wire struct {
		RequestID *string `json:"requestId"`
	}

	if err := json.Unmarshal(data, &wire); err != nil || wire.RequestID == nil {
		return "", invalidFrame(frameQuestionResolved)
	}

	return *wire.RequestID, nil
}

// invalidFrame reports an ACP frame that does not match the protocol.
func invalidFrame(frameType string) error {
	return fmt.Errorf("%w: %s frame is malformed", bridge.ErrInvalidFrame, frameType)
}

// classificationCommand answers a tool call candidate or question request
// with the host's classification.
func classificationCommand(requestID string, output any) (bridge.ToolResultCommand, error) {
	encoded, err := json.Marshal(output)
	if err != nil {
		return bridge.ToolResultCommand{}, fmt.Errorf("encode acp classification: %w", err)
	}

	return bridge.ToolResultCommand{
		Type:       "tool-result",
		ToolCallID: requestID,
		Output:     encoded,
	}, nil
}

// suppressionOutput classifies a tool call candidate.
type suppressionOutput struct {
	Suppress bool `json:"suppress"`
}

// questionClassification classifies a question request.
type questionClassification struct {
	// Type is "handled" or "unhandled".
	Type string `json:"type"`

	// ToolCallID is the askUserQuestions tool call of a handled request.
	ToolCallID string `json:"toolCallId,omitempty"`
}

// responseFormatWire converts a response format to its wire form.
func responseFormatWire(format *model.ResponseFormat) *bridge.ResponseFormatWire {
	if format == nil {
		return nil
	}

	formatType := format.Type
	if formatType == "" {
		formatType = model.ResponseFormatText
	}

	return &bridge.ResponseFormatWire{
		Type:        string(formatType),
		Schema:      format.Schema,
		Name:        format.Name,
		Description: format.Description,
	}
}

// jsonOrNull returns data, or JSON null when it is empty.
func jsonOrNull(data json.RawMessage) json.RawMessage {
	if len(data) == 0 {
		return json.RawMessage("null")
	}

	return data
}
