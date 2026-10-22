package harness

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// LifecycleStateType distinguishes the two lifecycle states.
type LifecycleStateType string

// Lifecycle state types.
const (
	StateResumeSession LifecycleStateType = "resume-session"
	StateContinueTurn  LifecycleStateType = "continue-turn"
)

// ResumeSessionState is returned by the between-turn lifecycle methods
// (Session.Detach and Session.Stop) and accepted by StartOptions.ResumeFrom
// to resume the same session before a new turn. It encodes to the upstream
// JSON shape, so callers can persist it.
type ResumeSessionState struct {
	// Type is StateResumeSession.
	Type LifecycleStateType `json:"type"`

	// HarnessID identifies the adapter that produced the state. Adapters
	// refuse mismatched states.
	HarnessID string `json:"harnessId"`

	// SpecificationVersion is SpecificationVersion.
	SpecificationVersion string `json:"specificationVersion"`

	// Data is the adapter-defined payload.
	Data json.RawMessage `json:"data"`

	// ContinueFrom is set when the session has an unfinished turn, which must
	// be continued after the session resumes.
	ContinueFrom *ContinueTurnState `json:"continueFrom,omitempty"`
}

// ContinueTurnState is returned by Session.SuspendTurn and accepted by
// StartOptions.ContinueFrom to reconnect to the session before continuing
// the suspended turn.
type ContinueTurnState struct {
	// Type is StateContinueTurn.
	Type LifecycleStateType `json:"type"`

	HarnessID            string          `json:"harnessId"`
	SpecificationVersion string          `json:"specificationVersion"`
	Data                 json.RawMessage `json:"data"`

	// PendingToolApprovals are the agent-owned approval requests awaiting a
	// decision. They live outside Data so that the agent, not the adapter,
	// owns them.
	PendingToolApprovals []PendingToolApproval `json:"pendingToolApprovals,omitempty"`

	// PendingToolResults are the agent-owned application tool calls waiting
	// for a caller-provided result.
	PendingToolResults []PendingToolResult `json:"pendingToolResults,omitempty"`

	// TurnSettings are the settings captured when the unfinished turn began.
	TurnSettings *TurnSettings `json:"turnSettings,omitempty"`
}

// NewResumeSessionState returns a resume state of the current specification
// version.
func NewResumeSessionState(harnessID string, data json.RawMessage) ResumeSessionState {
	return ResumeSessionState{
		Type:                 StateResumeSession,
		HarnessID:            harnessID,
		SpecificationVersion: SpecificationVersion,
		Data:                 data,
	}
}

// NewContinueTurnState returns a continuation state of the current
// specification version.
func NewContinueTurnState(harnessID string, data json.RawMessage) ContinueTurnState {
	return ContinueTurnState{
		Type:                 StateContinueTurn,
		HarnessID:            harnessID,
		SpecificationVersion: SpecificationVersion,
		Data:                 data,
	}
}

// ToolApprovalKind distinguishes approvals of built-in tools, answered by
// the adapter, from approvals of application tools, answered by the agent.
type ToolApprovalKind string

// Tool approval kinds.
const (
	ApprovalBuiltin ToolApprovalKind = "builtin"
	ApprovalCustom  ToolApprovalKind = "custom"
)

// PendingToolApproval is a tool call awaiting an approval decision.
type PendingToolApproval struct {
	ApprovalID string `json:"approvalId"`
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`

	// Input is the call's JSON input as the runtime produced it.
	Input string `json:"input"`

	Kind             ToolApprovalKind `json:"kind"`
	ProviderExecuted bool             `json:"providerExecuted,omitempty"`
	NativeName       string           `json:"nativeName,omitempty"`
}

// PendingToolResult is an application tool call waiting for a result.
type PendingToolResult struct {
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
	Input      string `json:"input"`

	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`

	// CompletedResult is the result of a call executed while the turn was
	// being suspended. It is submitted on continuation without running the
	// tool again.
	CompletedResult *CompletedToolResult `json:"completedResult,omitempty"`
}

// CompletedToolResult is a tool result recorded for a later submission.
type CompletedToolResult struct {
	// Output is the raw execution result. After a JSON round trip it is a
	// json.RawMessage.
	Output  any
	IsError bool

	// ToolResult is the model-facing representation, when the tool converted
	// its output.
	ToolResult *model.ToolResultPart
}

// completedToolResultJSON is the JSON form of CompletedToolResult.
type completedToolResultJSON struct {
	Output     json.RawMessage `json:"output"`
	IsError    bool            `json:"isError,omitempty"`
	ToolResult json.RawMessage `json:"toolResult,omitempty"`
}

// MarshalJSON encodes the result in the upstream JSON shape.
func (r CompletedToolResult) MarshalJSON() ([]byte, error) {
	output, err := json.Marshal(r.Output)
	if err != nil {
		return nil, fmt.Errorf("encode completed tool output: %w", err)
	}

	encoded := completedToolResultJSON{
		Output:  output,
		IsError: r.IsError,
	}

	if r.ToolResult != nil {
		toolResult, err := MarshalToolResultPart(*r.ToolResult)
		if err != nil {
			return nil, err
		}

		encoded.ToolResult = toolResult
	}

	return json.Marshal(encoded)
}

// UnmarshalJSON decodes the upstream JSON shape. Output becomes a
// json.RawMessage.
func (r *CompletedToolResult) UnmarshalJSON(data []byte) error {
	var decoded completedToolResultJSON
	if err := json.Unmarshal(data, &decoded); err != nil {
		return fmt.Errorf("decode completed tool result: %w", err)
	}

	r.Output = decoded.Output
	r.IsError = decoded.IsError
	r.ToolResult = nil

	if len(decoded.ToolResult) == 0 || string(decoded.ToolResult) == "null" {
		return nil
	}

	toolResult, err := UnmarshalToolResultPart(decoded.ToolResult)
	if err != nil {
		return err
	}

	r.ToolResult = &toolResult

	return nil
}
