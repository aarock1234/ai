package codemode

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// ApprovalInterruptKind is the payload kind of the interrupts that
// ApprovalModeInterrupt produces.
const ApprovalInterruptKind = "ai-sdk-code-mode/tool-approval"

// ApprovalResponse answers an approval interrupt.
type ApprovalResponse struct {
	// ApprovalID is the InterruptID of the approval interrupt.
	ApprovalID string
	Approved   bool

	// Reason optionally explains a denial.
	Reason string
}

// approvalResolution is the JSON resolution of an approval interrupt.
type approvalResolution struct {
	Approved *bool   `json:"approved"`
	Reason   *string `json:"reason,omitempty"`
}

// IsApprovalInterrupt reports whether value is an authentic approval
// Interrupt, as accepted by IsInterrupt.
func IsApprovalInterrupt(value any, security ContinuationSecurity) bool {
	interrupt, ok := interruptFromValue(value, security)

	return ok && isApprovalPayload(interrupt.Payload)
}

func isApprovalPayload(payload InterruptPayload) bool {
	return payload.Kind == ApprovalInterruptKind
}

// ContinueApprovalOptions configure ContinueApproval.
type ContinueApprovalOptions struct {
	Interrupt *Interrupt
	Response  ApprovalResponse

	// Tools must be the host tools of the interrupted invocation.
	Tools []ai.Tool

	// Messages are forwarded to the host tools.
	Messages []model.Message

	Options Options
}

// ContinueApproval resumes an invocation that stopped at an approval
// interrupt. Approved calls run; a denial fails the invocation with an
// Error whose code is ErrorCodeToolApprovalDenied. Calls of the same batch
// are exposed one at a time, and no call runs until the whole batch is
// decided.
func ContinueApproval(ctx context.Context, options ContinueApprovalOptions) (Result, error) {
	if options.Interrupt == nil {
		return Result{}, newProtocolError("a code mode approval interrupt is required")
	}

	if options.Response.ApprovalID != options.Interrupt.InterruptID {
		return Result{}, newProtocolError(fmt.Sprintf(
			"approval response %s does not match pending code mode approval %s",
			options.Response.ApprovalID,
			options.Interrupt.InterruptID,
		))
	}

	codeModeOptions := options.Options
	codeModeOptions.Approval.Mode = ApprovalModeInterrupt

	return ContinueInterrupt(ctx, ContinueInterruptOptions{
		Interrupt: options.Interrupt,
		Resolution: ApprovalDecision{
			Approved: options.Response.Approved,
			Reason:   options.Response.Reason,
		},
		Tools:    options.Tools,
		Messages: options.Messages,
		Options:  codeModeOptions,
	})
}

// ApprovalMessages returns the assistant message that asks the user to
// approve the call of an approval interrupt, for applications that collect
// approvals through the conversation.
func ApprovalMessages(interrupt *Interrupt) []model.Message {
	return []model.Message{
		model.AssistantMessage{
			Content: []model.AssistantPart{
				model.ToolCallPart{
					ToolCallID: interrupt.ToolCallID,
					ToolName:   interrupt.ToolName,
					Input:      slices.Clone(interrupt.Input),
				},
				model.ToolApprovalRequestPart{
					ApprovalID: interrupt.InterruptID,
					ToolCallID: interrupt.ToolCallID,
				},
			},
		},
	}
}

// ApprovalResponseFromMessages returns the latest approval response to the
// interrupt in the tool messages of a conversation.
func ApprovalResponseFromMessages(messages []model.Message, interrupt *Interrupt) (ApprovalResponse, bool) {
	for _, message := range slices.Backward(messages) {
		toolMessage, ok := message.(model.ToolMessage)
		if !ok {
			continue
		}

		for _, part := range toolMessage.Content {
			response, ok := part.(model.ToolApprovalResponsePart)
			if !ok || response.ApprovalID != interrupt.InterruptID {
				continue
			}

			return ApprovalResponse{
				ApprovalID: response.ApprovalID,
				Approved:   response.Approved,
				Reason:     response.Reason,
			}, true
		}
	}

	return ApprovalResponse{}, false
}

// decodeApprovalResolution reads the resolution of an approval interrupt.
func decodeApprovalResolution(resolution json.RawMessage) (ApprovalDecision, error) {
	var decoded approvalResolution
	if err := json.Unmarshal(resolution, &decoded); err != nil || decoded.Approved == nil {
		return ApprovalDecision{}, newProtocolError("code mode approval resolution must be a boolean approval decision")
	}

	decision := ApprovalDecision{
		Approved: *decoded.Approved,
	}

	if decoded.Reason != nil {
		decision.Reason = *decoded.Reason
	}

	return decision, nil
}

// encodeApprovalResolution normalizes the resolution of an approval
// interrupt.
func encodeApprovalResolution(resolution any) (json.RawMessage, error) {
	encoded, err := json.Marshal(resolution)
	if err != nil {
		return nil, newProtocolError("code mode approval resolution must be a boolean approval decision")
	}

	decision, err := decodeApprovalResolution(encoded)
	if err != nil {
		return nil, err
	}

	normalized := approvalResolution{
		Approved: &decision.Approved,
	}

	if decision.Reason != "" {
		normalized.Reason = &decision.Reason
	}

	return json.Marshal(normalized)
}
