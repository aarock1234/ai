package policyopa

import (
	"bytes"
	"encoding/json"

	"github.com/aarock1234/ai/pkg/ai"
)

// unrecognizedDecisionReason is the reason of the denial that replaces a
// policy result in an unknown format.
const unrecognizedDecisionReason = "unrecognized OPA policy decision"

// OPA decision values of the explicit result format.
const (
	opaDecisionAllow            = "allow"
	opaDecisionDeny             = "deny"
	opaDecisionRequiresApproval = "requires-approval"
	opaDecisionNotApplicable    = "not-applicable"
)

// NormalizeOPADecision converts an OPA result into a tool approval status.
// It accepts two result formats:
//
//   - Explicit: {"decision": "allow" | "deny" | "requires-approval" |
//     "not-applicable", "reason": string}, mapped to approved, denied,
//     user approval and not applicable.
//   - Boolean: {"allow": bool, "reason": string}, mapped to approved or
//     denied.
//
// A missing or null result is not applicable, so that a rule matching no
// branch has no opinion. Any other result is denied, so that malformed
// policy output cannot bypass the approval gate.
func NormalizeOPADecision(result json.RawMessage) ai.ToolApprovalStatus {
	if isUndefinedResult(result) {
		return notApplicable()
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(result, &fields); err != nil || fields == nil {
		return unrecognizedDecision()
	}

	reason, _ := decodeJSONValue[string](fields["reason"])

	if decision, ok := decodeJSONValue[string](fields["decision"]); ok {
		if status, known := explicitDecision(decision, reason); known {
			return status
		}
	}

	if allow, ok := decodeJSONValue[bool](fields["allow"]); ok {
		if allow {
			return withReason(ai.ToolApprovalApproved, reason)
		}

		return withReason(ai.ToolApprovalDenied, reason)
	}

	return unrecognizedDecision()
}

// explicitDecision maps a decision of the explicit format. It reports false
// for unknown decisions.
func explicitDecision(decision, reason string) (ai.ToolApprovalStatus, bool) {
	switch decision {
	case opaDecisionAllow:
		return withReason(ai.ToolApprovalApproved, reason), true
	case opaDecisionDeny:
		return withReason(ai.ToolApprovalDenied, reason), true
	case opaDecisionRequiresApproval:
		return withReason(ai.ToolApprovalUserApproval, reason), true
	case opaDecisionNotApplicable:
		return notApplicable(), true
	default:
		return ai.ToolApprovalStatus{}, false
	}
}

// isUndefinedResult reports whether a policy produced no result.
func isUndefinedResult(result json.RawMessage) bool {
	trimmed := bytes.TrimSpace(result)

	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}

// decodeJSONValue decodes a JSON value of type Value. It reports false when
// the value is missing or has another type.
func decodeJSONValue[Value any](raw json.RawMessage) (Value, bool) {
	var value Value

	if isUndefinedResult(raw) {
		return value, false
	}

	if err := json.Unmarshal(raw, &value); err != nil {
		return value, false
	}

	return value, true
}

func withReason(statusType ai.ToolApprovalStatusType, reason string) ai.ToolApprovalStatus {
	return ai.ToolApprovalStatus{
		Type:   statusType,
		Reason: reason,
	}
}

func notApplicable() ai.ToolApprovalStatus {
	return ai.ToolApprovalStatus{
		Type: ai.ToolApprovalNotApplicable,
	}
}

func unrecognizedDecision() ai.ToolApprovalStatus {
	return ai.ToolApprovalStatus{
		Type:   ai.ToolApprovalDenied,
		Reason: unrecognizedDecisionReason,
	}
}

// normalizeStatus treats the empty status type as not applicable.
func normalizeStatus(status ai.ToolApprovalStatus) ai.ToolApprovalStatus {
	if status.Type == "" {
		return notApplicable()
	}

	return status
}
