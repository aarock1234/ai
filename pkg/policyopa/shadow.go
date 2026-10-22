package policyopa

import (
	"context"
	"time"

	"github.com/aarock1234/ai/pkg/ai"
)

// ShadowOptions configure Shadow.
type ShadowOptions struct {
	// Enforce makes the wrapped policy's decision effective. By default
	// every call is approved.
	Enforce bool

	// OnDecision receives every evaluated decision. It runs on the
	// generation's goroutine before the decision takes effect, so it should
	// return quickly; hand slow work, such as remote logging, to a worker
	// the application owns.
	OnDecision func(ctx context.Context, event PolicyDecisionEvent)
}

// PolicyDecisionEvent describes one evaluation of a policy in shadow mode.
// A call where Decision and Effective differ is one that the policy would
// have blocked or escalated if it were enforced.
type PolicyDecisionEvent struct {
	// ToolCall is the evaluated tool call.
	ToolCall ai.ToolCall

	// Decision is the decision of the wrapped policy. An empty status type
	// is reported as not applicable.
	Decision ai.ToolApprovalStatus

	// Enforced reports whether Decision takes effect.
	Enforced bool

	// Effective is the decision that takes effect: Decision when enforced,
	// and approval otherwise.
	Effective ai.ToolApprovalStatus

	// Timestamp is the time of the evaluation.
	Timestamp time.Time
}

// Shadow wraps a tool approval configuration, a function or a per-tool
// map, so that its decisions are reported to OnDecision but, unless Enforce
// is set, every call is approved. Use it to roll out a policy: watch the
// decisions it would make, fix the policy, then enforce it. Calls the
// configuration has no decision for are reported as not applicable.
//
// Errors of the wrapped configuration are returned unchanged.
//
// Go adaptation: upstream calls onDecision without waiting for it and
// ignores its failures. Here OnDecision runs synchronously, because the
// SDK does not start goroutines that nothing waits for.
func Shadow(approval ai.ToolApprovalConfiguration, options ShadowOptions) ai.ToolApprovalFunc {
	return func(ctx context.Context, event ai.ToolApprovalEvent) (ai.ToolApprovalStatus, error) {
		decision, err := evaluateApproval(ctx, approval, event)
		if err != nil {
			return ai.ToolApprovalStatus{}, err
		}

		effective := ai.ToolApprovalStatus{
			Type: ai.ToolApprovalApproved,
		}

		if options.Enforce {
			effective = decision
		}

		if options.OnDecision != nil {
			options.OnDecision(ctx, PolicyDecisionEvent{
				ToolCall:  event.ToolCall,
				Decision:  decision,
				Enforced:  options.Enforce,
				Effective: effective,
				Timestamp: time.Now(),
			})
		}

		return effective, nil
	}
}

// evaluateApproval evaluates an approval configuration, treating a nil
// configuration and a call it has no decision for as having no opinion.
func evaluateApproval(ctx context.Context, approval ai.ToolApprovalConfiguration, event ai.ToolApprovalEvent) (ai.ToolApprovalStatus, error) {
	if approval == nil {
		return notApplicable(), nil
	}

	status, decided, err := approval.DecideToolApproval(ctx, event)
	if err != nil {
		return ai.ToolApprovalStatus{}, err
	}

	if !decided {
		return notApplicable(), nil
	}

	return normalizeStatus(status), nil
}
