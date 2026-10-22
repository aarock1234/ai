package policyopa

import (
	"context"
	"encoding/json"

	"github.com/aarock1234/ai/pkg/ai"
)

// WrapMCPToolsOptions configure WrapMCPTools.
type WrapMCPToolsOptions struct {
	// Default is the decision for tool calls the approval configuration has
	// no opinion on. The empty type selects ai.ToolApprovalUserApproval, so
	// that a person reviews every call no rule covers.
	Default ai.ToolApprovalStatusType
}

// WrappedMCPTools are tools with a total tool approval configuration.
type WrappedMCPTools struct {
	Tools        []ai.Tool
	ToolApproval ai.ToolApprovalConfiguration
}

// WrapMCPTools makes an approval policy total over a discovered tool set:
// every call that approval has no opinion on gets the Default decision
// instead of running unapproved.
//
// An ai.ToolApprovals map stays a map: tools it does not list get the
// default status, per-tool functions that return a not applicable status
// get the default instead, and configured static statuses are kept. Tools
// that are not in the tool set are dropped from the map. Any other
// configuration becomes a function that replaces not applicable decisions,
// and calls it has no decision for, with the default. A nil configuration
// sends every call to the default.
//
// The motivating case is MCP, where a server decides which tools exist and
// a forgotten rule would silently allow a tool. It works for any tool set:
//
//	tools, err := client.Tools(ctx, mcp.ToolsOptions{})
//	wrapped := policyopa.WrapMCPTools(tools, policy, policyopa.WrapMCPToolsOptions{})
func WrapMCPTools(tools []ai.Tool, approval ai.ToolApprovalConfiguration, options WrapMCPToolsOptions) WrappedMCPTools {
	fallback := ai.ToolApprovalStatus{
		Type: options.Default,
	}

	if fallback.Type == "" {
		fallback.Type = ai.ToolApprovalUserApproval
	}

	if approvals, isMap := approval.(ai.ToolApprovals); isMap {
		return WrappedMCPTools{
			Tools:        tools,
			ToolApproval: fillToolApprovals(tools, approvals, fallback),
		}
	}

	return WrappedMCPTools{
		Tools: tools,
		ToolApproval: ai.ToolApprovalFunc(func(ctx context.Context, event ai.ToolApprovalEvent) (ai.ToolApprovalStatus, error) {
			status, err := evaluateApproval(ctx, approval, event)
			if err != nil {
				return ai.ToolApprovalStatus{}, err
			}

			return statusOrFallback(status, fallback), nil
		}),
	}
}

// fillToolApprovals returns a per-tool map that covers every tool of the
// set, giving the fallback to tools the approvals do not list.
func fillToolApprovals(tools []ai.Tool, approvals ai.ToolApprovals, fallback ai.ToolApprovalStatus) ai.ToolApprovals {
	filled := make(ai.ToolApprovals, len(tools))

	for _, tool := range tools {
		rule, configured := approvals[tool.Name]

		switch {
		case !configured:
			filled[tool.Name] = ai.ToolApprovalRule{
				Status: fallback,
			}

		case rule.Decide != nil:
			filled[tool.Name] = ai.ToolApprovalRule{
				Decide: decideWithFallback(rule.Decide, fallback),
			}

		default:
			filled[tool.Name] = rule
		}
	}

	return filled
}

// decideWithFallback wraps a per-tool approval function so that a not
// applicable decision becomes the fallback.
func decideWithFallback(decide ai.SingleToolApprovalFunc, fallback ai.ToolApprovalStatus) ai.SingleToolApprovalFunc {
	return func(ctx context.Context, input json.RawMessage, options ai.ToolNeedsApprovalOptions) (ai.ToolApprovalStatus, error) {
		status, err := decide(ctx, input, options)
		if err != nil {
			return ai.ToolApprovalStatus{}, err
		}

		return statusOrFallback(normalizeStatus(status), fallback), nil
	}
}

// statusOrFallback returns the fallback for a not applicable status.
func statusOrFallback(status ai.ToolApprovalStatus, fallback ai.ToolApprovalStatus) ai.ToolApprovalStatus {
	if status.Type == ai.ToolApprovalNotApplicable {
		return fallback
	}

	return status
}
