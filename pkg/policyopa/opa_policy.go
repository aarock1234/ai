package policyopa

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// OPAPolicyOptions configure OPAPolicy.
type OPAPolicyOptions struct {
	// Client evaluates the policy. It is required.
	Client PolicyClient

	// Path is the Rego entrypoint that returns the decision, such as
	// "agent/call/decision".
	Path string

	// ToInput shapes the OPA input of a tool call. Nil, or a nil result,
	// uses DefaultOPAInput.
	ToInput func(ctx context.Context, event ai.ToolApprovalEvent) any
}

// DefaultOPAInput is the OPA input of a tool call when ToInput is not set.
// Rego rules read input.tool.name, input.args and input.messages. Messages
// are encoded in the upstream model message format.
type DefaultOPAInput struct {
	Tool     OPATool
	Args     json.RawMessage
	Messages []model.Message
}

// OPATool identifies the called tool in DefaultOPAInput.
type OPATool struct {
	Name string `json:"name"`
}

// defaultOPAInputJSON is the JSON encoding of DefaultOPAInput.
type defaultOPAInputJSON struct {
	Tool     OPATool            `json:"tool"`
	Args     json.RawMessage    `json:"args"`
	Messages []wirejson.Message `json:"messages"`
}

// MarshalJSON encodes the input with its messages in the upstream model
// message format.
func (i DefaultOPAInput) MarshalJSON() ([]byte, error) {
	return json.Marshal(defaultOPAInputJSON{
		Tool:     i.Tool,
		Args:     wirejson.RawValue(i.Args),
		Messages: policyJSON.Messages(i.Messages),
	})
}

// OPAPolicy returns a tool approval function backed by an OPA policy. It
// evaluates the policy at Path for every tool call and converts the result
// with NormalizeOPADecision. When the evaluation fails the call is denied
// with the failure as the reason, so that an unavailable policy engine
// blocks tools instead of allowing them.
func OPAPolicy(options OPAPolicyOptions) ai.ToolApprovalFunc {
	return func(ctx context.Context, event ai.ToolApprovalEvent) (ai.ToolApprovalStatus, error) {
		input := opaPolicyInput(ctx, options.ToInput, event)

		result, err := options.Client.Evaluate(ctx, options.Path, input)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return ai.ToolApprovalStatus{}, fmt.Errorf("evaluate tool approval policy: %w", ctxErr)
			}

			return ai.ToolApprovalStatus{
				Type:   ai.ToolApprovalDenied,
				Reason: "policy evaluation failed: " + err.Error(),
			}, nil
		}

		return NormalizeOPADecision(result), nil
	}
}

// OptionalOPAPolicy returns OPAPolicy(options), or nil when options.Client
// is nil. A nil tool approval function lets the tools' own approval
// settings decide, so a policy can be configured per environment.
func OptionalOPAPolicy(options OPAPolicyOptions) ai.ToolApprovalFunc {
	if options.Client == nil {
		return nil
	}

	return OPAPolicy(options)
}

func opaPolicyInput(
	ctx context.Context,
	toInput func(ctx context.Context, event ai.ToolApprovalEvent) any,
	event ai.ToolApprovalEvent,
) any {
	if toInput != nil {
		if input := toInput(ctx, event); input != nil {
			return input
		}
	}

	return DefaultOPAInput{
		Tool: OPATool{
			Name: event.ToolCall.ToolName,
		},
		Args:     event.ToolCall.Input,
		Messages: event.Messages,
	}
}
