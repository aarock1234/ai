package ai

import (
	"context"

	"github.com/aarock1234/ai/pkg/model"
)

// Agent generates or streams output for a prompt, calling tools across
// steps. ToolLoopAgent is the standard implementation; applications may
// implement their own.
type Agent interface {
	// ID identifies the agent. It may be empty.
	ID() string

	// Tools returns the tools the agent can use, in declaration order.
	Tools() []Tool

	// Generate runs the agent without streaming.
	Generate(ctx context.Context, options AgentCallOptions) (*GenerateTextResult, error)

	// Stream runs the agent and streams its output.
	Stream(ctx context.Context, options AgentStreamOptions) (*StreamTextResult, error)
}

// AgentCallOptions configure one call of an agent. Set either Prompt or
// Messages. The callbacks run after the agent's own callbacks of the same
// kind.
type AgentCallOptions struct {
	// Prompt is a text prompt, sent as a single user message.
	Prompt string

	// Messages is the conversation to continue.
	Messages []model.Message

	// Options are application-defined call options. The agent passes them to
	// its PrepareCall function, which type-asserts them.
	Options any

	// Timeout replaces the timeouts of the agent's settings for this call.
	// Nil keeps them.
	Timeout *TimeoutConfiguration

	// The callbacks have the meaning of the GenerateTextOptions fields of the
	// same names.
	OnStart              func(ctx context.Context, event GenerateTextStartEvent)
	OnStepStart          func(ctx context.Context, event GenerateTextStepStartEvent)
	OnToolExecutionStart func(ctx context.Context, event ToolExecutionStartEvent)
	OnToolExecutionEnd   func(ctx context.Context, event ToolExecutionEndEvent)
	OnStepEnd            func(ctx context.Context, step StepResult)
	OnEnd                func(ctx context.Context, event GenerateTextEndEvent)
}

// AgentStreamOptions configure one streaming call of an agent.
type AgentStreamOptions struct {
	AgentCallOptions

	// Transforms are applied to the stream in order, for example
	// SmoothStream.
	Transforms []StreamTransform
}
