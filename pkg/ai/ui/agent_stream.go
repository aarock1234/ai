package ui

import (
	"context"
	"fmt"
	"iter"
	"net/http"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// CreateAgentStreamOptions configure CreateAgentStream.
type CreateAgentStreamOptions struct {
	// Agent runs the conversation, such as an *ai.ToolLoopAgent.
	Agent ai.Agent

	// Messages are the UI messages received from the client. They are
	// validated against the agent's tools before conversion.
	Messages []Message

	// ConvertDataPart converts data parts into model parts. Nil leaves data
	// parts out.
	ConvertDataPart func(ctx context.Context, part DataPart) (model.UserPart, error)

	// Options are the agent's call options.
	Options any

	// Transforms are applied to the agent's stream in order.
	Transforms []ai.StreamTransform

	// OnStepEnd is called after each generation step.
	OnStepEnd func(ctx context.Context, step ai.StepResult)

	// MessageStream configures the UI message stream. Its Tools are replaced
	// by the agent's tools, and nil OriginalMessages default to the validated
	// Messages. Its OnStepFinish receives the response message after each
	// step.
	MessageStream MessageStreamOptions
}

// AgentStreamResponseOptions configure PipeAgentStreamToResponse and
// CreateAgentStreamResponse.
type AgentStreamResponseOptions struct {
	CreateAgentStreamOptions

	MessageStreamResponseInit
}

// CreateAgentStream validates and converts the UI messages, runs the agent
// with them and returns its output as a UI message stream. Validation,
// conversion and agent option errors are returned before streaming.
func CreateAgentStream(ctx context.Context, options CreateAgentStreamOptions) (iter.Seq2[Chunk, error], error) {
	if options.Agent == nil {
		return nil, &ai.InvalidArgumentError{
			Parameter: "agent",
			Message:   "agent must not be nil",
		}
	}

	tools := options.Agent.Tools()

	messages, err := validateMessagesForAgent(ctx, ValidateMessagesOptions{
		Messages: options.Messages,
		Tools:    tools,
	})
	if err != nil {
		return nil, fmt.Errorf("validate ui messages: %w", err)
	}

	modelMessages, err := ConvertToModelMessages(ctx, messages, ConvertToModelMessagesOptions{
		Tools:           tools,
		ConvertDataPart: options.ConvertDataPart,
	})
	if err != nil {
		return nil, fmt.Errorf("convert ui messages: %w", err)
	}

	result, err := options.Agent.Stream(ctx, ai.AgentStreamOptions{
		AgentCallOptions: ai.AgentCallOptions{
			Messages:  modelMessages,
			Options:   options.Options,
			OnStepEnd: options.OnStepEnd,
		},
		Transforms: options.Transforms,
	})
	if err != nil {
		return nil, err
	}

	streamOptions := options.MessageStream
	streamOptions.Tools = tools

	if streamOptions.OriginalMessages == nil {
		streamOptions.OriginalMessages = messages
	}

	return ToMessageStream(ctx, result.FullStream(), streamOptions), nil
}

// PipeAgentStreamToResponse runs the agent like CreateAgentStream and
// writes its UI message stream to w. Errors before streaming are returned
// without writing a response.
func PipeAgentStreamToResponse(ctx context.Context, w http.ResponseWriter, options AgentStreamResponseOptions) error {
	stream, err := CreateAgentStream(ctx, options.CreateAgentStreamOptions)
	if err != nil {
		return err
	}

	return PipeMessageStreamToResponse(w, MessageStreamResponseOptions{
		Stream:                    stream,
		MessageStreamResponseInit: options.MessageStreamResponseInit,
	})
}

// CreateAgentStreamResponse runs the agent like CreateAgentStream and
// returns an HTTP response that streams its UI message stream.
func CreateAgentStreamResponse(ctx context.Context, options AgentStreamResponseOptions) (*http.Response, error) {
	stream, err := CreateAgentStream(ctx, options.CreateAgentStreamOptions)
	if err != nil {
		return nil, err
	}

	return CreateMessageStreamResponse(MessageStreamResponseOptions{
		Stream:                    stream,
		MessageStreamResponseInit: options.MessageStreamResponseInit,
	})
}
