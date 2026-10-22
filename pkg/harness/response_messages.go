package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/harness/internal/jsonvalue"
	"github.com/aarock1234/ai/pkg/model"
)

// responseMessageInput is the step content to convert into response
// messages.
type responseMessageInput struct {
	content []ai.ContentPart
	tools   []ai.Tool

	// modelOutputs are the model-facing outputs already submitted to the
	// runtime, keyed by tool call ID. They replace the default conversion.
	modelOutputs map[string]model.ToolResultOutput
}

// responseMessages converts the content of a step into an assistant message
// with the generated content and provider-executed tool results, and a tool
// message with the results of host tools and the approval responses.
func responseMessages(ctx context.Context, input responseMessageInput) ([]model.Message, error) {
	var assistantParts []model.AssistantPart
	var toolParts []model.ToolPart

	for _, part := range input.content {
		switch typed := part.(type) {
		case ai.TextContent:
			if typed.Text == "" {
				continue
			}

			assistantParts = append(assistantParts, model.TextPart{
				Text:            typed.Text,
				ProviderOptions: typed.ProviderMetadata.Options(),
			})
		case ai.ReasoningContent:
			assistantParts = append(assistantParts, model.ReasoningPart{
				Text:            typed.Text,
				ProviderOptions: typed.ProviderMetadata.Options(),
			})
		case ai.ToolCall:
			assistantParts = append(assistantParts, model.ToolCallPart{
				ToolCallID:       typed.ToolCallID,
				ToolName:         typed.ToolName,
				Input:            responseToolCallInput(typed),
				ProviderExecuted: typed.ProviderExecuted,
				ProviderOptions:  typed.ProviderMetadata.Options(),
			})
		case ai.ToolApprovalRequest:
			assistantParts = append(assistantParts, model.ToolApprovalRequestPart{
				ApprovalID:  typed.ApprovalID,
				ToolCallID:  typed.ToolCall.ToolCallID,
				Reason:      typed.Reason,
				IsAutomatic: typed.IsAutomatic,
			})
		case ai.ToolApprovalResponse:
			toolParts = append(toolParts, approvalResponseParts(typed)...)
		case ai.ToolResult, ai.ToolError:
			outputPart, err := toolOutputPart(ctx, typed.(ai.ToolOutput), input)
			if err != nil {
				return nil, err
			}

			if isProviderExecutedOutput(typed.(ai.ToolOutput)) {
				assistantParts = append(assistantParts, outputPart)
			} else {
				toolParts = append(toolParts, outputPart)
			}
		}
	}

	var messages []model.Message

	if len(assistantParts) > 0 {
		messages = append(messages, model.AssistantMessage{
			Content: assistantParts,
		})
	}

	if len(toolParts) > 0 {
		messages = append(messages, model.ToolMessage{
			Content: toolParts,
		})
	}

	return messages, nil
}

// approvalResponseParts returns an approval response, followed by an
// execution denial when the call was denied.
func approvalResponseParts(response ai.ToolApprovalResponse) []model.ToolPart {
	parts := []model.ToolPart{
		model.ToolApprovalResponsePart{
			ApprovalID:       response.ApprovalID,
			Approved:         response.Approved,
			Reason:           response.Reason,
			ProviderExecuted: response.ProviderExecuted,
		},
	}

	if response.Approved {
		return parts
	}

	return append(parts, model.ToolResultPart{
		ToolCallID: response.ToolCall.ToolCallID,
		ToolName:   response.ToolCall.ToolName,
		Output: model.ExecutionDeniedOutput{
			Reason: response.Reason,
		},
	})
}

// toolOutputPart converts a tool result or error into the part that reports
// it to the model.
func toolOutputPart(ctx context.Context, output ai.ToolOutput, input responseMessageInput) (model.ToolResultPart, error) {
	switch typed := output.(type) {
	case ai.ToolResult:
		modelOutput, err := resultModelOutput(ctx, typed, input)
		if err != nil {
			return model.ToolResultPart{}, err
		}

		return model.ToolResultPart{
			ToolCallID:      typed.ToolCallID,
			ToolName:        typed.ToolName,
			Output:          modelOutput,
			ProviderOptions: typed.ProviderMetadata.Options(),
		}, nil
	case ai.ToolError:
		modelOutput, err := errorModelOutput(typed)
		if err != nil {
			return model.ToolResultPart{}, err
		}

		return model.ToolResultPart{
			ToolCallID:      typed.ToolCallID,
			ToolName:        typed.ToolName,
			Output:          modelOutput,
			ProviderOptions: typed.ProviderMetadata.Options(),
		}, nil
	default:
		return model.ToolResultPart{}, fmt.Errorf("convert tool output: unsupported type %T", output)
	}
}

// resultModelOutput returns the submitted model output of a result, or its
// default conversion.
func resultModelOutput(ctx context.Context, result ai.ToolResult, input responseMessageInput) (model.ToolResultOutput, error) {
	if submitted, ok := input.modelOutputs[result.ToolCallID]; ok {
		return submitted, nil
	}

	tool, _ := findTool(input.tools, result.ToolName)

	return toolModelOutput(ctx, toolModelOutputInput{
		tool:       tool,
		toolCallID: result.ToolCallID,
		input:      result.Input,
		output:     result.Output,
	})
}

// errorModelOutput reports a failure to the model: error JSON for
// provider-executed tools and error text otherwise.
func errorModelOutput(toolError ai.ToolError) (model.ToolResultOutput, error) {
	if !toolError.ProviderExecuted {
		return model.ErrorTextOutput{
			Value: errorText(toolError.Error),
		}, nil
	}

	var providerError *ai.ProviderToolError
	if errors.As(toolError.Error, &providerError) {
		return model.ErrorJSONOutput{
			Value: providerError.Result,
		}, nil
	}

	encoded, err := json.Marshal(errorText(toolError.Error))
	if err != nil {
		return nil, fmt.Errorf("encode tool error: %w", err)
	}

	return model.ErrorJSONOutput{
		Value: encoded,
	}, nil
}

// toolModelOutputInput is a successful tool output to convert for the model.
type toolModelOutputInput struct {
	// tool is nil when the tool is unknown.
	tool       *ai.Tool
	toolCallID string
	input      json.RawMessage
	output     any
}

// toolModelOutput converts a tool output with the tool's ToModelOutput, or
// into text for strings and JSON for everything else.
func toolModelOutput(ctx context.Context, input toolModelOutputInput) (model.ToolResultOutput, error) {
	if input.tool != nil && input.tool.ToModelOutput != nil {
		output, err := input.tool.ToModelOutput(ctx, ai.ToolOutputEvent[json.RawMessage, any]{
			ToolCallID: input.toolCallID,
			Input:      input.input,
			Output:     input.output,
		})
		if err != nil {
			return nil, fmt.Errorf("convert output of tool %q: %w", input.tool.Name, err)
		}

		return output, nil
	}

	if text, ok := input.output.(string); ok {
		return model.TextOutput{
			Value: text,
		}, nil
	}

	encoded, err := json.Marshal(input.output)
	if err != nil {
		return nil, fmt.Errorf("encode tool output: %w", err)
	}

	return model.JSONOutput{
		Value: encoded,
	}, nil
}

// responseToolCallInput returns the input to record for a call: invalid
// calls whose input is not a JSON object are recorded with an empty object.
func responseToolCallInput(call ai.ToolCall) json.RawMessage {
	if call.Invalid && !jsonvalue.IsObject(call.Input) {
		return json.RawMessage(`{}`)
	}

	return call.Input
}

// isProviderExecutedOutput reports whether a tool output came from the
// runtime itself.
func isProviderExecutedOutput(output ai.ToolOutput) bool {
	switch typed := output.(type) {
	case ai.ToolResult:
		return typed.ProviderExecuted
	case ai.ToolError:
		return typed.ProviderExecuted
	default:
		return false
	}
}

// errorText describes a failure.
func errorText(err error) string {
	if err == nil {
		return "unknown error"
	}

	return err.Error()
}
