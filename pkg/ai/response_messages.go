package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// toResponseMessages converts the content of a step into the messages that
// continue the conversation: an assistant message with the generated content
// and provider-executed tool results, and a tool message with the results of
// local tools and the approval responses. Tool results follow the order of
// the tool calls.
func toResponseMessages(ctx context.Context, content []ContentPart, tools []Tool) ([]model.Message, error) {
	var messages []model.Message

	assistantContent, callOrder, err := assistantResponseContent(ctx, content, tools)
	if err != nil {
		return nil, err
	}

	if len(assistantContent) > 0 {
		messages = append(messages, model.AssistantMessage{
			Content: assistantContent,
		})
	}

	toolContent, err := toolResponseContent(ctx, content, tools)
	if err != nil {
		return nil, err
	}

	if len(toolContent) > 0 {
		messages = append(messages, model.ToolMessage{
			Content: sortToolResultsByCallOrder(toolContent, callOrder),
		})
	}

	return messages, nil
}

// assistantResponseContent returns the assistant parts and the order of the
// tool calls. Sources, empty text and local tool outputs are left out.
func assistantResponseContent(ctx context.Context, content []ContentPart, tools []Tool) ([]model.AssistantPart, map[string]int, error) {
	parts := make([]model.AssistantPart, 0, len(content))
	callOrder := make(map[string]int)

	for _, part := range content {
		switch typed := part.(type) {
		case TextContent:
			if typed.Text == "" {
				continue
			}

			parts = append(parts, model.TextPart{
				Text:            typed.Text,
				ProviderOptions: typed.ProviderMetadata.Options(),
			})

		case CustomContent:
			parts = append(parts, model.CustomPart{
				Kind:            typed.Kind,
				ProviderOptions: typed.ProviderMetadata.Options(),
			})

		case ReasoningContent:
			parts = append(parts, model.ReasoningPart{
				Text:            typed.Text,
				ProviderOptions: typed.ProviderMetadata.Options(),
			})

		case FileContent:
			parts = append(parts, model.FilePart{
				MediaType: typed.File.MediaType,
				Data: model.FileBytes{
					Data: typed.File.Data,
				},
				ProviderOptions: typed.ProviderMetadata.Options(),
			})

		case ReasoningFileContent:
			parts = append(parts, model.ReasoningFilePart{
				MediaType: typed.File.MediaType,
				Data: model.FileBytes{
					Data: typed.File.Data,
				},
				ProviderOptions: typed.ProviderMetadata.Options(),
			})

		case ToolCall:
			if _, seen := callOrder[typed.ToolCallID]; !seen {
				callOrder[typed.ToolCallID] = len(callOrder)
			}

			parts = append(parts, model.ToolCallPart{
				ToolCallID:       typed.ToolCallID,
				ToolName:         typed.ToolName,
				Input:            responseToolCallInput(typed),
				ProviderExecuted: typed.ProviderExecuted,
				ProviderOptions:  typed.ProviderMetadata.Options(),
			})

		case ToolOutput:
			if !toolOutputIsProviderExecuted(typed) {
				continue
			}

			outputPart, err := toolOutputPart(ctx, typed, tools)
			if err != nil {
				return nil, nil, err
			}

			parts = append(parts, outputPart)

		case ToolApprovalRequest:
			parts = append(parts, model.ToolApprovalRequestPart{
				ApprovalID:       typed.ApprovalID,
				ToolCallID:       typed.ToolCall.ToolCallID,
				Reason:           typed.Reason,
				IsAutomatic:      typed.IsAutomatic,
				Signature:        typed.Signature,
				InputSchemaInput: refinedFromInput(typed.ToolCall),
			})
		}
	}

	return parts, callOrder, nil
}

// refinedFromInput returns the validated input before refinement when
// refinement changed it, and nil otherwise.
func refinedFromInput(call ToolCall) json.RawMessage {
	if call.schemaInput == nil || equalJSON(call.schemaInput, call.Input) {
		return nil
	}

	return call.schemaInput
}

// responseToolCallInput returns the input to record for a call. Invalid
// calls whose input is not a JSON object are recorded with an empty object,
// which providers accept.
func responseToolCallInput(call ToolCall) json.RawMessage {
	trimmed := bytes.TrimSpace(call.Input)
	isObject := len(trimmed) > 0 && trimmed[0] == '{'

	if call.Invalid && !isObject {
		return json.RawMessage(`{}`)
	}

	return call.Input
}

// toolResponseContent returns the approval responses, with an execution
// denial for each denied call, and the outputs of local tools.
func toolResponseContent(ctx context.Context, content []ContentPart, tools []Tool) ([]model.ToolPart, error) {
	var parts []model.ToolPart

	for _, part := range content {
		switch typed := part.(type) {
		case ToolApprovalResponse:
			parts = append(parts, model.ToolApprovalResponsePart{
				ApprovalID:       typed.ApprovalID,
				Approved:         typed.Approved,
				Reason:           typed.Reason,
				ProviderExecuted: typed.ProviderExecuted,
			})

			if !typed.Approved {
				parts = append(parts, model.ToolResultPart{
					ToolCallID: typed.ToolCall.ToolCallID,
					ToolName:   typed.ToolCall.ToolName,
					Output: model.ExecutionDeniedOutput{
						Reason: typed.Reason,
					},
				})
			}

		case ToolOutput:
			if toolOutputIsProviderExecuted(typed) {
				continue
			}

			outputPart, err := toolOutputPart(ctx, typed, tools)
			if err != nil {
				return nil, err
			}

			parts = append(parts, outputPart)
		}
	}

	return parts, nil
}

// sortToolResultsByCallOrder orders the tool results by the order of their
// calls, keeping approval responses in place. Results of unknown calls keep
// their relative order after the known ones.
func sortToolResultsByCallOrder(parts []model.ToolPart, callOrder map[string]int) []model.ToolPart {
	var results []model.ToolResultPart
	for _, part := range parts {
		if result, ok := part.(model.ToolResultPart); ok {
			results = append(results, result)
		}
	}

	slices.SortStableFunc(results, func(first, second model.ToolResultPart) int {
		firstOrder, firstKnown := callOrder[first.ToolCallID]
		secondOrder, secondKnown := callOrder[second.ToolCallID]

		switch {
		case !firstKnown && !secondKnown:
			return 0

		case !firstKnown:
			return 1

		case !secondKnown:
			return -1

		default:
			return firstOrder - secondOrder
		}
	})

	sorted := make([]model.ToolPart, 0, len(parts))
	next := 0

	for _, part := range parts {
		if _, ok := part.(model.ToolResultPart); ok {
			sorted = append(sorted, results[next])
			next++

			continue
		}

		sorted = append(sorted, part)
	}

	return sorted
}

// lookupTool returns a pointer to the named tool, or nil when it is not
// declared.
func lookupTool(tools []Tool, name string) *Tool {
	tool, found := findTool(tools, name)
	if !found {
		return nil
	}

	return &tool
}
