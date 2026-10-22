package harness

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// CollectToolApprovalContinuations returns the approval decisions of the
// trailing tool message that continue a suspended turn.
//
// Clients send approval decisions as a trailing tool message with approval
// response parts. A response only carries the approval ID, so the matching
// request is recovered from the assistant messages to find the original tool
// call. Responses whose call already has a tool result are skipped, because
// a previous continuation consumed them; an execution denial does not count
// as a result, since clients synthesize one for a newly denied approval.
func CollectToolApprovalContinuations(messages []model.Message) ([]model.ToolApprovalResponsePart, error) {
	toolMessage, ok := trailingToolMessage(messages)
	if !ok {
		return nil, nil
	}

	toolCallIDs, approvalRequests := assistantToolHistory(messages)
	completedToolCallIDs := completedToolResults(toolMessage)

	var continuations []model.ToolApprovalResponsePart

	for _, part := range toolMessage.Content {
		response, ok := part.(model.ToolApprovalResponsePart)
		if !ok {
			continue
		}

		request, known := approvalRequests[response.ApprovalID]
		if !known {
			return nil, &Error{
				Message: fmt.Sprintf("Tool approval response '%s' does not match a prior tool approval request.", response.ApprovalID),
			}
		}

		if completedToolCallIDs[request.ToolCallID] {
			continue
		}

		if !toolCallIDs[request.ToolCallID] {
			return nil, &Error{
				Message: fmt.Sprintf(
					"Tool approval request '%s' references unknown tool call '%s'.",
					request.ApprovalID,
					request.ToolCallID,
				),
			}
		}

		continuations = append(continuations, response)
	}

	return continuations, nil
}

// CollectToolResultContinuations returns the tool results of the trailing
// tool message, which provide results for pending application tool calls.
func CollectToolResultContinuations(messages []model.Message) []model.ToolResultPart {
	toolMessage, ok := trailingToolMessage(messages)
	if !ok {
		return nil
	}

	var results []model.ToolResultPart

	for _, part := range toolMessage.Content {
		if result, ok := part.(model.ToolResultPart); ok {
			results = append(results, result)
		}
	}

	return results
}

// trailingToolMessage returns the last message when it is a tool message.
func trailingToolMessage(messages []model.Message) (model.ToolMessage, bool) {
	if len(messages) == 0 {
		return model.ToolMessage{}, false
	}

	switch last := messages[len(messages)-1].(type) {
	case model.ToolMessage:
		return last, true
	case *model.ToolMessage:
		return *last, last != nil
	default:
		return model.ToolMessage{}, false
	}
}

// assistantToolHistory returns the tool call IDs and the approval requests of
// the assistant messages.
func assistantToolHistory(messages []model.Message) (map[string]bool, map[string]model.ToolApprovalRequestPart) {
	toolCallIDs := make(map[string]bool)
	approvalRequests := make(map[string]model.ToolApprovalRequestPart)

	for _, message := range messages {
		assistant, ok := asAssistantMessage(message)
		if !ok {
			continue
		}

		for _, part := range assistant.Content {
			switch typed := part.(type) {
			case model.ToolCallPart:
				toolCallIDs[typed.ToolCallID] = true
			case model.ToolApprovalRequestPart:
				approvalRequests[typed.ApprovalID] = typed
			}
		}
	}

	return toolCallIDs, approvalRequests
}

// completedToolResults returns the tool call IDs with a result other than an
// execution denial.
func completedToolResults(message model.ToolMessage) map[string]bool {
	completed := make(map[string]bool)

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			continue
		}

		if _, denied := result.Output.(model.ExecutionDeniedOutput); denied {
			continue
		}

		completed[result.ToolCallID] = true
	}

	return completed
}

// asAssistantMessage returns message when it is an assistant message.
func asAssistantMessage(message model.Message) (model.AssistantMessage, bool) {
	switch typed := message.(type) {
	case model.AssistantMessage:
		return typed, true
	case *model.AssistantMessage:
		if typed == nil {
			return model.AssistantMessage{}, false
		}

		return *typed, true
	default:
		return model.AssistantMessage{}, false
	}
}
