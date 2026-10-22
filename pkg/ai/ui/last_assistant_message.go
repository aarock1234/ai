package ui

// LastAssistantMessageIsCompleteWithToolCalls reports whether the last
// message is an assistant message whose last step has at least one
// application tool call and a final output or error for each of them. Text
// that is still streaming after the last tool call prevents resubmission.
func LastAssistantMessageIsCompleteWithToolCalls(messages []Message) bool {
	parts, ok := lastAssistantStepParts(messages)
	if !ok {
		return false
	}

	lastToolIndex := -1
	toolCalls := 0

	for index, part := range parts {
		tool, isTool := part.(ToolPart)
		if !isTool || tool.ProviderExecuted {
			continue
		}

		if !hasFinalOutput(tool) {
			return false
		}

		lastToolIndex = index
		toolCalls++
	}

	if toolCalls == 0 {
		return false
	}

	for _, part := range parts[lastToolIndex+1:] {
		if text, isText := part.(TextPart); isText && text.State != PartDone {
			return false
		}
	}

	return true
}

// LastAssistantMessageIsCompleteWithApprovalResponses reports whether the
// last message is an assistant message whose last step has at least one
// approval response, and whose tool calls all have a response or a final
// output.
func LastAssistantMessageIsCompleteWithApprovalResponses(messages []Message) bool {
	parts, ok := lastAssistantStepParts(messages)
	if !ok {
		return false
	}

	responses := 0

	for _, part := range parts {
		tool, isTool := part.(ToolPart)
		if !isTool {
			continue
		}

		switch {
		case tool.State == ToolApprovalResponded:
			responses++
		case tool.State == ToolOutputDenied, hasFinalOutput(tool):
		default:
			return false
		}
	}

	return responses > 0
}

// lastAssistantStepParts returns the parts of the last step of the last
// message when it is an assistant message.
func lastAssistantStepParts(messages []Message) ([]Part, bool) {
	if len(messages) == 0 {
		return nil, false
	}

	message := messages[len(messages)-1]
	if message.Role != RoleAssistant {
		return nil, false
	}

	return message.Parts[lastStepStart(message.Parts):], true
}

// hasFinalOutput reports whether a tool part has a final output or error.
func hasFinalOutput(part ToolPart) bool {
	switch part.State {
	case ToolOutputAvailable:
		return !part.Preliminary
	case ToolOutputError:
		return true
	default:
		return false
	}
}
