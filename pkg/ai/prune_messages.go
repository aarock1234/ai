package ai

import (
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// ReasoningPruning selects the assistant messages whose reasoning
// PruneMessages removes.
type ReasoningPruning string

// Reasoning pruning modes. The empty mode keeps all reasoning.
const (
	PruneReasoningNone              ReasoningPruning = "none"
	PruneReasoningAll               ReasoningPruning = "all"
	PruneReasoningBeforeLastMessage ReasoningPruning = "before-last-message"
)

// PruneMessagesOptions configures PruneMessages.
type PruneMessagesOptions struct {
	Messages []model.Message

	// Reasoning removes reasoning and reasoning file parts from assistant
	// messages.
	Reasoning ReasoningPruning

	// ToolCalls are applied in order. Each removes tool calls, tool results
	// and approvals.
	ToolCalls []ToolCallPruning

	// KeepEmptyMessages keeps messages left without content. By default they
	// are removed.
	KeepEmptyMessages bool
}

// ToolCallPruning removes tool calls, tool results and tool approvals from
// the messages before the last KeepLastMessages messages.
type ToolCallPruning struct {
	// KeepLastMessages is the number of trailing messages left untouched.
	// Tool calls with results or approvals in those messages are kept
	// everywhere. Zero prunes every message.
	KeepLastMessages int

	// Tools restricts pruning to the named tools. Nil prunes every tool.
	Tools []string
}

// PruneMessages removes reasoning and tool content from a conversation, for
// example to reduce the prompt size of long conversations. The input
// messages are not modified.
func PruneMessages(options PruneMessagesOptions) []model.Message {
	messages := slices.Clone(options.Messages)

	if options.Reasoning == PruneReasoningAll || options.Reasoning == PruneReasoningBeforeLastMessage {
		messages = pruneReasoning(messages, options.Reasoning)
	}

	for _, pruning := range options.ToolCalls {
		messages = pruneToolCalls(messages, pruning)
	}

	if options.KeepEmptyMessages {
		return messages
	}

	return slices.DeleteFunc(messages, isEmptyMessage)
}

func pruneReasoning(messages []model.Message, mode ReasoningPruning) []model.Message {
	for index, message := range messages {
		assistant, ok := message.(model.AssistantMessage)
		isLastMessage := index == len(messages)-1

		if !ok || (mode == PruneReasoningBeforeLastMessage && isLastMessage) {
			continue
		}

		assistant.Content = slices.DeleteFunc(slices.Clone(assistant.Content), isReasoningPart)
		messages[index] = assistant
	}

	return messages
}

func isReasoningPart(part model.AssistantPart) bool {
	switch part.(type) {
	case model.ReasoningPart, model.ReasoningFilePart:
		return true
	default:
		return false
	}
}

// toolContentIndex resolves tool names and approvals across messages,
// because approval responses live in tool messages while their requests
// live in assistant messages.
type toolContentIndex struct {
	keptToolCallIDs  map[string]bool
	keptApprovalIDs  map[string]bool
	toolNameByCallID map[string]string
	callIDByApproval map[string]string
}

func pruneToolCalls(messages []model.Message, pruning ToolCallPruning) []model.Message {
	index := indexToolContent(messages, pruning.KeepLastMessages)
	firstKeptMessage := len(messages) - pruning.KeepLastMessages

	for position, message := range messages {
		if pruning.KeepLastMessages > 0 && position >= firstKeptMessage {
			continue
		}

		switch typed := message.(type) {
		case model.AssistantMessage:
			keep := make([]bool, len(typed.Content))
			for partIndex, part := range typed.Content {
				keep[partIndex] = index.keepAssistantPart(part, pruning.Tools)
			}

			dropOrphanedReasoning(typed.Content, keep)
			typed.Content = filterParts(typed.Content, keep)
			messages[position] = typed

		case model.ToolMessage:
			keep := make([]bool, len(typed.Content))
			for partIndex, part := range typed.Content {
				keep[partIndex] = index.keepToolPart(part, pruning.Tools)
			}

			typed.Content = filterParts(typed.Content, keep)
			messages[position] = typed
		}
	}

	return messages
}

func indexToolContent(messages []model.Message, keepLastMessages int) toolContentIndex {
	index := toolContentIndex{
		keptToolCallIDs:  make(map[string]bool),
		keptApprovalIDs:  make(map[string]bool),
		toolNameByCallID: make(map[string]string),
		callIDByApproval: make(map[string]string),
	}

	if keepLastMessages > 0 {
		for _, message := range messages[max(len(messages)-keepLastMessages, 0):] {
			index.markKept(message)
		}
	}

	for _, message := range messages {
		forEachToolContent(message, func(part any) {
			switch typed := part.(type) {
			case model.ToolCallPart:
				index.toolNameByCallID[typed.ToolCallID] = typed.ToolName
			case model.ToolResultPart:
				index.toolNameByCallID[typed.ToolCallID] = typed.ToolName
			case model.ToolApprovalRequestPart:
				index.callIDByApproval[typed.ApprovalID] = typed.ToolCallID
			}
		})
	}

	// An approval kept in the trailing messages needs the call it approves.
	for approvalID := range index.keptApprovalIDs {
		if toolCallID, ok := index.callIDByApproval[approvalID]; ok {
			index.keptToolCallIDs[toolCallID] = true
		}
	}

	return index
}

func (i toolContentIndex) markKept(message model.Message) {
	forEachToolContent(message, func(part any) {
		switch typed := part.(type) {
		case model.ToolCallPart:
			i.keptToolCallIDs[typed.ToolCallID] = true
		case model.ToolResultPart:
			i.keptToolCallIDs[typed.ToolCallID] = true
		case model.ToolApprovalRequestPart:
			i.keptApprovalIDs[typed.ApprovalID] = true
		case model.ToolApprovalResponsePart:
			i.keptApprovalIDs[typed.ApprovalID] = true
		}
	})
}

// forEachToolContent calls visit with every part of assistant and tool
// messages.
func forEachToolContent(message model.Message, visit func(part any)) {
	switch typed := message.(type) {
	case model.AssistantMessage:
		for _, part := range typed.Content {
			visit(part)
		}
	case model.ToolMessage:
		for _, part := range typed.Content {
			visit(part)
		}
	}
}

func (i toolContentIndex) keepAssistantPart(part model.AssistantPart, tools []string) bool {
	switch typed := part.(type) {
	case model.ToolCallPart:
		return i.keepToolCall(typed.ToolCallID, typed.ToolName, tools)
	case model.ToolResultPart:
		return i.keepToolCall(typed.ToolCallID, typed.ToolName, tools)
	case model.ToolApprovalRequestPart:
		return i.keepApproval(typed.ApprovalID, tools)
	default:
		return true
	}
}

func (i toolContentIndex) keepToolPart(part model.ToolPart, tools []string) bool {
	switch typed := part.(type) {
	case model.ToolResultPart:
		return i.keepToolCall(typed.ToolCallID, typed.ToolName, tools)
	case model.ToolApprovalResponsePart:
		return i.keepApproval(typed.ApprovalID, tools)
	default:
		return true
	}
}

func (i toolContentIndex) keepToolCall(toolCallID, toolName string, tools []string) bool {
	if i.keptToolCallIDs[toolCallID] {
		return true
	}

	return isUnprunedTool(toolName, tools)
}

func (i toolContentIndex) keepApproval(approvalID string, tools []string) bool {
	if i.keptApprovalIDs[approvalID] {
		return true
	}

	toolName, known := i.toolNameByCallID[i.callIDByApproval[approvalID]]
	if !known {
		return false
	}

	return isUnprunedTool(toolName, tools)
}

// isUnprunedTool reports whether a tool falls outside a restricted pruning.
func isUnprunedTool(toolName string, tools []string) bool {
	return tools != nil && !slices.Contains(tools, toolName)
}

// dropOrphanedReasoning removes groups of consecutive reasoning parts whose
// following content was entirely pruned, because they belong to the same
// provider item.
func dropOrphanedReasoning(content []model.AssistantPart, keep []bool) {
	for start := 0; start < len(content); start++ {
		if !isReasoningPart(content[start]) {
			continue
		}

		end := start + 1
		for end < len(content) && isReasoningPart(content[end]) {
			end++
		}

		following := end
		hasFollowing := false
		keptFollowing := false

		for following < len(content) && !isReasoningPart(content[following]) {
			hasFollowing = true
			keptFollowing = keptFollowing || keep[following]
			following++
		}

		if hasFollowing && !keptFollowing {
			for index := start; index < end; index++ {
				keep[index] = false
			}
		}

		start = following - 1
	}
}

func filterParts[Part any](parts []Part, keep []bool) []Part {
	filtered := make([]Part, 0, len(parts))
	for index, part := range parts {
		if keep[index] {
			filtered = append(filtered, part)
		}
	}

	return filtered
}

func isEmptyMessage(message model.Message) bool {
	switch typed := message.(type) {
	case model.SystemMessage:
		return typed.Content == ""
	case model.UserMessage:
		return len(typed.Content) == 0
	case model.AssistantMessage:
		return len(typed.Content) == 0
	case model.ToolMessage:
		return len(typed.Content) == 0
	default:
		return false
	}
}
