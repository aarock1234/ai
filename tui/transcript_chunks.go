package tui

import (
	"github.com/aarock1234/ai/pkg/ai/ui"
)

// applyChunk renders one chunk of a remote agent's UI message stream into
// the transcript, like apply renders the parts of a local stream.
func (t *transcript) applyChunk(chunk ui.Chunk) {
	switch typed := chunk.(type) {
	case ui.StartStepChunk:
		t.step++
	case ui.TextDeltaChunk:
		t.appendBlock(sectionAssistant, typed.ID, typed.Delta)
	case ui.ReasoningDeltaChunk:
		t.appendBlock(sectionReasoning, typed.ID, typed.Delta)
	case ui.ToolInputStartChunk:
		t.startRemoteToolInput(typed)
	case ui.ToolInputDeltaChunk:
		t.updateTool(typed.ToolCallID, func(tool *toolInvocation) {
			tool.input += typed.InputTextDelta
		})
	case ui.ToolInputAvailableChunk:
		t.receiveToolInput(toolInput{
			toolCallID: typed.ToolCallID,
			name:       typed.ToolName,
			title:      typed.Title,
			input:      typed.Input,
		})
	case ui.ToolInputErrorChunk:
		t.receiveInvalidToolInput(typed)
	default:
		t.applyToolOutcomeChunk(chunk)
	}
}

// applyToolOutcomeChunk renders the chunks of approvals, tool outputs and
// errors.
func (t *transcript) applyToolOutcomeChunk(chunk ui.Chunk) {
	switch typed := chunk.(type) {
	case ui.ToolApprovalRequestChunk:
		t.approvalTools[typed.ApprovalID] = typed.ToolCallID

		t.updateTool(typed.ToolCallID, func(tool *toolInvocation) {
			tool.state = toolApprovalRequested
		})
	case ui.ToolApprovalResponseChunk:
		if toolCallID, ok := t.approvalTools[typed.ApprovalID]; ok {
			t.recordApproval(toolCallID, typed.Approved, typed.Reason)
		}
	case ui.ToolOutputAvailableChunk:
		t.updateTool(typed.ToolCallID, func(tool *toolInvocation) {
			tool.state = toolOutputAvailable
			tool.output = formatJSON(typed.Output)
		})
	case ui.ToolOutputErrorChunk:
		t.updateTool(typed.ToolCallID, func(tool *toolInvocation) {
			tool.state = toolOutputError
			tool.errorText = typed.ErrorText
		})
	case ui.ToolOutputDeniedChunk:
		t.updateTool(typed.ToolCallID, func(tool *toolInvocation) {
			tool.state = toolOutputDenied
		})
	case ui.ErrorChunk:
		t.addError(typed.ErrorText)
	}
}

func (t *transcript) startRemoteToolInput(chunk ui.ToolInputStartChunk) {
	tool := t.ensureTool(chunk.ToolCallID, chunk.ToolName, chunk.Title)
	if tool == nil {
		return
	}

	tool.state = toolInputStreaming
	t.renderTool(tool)
}

// receiveInvalidToolInput shows a tool call whose input failed validation.
func (t *transcript) receiveInvalidToolInput(chunk ui.ToolInputErrorChunk) {
	t.receiveToolInput(toolInput{
		toolCallID: chunk.ToolCallID,
		name:       chunk.ToolName,
		title:      chunk.Title,
		input:      chunk.Input,
	})

	t.updateTool(chunk.ToolCallID, func(tool *toolInvocation) {
		tool.state = toolOutputError
		tool.errorText = chunk.ErrorText
	})
}
