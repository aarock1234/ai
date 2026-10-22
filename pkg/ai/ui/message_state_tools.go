package ui

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/partialjson"
	"github.com/aarock1234/ai/pkg/model"
)

// toolUpdate is the new state of a tool part. Absent values clear the
// corresponding fields, except Title and ToolMetadata, which are kept when
// empty, and ProviderExecuted, which stays set once set.
type toolUpdate struct {
	toolName   string
	toolCallID string
	dynamic    bool
	state      ToolState

	input         json.RawMessage
	rawInput      json.RawMessage
	inputEncoding model.ToolInputEncoding
	output        json.RawMessage
	errorText     string
	preliminary   bool

	providerExecuted bool
	providerMetadata model.ProviderMetadata
	title            string
	toolMetadata     map[string]any
}

// updateToolPart applies update to the part at index, or to the current
// step's part with the same tool call ID and encoding when index is
// negative. It appends a new part when there is none.
func (s *messageState) updateToolPart(update toolUpdate, index int) {
	if index < 0 {
		index = s.findCurrentStepToolPart(update.toolCallID, update.dynamic)
	}

	if index < 0 {
		s.appendPart(newToolPart(update))

		return
	}

	part := s.message.Parts[index].(ToolPart)
	part.ToolName = update.toolName
	part.State = update.state
	part.Input = update.input
	part.RawInput = update.rawInput
	part.InputEncoding = update.inputEncoding
	part.Output = update.output
	part.ErrorText = update.errorText
	part.Preliminary = update.preliminary
	part.ProviderExecuted = part.ProviderExecuted || update.providerExecuted

	if update.title != "" {
		part.Title = update.title
	}

	if update.toolMetadata != nil {
		part.ToolMetadata = update.toolMetadata
	}

	if update.providerMetadata != nil {
		if isOutputState(update.state) {
			part.ResultProviderMetadata = update.providerMetadata
		} else {
			part.CallProviderMetadata = update.providerMetadata
		}
	}

	s.message.Parts[index] = part
}

func newToolPart(update toolUpdate) ToolPart {
	part := ToolPart{
		ToolName:         update.toolName,
		ToolCallID:       update.toolCallID,
		Dynamic:          update.dynamic,
		State:            update.state,
		Title:            update.title,
		ToolMetadata:     update.toolMetadata,
		ProviderExecuted: update.providerExecuted,
		Input:            update.input,
		RawInput:         update.rawInput,
		InputEncoding:    update.inputEncoding,
		Output:           update.output,
		ErrorText:        update.errorText,
		Preliminary:      update.preliminary,
	}

	if isOutputState(update.state) {
		part.ResultProviderMetadata = update.providerMetadata
	} else {
		part.CallProviderMetadata = update.providerMetadata
	}

	return part
}

// findCurrentStepToolPart returns the index of the current step's tool part
// with the tool call ID and the given encoding, or -1.
func (s *messageState) findCurrentStepToolPart(toolCallID string, dynamic bool) int {
	for index := s.currentStepStart(); index < len(s.message.Parts); index++ {
		part, ok := s.message.Parts[index].(ToolPart)
		if ok && part.ToolCallID == toolCallID && part.isDynamicPart() == dynamic {
			return index
		}
	}

	return -1
}

// findToolPart returns the index of the tool part with the tool call ID,
// preferring the current step and otherwise searching from the end.
func (s *messageState) findToolPart(toolCallID, chunkType string) (int, error) {
	stepStart := s.currentStepStart()

	for index := stepStart; index < len(s.message.Parts); index++ {
		if part, ok := s.message.Parts[index].(ToolPart); ok && part.ToolCallID == toolCallID {
			return index, nil
		}
	}

	for index := len(s.message.Parts) - 1; index >= 0; index-- {
		if part, ok := s.message.Parts[index].(ToolPart); ok && part.ToolCallID == toolCallID {
			return index, nil
		}
	}

	return -1, &MessageStreamError{
		ChunkType: chunkType,
		ChunkID:   toolCallID,
		Message:   "no tool invocation found for the tool call ID",
	}
}

// findToolPartByApproval returns the index of the tool part whose approval
// has the ID.
func (s *messageState) findToolPartByApproval(approvalID string) (int, error) {
	for index, part := range s.message.Parts {
		tool, ok := part.(ToolPart)
		if ok && tool.Approval != nil && tool.Approval.ID == approvalID {
			return index, nil
		}
	}

	return -1, &MessageStreamError{
		ChunkType: "tool-approval-response",
		ChunkID:   approvalID,
		Message:   "no tool invocation found for the approval ID",
	}
}

func (s *messageState) applyToolInputStart(chunk ToolInputStartChunk) {
	s.partialToolCalls[chunk.ToolCallID] = &partialToolCall{
		inputEncoding: chunk.InputEncoding,
		toolName:      chunk.ToolName,
		dynamic:       chunk.Dynamic,
		title:         chunk.Title,
		toolMetadata:  chunk.ToolMetadata,
	}

	s.updateToolPart(toolUpdate{
		toolName:         chunk.ToolName,
		toolCallID:       chunk.ToolCallID,
		dynamic:          chunk.Dynamic,
		state:            ToolInputStreaming,
		inputEncoding:    chunk.InputEncoding,
		providerExecuted: chunk.ProviderExecuted,
		providerMetadata: chunk.ProviderMetadata,
		title:            chunk.Title,
		toolMetadata:     chunk.ToolMetadata,
	}, -1)
}

// applyToolInputDelta appends to the input text and sets the part's input to
// the partial input parsed so far.
func (s *messageState) applyToolInputDelta(chunk ToolInputDeltaChunk) error {
	partial, ok := s.partialToolCalls[chunk.ToolCallID]
	if !ok {
		return missingPartError("tool-input-delta", chunk.ToolCallID, "tool-input-start")
	}

	partial.text += chunk.InputTextDelta

	rawInput, err := encodeJSON(partial.text)
	if err != nil {
		return fmt.Errorf("encode tool input text: %w", err)
	}

	input := rawInput
	if partial.inputEncoding != model.ToolInputText {
		input, _ = partialjson.Parse(partial.text)
	}

	s.updateToolPart(toolUpdate{
		toolName:      partial.toolName,
		toolCallID:    chunk.ToolCallID,
		dynamic:       partial.dynamic,
		state:         ToolInputStreaming,
		input:         input,
		rawInput:      rawInput,
		inputEncoding: partial.inputEncoding,
		title:         partial.title,
		toolMetadata:  partial.toolMetadata,
	}, -1)

	return nil
}

// applyToolInputError keeps the encoding of an existing part for the call,
// so that the part is updated in place.
func (s *messageState) applyToolInputError(chunk ToolInputErrorChunk) {
	dynamic := chunk.Dynamic

	for index := s.currentStepStart(); index < len(s.message.Parts); index++ {
		part, ok := s.message.Parts[index].(ToolPart)
		if ok && part.ToolCallID == chunk.ToolCallID {
			dynamic = part.isDynamicPart()

			break
		}
	}

	s.updateToolPart(toolUpdate{
		toolName:         chunk.ToolName,
		toolCallID:       chunk.ToolCallID,
		dynamic:          dynamic,
		state:            ToolOutputError,
		input:            chunk.Input,
		errorText:        chunk.ErrorText,
		providerExecuted: chunk.ProviderExecuted,
		providerMetadata: chunk.ProviderMetadata,
		toolMetadata:     chunk.ToolMetadata,
	}, -1)
}

func (s *messageState) applyApprovalRequest(chunk ToolApprovalRequestChunk) error {
	index, err := s.findToolPart(chunk.ToolCallID, "tool-approval-request")
	if err != nil {
		return err
	}

	part := s.message.Parts[index].(ToolPart)
	part.State = ToolApprovalRequested
	part.Approval = &ToolApproval{
		ID:               chunk.ApprovalID,
		Descriptor:       chunk.ApprovalDescriptor,
		RequestReason:    chunk.Reason,
		IsAutomatic:      chunk.IsAutomatic,
		Signature:        chunk.Signature,
		InputSchemaInput: chunk.InputSchemaInput,
	}
	s.message.Parts[index] = part

	return nil
}

func (s *messageState) applyApprovalResponse(chunk ToolApprovalResponseChunk) error {
	index, err := s.findToolPartByApproval(chunk.ApprovalID)
	if err != nil {
		return err
	}

	part := s.message.Parts[index].(ToolPart)

	approval := ToolApproval{
		ID: chunk.ApprovalID,
	}
	if part.Approval != nil {
		approval = *part.Approval
	}

	approval.Approved = new(chunk.Approved)
	if chunk.Reason != "" {
		approval.Reason = chunk.Reason
	}

	part.State = ToolApprovalResponded
	part.Approval = &approval
	part.ProviderExecuted = part.ProviderExecuted || chunk.ProviderExecuted

	if chunk.ProviderMetadata != nil {
		part.CallProviderMetadata = chunk.ProviderMetadata
	}

	s.message.Parts[index] = part

	return nil
}

func (s *messageState) applyToolOutputAvailable(chunk ToolOutputAvailableChunk) error {
	index, err := s.findToolPart(chunk.ToolCallID, "tool-output-available")
	if err != nil {
		return err
	}

	part := s.message.Parts[index].(ToolPart)

	s.updateToolPart(toolUpdate{
		toolName:         part.ToolName,
		toolCallID:       chunk.ToolCallID,
		dynamic:          part.isDynamicPart(),
		state:            ToolOutputAvailable,
		input:            part.Input,
		output:           chunk.Output,
		preliminary:      chunk.Preliminary,
		providerExecuted: chunk.ProviderExecuted,
		providerMetadata: chunk.ProviderMetadata,
		title:            part.Title,
		toolMetadata:     toolMetadataOr(chunk.ToolMetadata, part.ToolMetadata),
	}, index)

	return nil
}

// applyToolOutputError records the failure. Static parts keep their raw
// input, which older clients used for failed calls.
func (s *messageState) applyToolOutputError(chunk ToolOutputErrorChunk) error {
	index, err := s.findToolPart(chunk.ToolCallID, "tool-output-error")
	if err != nil {
		return err
	}

	part := s.message.Parts[index].(ToolPart)

	var rawInput json.RawMessage
	if !part.isDynamicPart() {
		rawInput = part.RawInput
	}

	s.updateToolPart(toolUpdate{
		toolName:         part.ToolName,
		toolCallID:       chunk.ToolCallID,
		dynamic:          part.isDynamicPart(),
		state:            ToolOutputError,
		input:            part.Input,
		rawInput:         rawInput,
		errorText:        chunk.ErrorText,
		providerExecuted: chunk.ProviderExecuted,
		providerMetadata: chunk.ProviderMetadata,
		title:            part.Title,
		toolMetadata:     toolMetadataOr(chunk.ToolMetadata, part.ToolMetadata),
	}, index)

	return nil
}

func toolMetadataOr(metadata, fallback map[string]any) map[string]any {
	if metadata != nil {
		return metadata
	}

	return fallback
}
