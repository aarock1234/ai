package ui

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/merge"
	"github.com/aarock1234/ai/pkg/model"
)

// messageState builds the assistant message of a UI message stream, the
// upstream processUIMessageStream state. It is not safe for concurrent use.
type messageState struct {
	message Message

	// activeText and activeReasoning map the IDs of open parts to their
	// index in the message parts.
	activeText      map[string]int
	activeReasoning map[string]int

	// partialToolCalls holds the input text of tool calls whose input is
	// still streaming, keyed by tool call ID.
	partialToolCalls map[string]*partialToolCall

	finishReason model.UnifiedFinishReason

	// resetOnMessageIDChange starts a new message when a start chunk carries
	// a different message ID, as when a resumed stream belongs to another
	// response.
	resetOnMessageIDChange bool
}

// partialToolCall is a tool call whose input is still streaming.
type partialToolCall struct {
	text          string
	inputEncoding model.ToolInputEncoding
	toolName      string
	dynamic       bool
	title         string
	toolMetadata  map[string]any
}

// newMessageState continues lastMessage when it is an assistant message and
// otherwise starts an empty assistant message with messageID.
func newMessageState(lastMessage *Message, messageID string) *messageState {
	message := Message{
		ID:   messageID,
		Role: RoleAssistant,
	}

	if lastMessage != nil && lastMessage.Role == RoleAssistant {
		message = lastMessage.clone()
	}

	state := &messageState{
		message:          message,
		activeText:       make(map[string]int),
		activeReasoning:  make(map[string]int),
		partialToolCalls: make(map[string]*partialToolCall),
	}

	state.resumePartialToolCalls()

	return state
}

// rawInputText returns the input text of a streaming tool part: the decoded
// string when RawInput holds one, and otherwise the raw JSON itself.
func rawInputText(rawInput json.RawMessage) string {
	if rawInput == nil {
		return ""
	}

	var text string
	if err := json.Unmarshal(rawInput, &text); err != nil {
		return string(rawInput)
	}

	return text
}

// resumePartialToolCalls restores the streaming inputs of the current step,
// so that a resumed stream can continue them.
func (s *messageState) resumePartialToolCalls() {
	for _, part := range s.message.Parts[s.currentStepStart():] {
		tool, ok := part.(ToolPart)
		if !ok || tool.State != ToolInputStreaming {
			continue
		}

		s.partialToolCalls[tool.ToolCallID] = &partialToolCall{
			text:          rawInputText(tool.RawInput),
			inputEncoding: tool.InputEncoding,
			toolName:      tool.ToolName,
			dynamic:       tool.isDynamicPart(),
			title:         tool.Title,
			toolMetadata:  tool.ToolMetadata,
		}
	}
}

// apply updates the message with one chunk. It reports whether the message
// changed in a way the client renders.
func (s *messageState) apply(chunk Chunk) (bool, error) {
	switch typed := chunk.(type) {
	case TextStartChunk:
		s.activeText[typed.ID] = s.appendPart(TextPart{
			State:            PartStreaming,
			ProviderMetadata: typed.ProviderMetadata,
		})

		return true, nil

	case TextDeltaChunk:
		return s.updateText(typed.ID, "text-delta", func(part *TextPart) {
			part.Text += typed.Delta
			part.ProviderMetadata = metadataOr(typed.ProviderMetadata, part.ProviderMetadata)
		})

	case TextEndChunk:
		changed, err := s.updateText(typed.ID, "text-end", func(part *TextPart) {
			part.State = PartDone
			part.ProviderMetadata = metadataOr(typed.ProviderMetadata, part.ProviderMetadata)

			if typed.Citations != nil {
				part.Citations = typed.Citations
			}
		})
		delete(s.activeText, typed.ID)

		return changed, err

	case ReasoningStartChunk:
		s.activeReasoning[typed.ID] = s.appendPart(ReasoningPart{
			ID:               typed.ID,
			State:            PartStreaming,
			ProviderMetadata: typed.ProviderMetadata,
		})

		return true, nil

	case ReasoningDeltaChunk:
		return s.updateReasoning(typed.ID, "reasoning-delta", func(part *ReasoningPart) {
			part.Text += typed.Delta
			part.ProviderMetadata = metadataOr(typed.ProviderMetadata, part.ProviderMetadata)
		})

	case ReasoningEndChunk:
		changed, err := s.updateReasoning(typed.ID, "reasoning-end", func(part *ReasoningPart) {
			part.State = PartDone
			part.ProviderMetadata = metadataOr(typed.ProviderMetadata, part.ProviderMetadata)
		})
		delete(s.activeReasoning, typed.ID)

		return changed, err

	case CustomChunk:
		s.appendPart(CustomPart(typed))

		return true, nil

	case FileChunk:
		s.appendPart(FilePart{
			MediaType:        typed.MediaType,
			URL:              typed.URL,
			ProviderMetadata: typed.ProviderMetadata,
		})

		return true, nil

	case ReasoningFileChunk:
		s.appendPart(ReasoningFilePart{
			MediaType:        typed.MediaType,
			URL:              typed.URL,
			ProviderMetadata: typed.ProviderMetadata,
		})

		return true, nil

	case SourceURLChunk:
		s.appendPart(SourceURLPart(typed))

		return true, nil

	case SourceDocumentChunk:
		s.appendPart(SourceDocumentPart(typed))

		return true, nil

	case DataChunk:
		return s.applyData(typed), nil

	case StartStepChunk:
		s.appendPart(StepStartPart{})

		return false, nil

	case FinishStepChunk, ErrorChunk:
		return false, nil

	case ResetStepChunk:
		return s.resetStep(), nil

	case StartChunk:
		return s.applyStart(typed)

	case FinishChunk:
		if typed.FinishReason != "" {
			s.finishReason = typed.FinishReason
		}

		return s.updateMetadata(typed.MessageMetadata)

	case MessageMetadataChunk:
		return s.updateMetadata(typed.MessageMetadata)

	default:
		return s.applyTool(chunk)
	}
}

// applyTool applies the chunks that update tool parts.
func (s *messageState) applyTool(chunk Chunk) (bool, error) {
	switch typed := chunk.(type) {
	case ToolInputStartChunk:
		s.applyToolInputStart(typed)

		return true, nil

	case ToolInputDeltaChunk:
		return true, s.applyToolInputDelta(typed)

	case ToolInputAvailableChunk:
		s.updateToolPart(toolUpdate{
			toolName:         typed.ToolName,
			toolCallID:       typed.ToolCallID,
			dynamic:          typed.Dynamic,
			state:            ToolInputAvailable,
			input:            typed.Input,
			providerExecuted: typed.ProviderExecuted,
			providerMetadata: typed.ProviderMetadata,
			title:            typed.Title,
			toolMetadata:     typed.ToolMetadata,
		}, -1)

		return true, nil

	case ToolInputErrorChunk:
		s.applyToolInputError(typed)

		return true, nil

	case ToolApprovalRequestChunk:
		return true, s.applyApprovalRequest(typed)

	case ToolApprovalResponseChunk:
		return true, s.applyApprovalResponse(typed)

	case ToolOutputDeniedChunk:
		index, err := s.findToolPart(typed.ToolCallID, "tool-output-denied")
		if err != nil {
			return false, err
		}

		part := s.message.Parts[index].(ToolPart)
		part.State = ToolOutputDenied
		s.message.Parts[index] = part

		return true, nil

	case ToolOutputAvailableChunk:
		return true, s.applyToolOutputAvailable(typed)

	case ToolOutputErrorChunk:
		return true, s.applyToolOutputError(typed)

	default:
		return false, fmt.Errorf("unsupported chunk type %T", chunk)
	}
}

func (s *messageState) appendPart(part Part) int {
	s.message.Parts = append(s.message.Parts, part)

	return len(s.message.Parts) - 1
}

func (s *messageState) updateText(id, chunkType string, update func(part *TextPart)) (bool, error) {
	index, ok := s.activeText[id]
	if !ok {
		return false, missingPartError(chunkType, id, "text-start")
	}

	part := s.message.Parts[index].(TextPart)
	update(&part)
	s.message.Parts[index] = part

	return true, nil
}

func (s *messageState) updateReasoning(id, chunkType string, update func(part *ReasoningPart)) (bool, error) {
	index, ok := s.activeReasoning[id]
	if !ok {
		return false, missingPartError(chunkType, id, "reasoning-start")
	}

	part := s.message.Parts[index].(ReasoningPart)
	update(&part)
	s.message.Parts[index] = part

	return true, nil
}

func missingPartError(chunkType, id, startType string) error {
	return &MessageStreamError{
		ChunkType: chunkType,
		ChunkID:   id,
		Message:   fmt.Sprintf("no open part: send a %q chunk before %q chunks", startType, chunkType),
	}
}

// metadataOr returns metadata, or fallback when metadata is nil.
func metadataOr(metadata, fallback model.ProviderMetadata) model.ProviderMetadata {
	if metadata != nil {
		return metadata
	}

	return fallback
}

// applyData adds a data part or replaces the data of the part with the same
// name and ID. Transient chunks leave the message unchanged.
func (s *messageState) applyData(chunk DataChunk) bool {
	if chunk.Transient {
		return false
	}

	if chunk.ID != "" {
		for index, part := range s.message.Parts {
			existing, ok := part.(DataPart)
			if ok && existing.Name == chunk.Name && existing.ID == chunk.ID {
				existing.Data = chunk.Data
				s.message.Parts[index] = existing

				return true
			}
		}
	}

	s.appendPart(DataPart{
		Name: chunk.Name,
		ID:   chunk.ID,
		Data: chunk.Data,
	})

	return true
}

// resetStep removes the parts of the current step.
func (s *messageState) resetStep() bool {
	start := s.currentStepStart()

	s.activeText = make(map[string]int)
	s.activeReasoning = make(map[string]int)
	s.partialToolCalls = make(map[string]*partialToolCall)

	if start == len(s.message.Parts) {
		return false
	}

	s.message.Parts = s.message.Parts[:start]

	return true
}

func (s *messageState) applyStart(chunk StartChunk) (bool, error) {
	shouldReset := s.resetOnMessageIDChange && chunk.MessageID != "" && chunk.MessageID != s.message.ID

	if shouldReset {
		fresh := newMessageState(nil, chunk.MessageID)
		fresh.resetOnMessageIDChange = s.resetOnMessageIDChange
		*s = *fresh
	}

	if chunk.MessageID != "" {
		s.message.ID = chunk.MessageID
	}

	metadataChanged, err := s.updateMetadata(chunk.MessageMetadata)
	if err != nil {
		return false, err
	}

	return shouldReset || chunk.MessageID != "" || metadataChanged, nil
}

// updateMetadata merges object metadata into the message metadata and
// replaces other metadata. Absent and null metadata leave it unchanged.
func (s *messageState) updateMetadata(metadata json.RawMessage) (bool, error) {
	if isJSONNull(metadata) {
		return false, nil
	}

	if isJSONNull(s.message.Metadata) {
		s.message.Metadata = metadata

		return true, nil
	}

	base, baseIsObject := merge.Object(s.message.Metadata)
	overrides, overridesAreObject := merge.Object(metadata)

	if !baseIsObject || !overridesAreObject {
		s.message.Metadata = metadata

		return true, nil
	}

	merged, err := encodeJSON(merge.Deep(base, overrides))
	if err != nil {
		return false, fmt.Errorf("encode merged message metadata: %w", err)
	}

	s.message.Metadata = merged

	return true, nil
}

// currentStepStart returns the index of the first part of the current step.
func (s *messageState) currentStepStart() int {
	return lastStepStart(s.message.Parts)
}

// lastStepStart returns the index of the first part after the last
// step-start part.
func lastStepStart(parts []Part) int {
	for index, part := range slices.Backward(parts) {
		if _, isStepStart := part.(StepStartPart); isStepStart {
			return index + 1
		}
	}

	return 0
}
