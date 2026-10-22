package anthropic

import (
	"errors"
	"strconv"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// streamBlockKind is the kind of output a streamed content block produces.
type streamBlockKind int

const (
	streamText streamBlockKind = iota
	streamReasoning
	streamToolCall
)

// streamBlock is an open content block.
type streamBlock struct {
	kind streamBlockKind

	// citations are collected for text blocks.
	citations []Citation

	toolCallID       string
	toolName         string
	input            string
	providerExecuted bool
	dynamic          bool

	// firstDelta marks a tool call whose input has not started yet.
	firstDelta bool

	// providerToolName is the API name of a provider-executed tool.
	providerToolName string

	// inputType is injected as the input's type by the first delta, for
	// code execution subtools and programmatic calls.
	inputType string

	// toolsetName and memberName identify a toolset member call, whose
	// input is buffered and emitted with the member as its action.
	toolsetName string
	memberName  string

	caller *ToolCaller
}

func blockID(index int) string {
	return strconv.Itoa(index)
}

// blockStart opens a content block. Provider tool results arrive complete
// and are emitted at once.
func (s *streamState) blockStart(event *contentBlockStartEvent) []model.StreamPart {
	if fallback, ok := event.ContentBlock.(*responseFallback); ok {
		return []model.StreamPart{
			fallbackContent(fallback),
		}
	}

	_, s.inThinkingBlock = event.ContentBlock.(*responseThinking)
	index := event.Index

	switch block := event.ContentBlock.(type) {
	case *responseText:
		if s.converter.usesJSONResponseTool {
			return nil
		}

		return s.openText(index, nil)

	case *responseThinking:
		return s.openReasoning(index, nil)

	case *responseRedactedThinking:
		return s.openReasoning(index, redactedMetadata(block.Data))

	case *responseCompaction:
		return s.openCompaction(index, block)

	case *responseToolUse:
		return s.openToolUse(index, block)

	case *responseServerToolUse:
		return s.openServerToolUse(index, block)

	case *responseWebFetchToolResult:
		return toolResultParts(s.converter.webFetchResult(block))

	case *responseWebSearchToolResult:
		return s.webSearchResultParts(block)

	case *responseCodeExecutionToolResult:
		return toolResultParts(s.converter.codeExecutionResult(block))

	case *responseSubtoolResult:
		return toolResultParts(s.converter.subtoolResult(block))

	case *responseToolSearchToolResult:
		return toolResultParts(s.converter.toolSearchResult(block))

	case *responseAdvisorToolResult:
		return toolResultParts(s.converter.advisorResult(block))

	case *responseMCPToolUse:
		return []model.StreamPart{
			s.converter.mcpToolCall(block),
		}

	case *responseMCPToolResult:
		return toolResultParts(s.converter.mcpToolResult(block))

	case *responseContainerUpload:
		return []model.StreamPart{
			model.Custom{
				Kind: containerUploadKind,
				ProviderMetadata: model.ProviderMetadata{
					Namespace: ContainerUploadMetadata{
						FileID: block.FileID,
					},
				},
			},
		}

	default:
		return nil
	}
}

func toolResultParts(result model.ToolResult) []model.StreamPart {
	return []model.StreamPart{
		result,
	}
}

func (s *streamState) openText(index int, metadata model.ProviderMetadata) []model.StreamPart {
	s.blocks[index] = &streamBlock{
		kind: streamText,
	}

	return []model.StreamPart{
		model.TextStart{
			ID:               blockID(index),
			ProviderMetadata: metadata,
		},
	}
}

func (s *streamState) openReasoning(index int, metadata model.ProviderMetadata) []model.StreamPart {
	s.blocks[index] = &streamBlock{
		kind: streamReasoning,
	}

	return []model.StreamPart{
		model.ReasoningStart{
			ID:               blockID(index),
			ProviderMetadata: metadata,
		},
	}
}

// openCompaction opens a compaction summary as text. On-demand compaction
// blocks arrive complete, without compaction deltas.
func (s *streamState) openCompaction(index int, block *responseCompaction) []model.StreamPart {
	parts := s.openText(index, compactionMetadata(block.Signature))

	content := valueOrZero(block.Content)
	if block.Signature != nil && content != "" {
		parts = append(parts, model.TextDelta{
			ID:    blockID(index),
			Delta: content,
		})
	}

	return parts
}

// openToolUse opens a client tool call. With the JSON response tool, the
// call streams as text. Deferred programmatic calls may carry their input
// already.
func (s *streamState) openToolUse(index int, block *responseToolUse) []model.StreamPart {
	if s.converter.isJSONResponseTool(block.Name) {
		s.jsonFromTool = true

		return s.openText(index, nil)
	}

	initialInput := nonEmptyInput(block.Input)
	opened := &streamBlock{
		kind:       streamToolCall,
		toolCallID: block.ID,
		toolName:   block.Name,
		input:      initialInput,
		firstDelta: initialInput == "",
		caller:     toolCallerOf(block.Caller),
	}

	toolsetName := valueOrZero(block.ToolsetName)
	if toolsetName != "" {
		opened.toolName = s.converter.toolNames.ToApplication(toolsetName)
		opened.firstDelta = true
		opened.toolsetName = toolsetName
		opened.memberName = block.Name
	}

	s.blocks[index] = opened

	return []model.StreamPart{
		model.ToolInputStart{
			ID:       block.ID,
			ToolName: opened.toolName,
		},
	}
}

// nonEmptyInput returns input that is a non-empty object as compact JSON,
// and otherwise an empty string, meaning the input arrives as deltas.
func nonEmptyInput(input []byte) string {
	object, err := jsonobject.Parse(input)
	if err != nil || len(object) == 0 {
		return ""
	}

	return compactInput(input)
}

// openServerToolUse opens a provider-executed tool call.
func (s *streamState) openServerToolUse(index int, block *responseServerToolUse) []model.StreamPart {
	opened := &streamBlock{
		kind:             streamToolCall,
		toolCallID:       block.ID,
		providerExecuted: true,
		firstDelta:       true,
		providerToolName: block.Name,
		caller:           toolCallerOf(block.Caller),
	}

	switch block.Name {
	case nameWebFetch, nameWebSearch:
		// Dynamic web tools provide their input here.
		opened.input = nonEmptyInput(block.Input)

	case nameCodeExecution:
		opened.input = nonEmptyInput(block.Input)
		opened.inputType = programmaticCallType

	case nameTextEditorCodeExecution, nameBashCodeExecution:
		opened.input = nonEmptyInput(block.Input)
		opened.inputType = block.Name
		opened.providerToolName = nameCodeExecution

	case nameToolSearchRegex, nameToolSearchBM25:
		s.converter.serverToolCalls[block.ID] = block.Name

	case nameAdvisor:
		opened.input = "{}"

	default:
		return nil
	}

	opened.firstDelta = opened.input == "" || block.Name == nameAdvisor
	opened.toolName = s.converter.toolNames.ToApplication(opened.providerToolName)
	opened.dynamic = s.converter.markCodeExecutionDynamic && opened.providerToolName == nameCodeExecution

	s.blocks[index] = opened

	return []model.StreamPart{
		model.ToolInputStart{
			ID:               block.ID,
			ToolName:         opened.toolName,
			ProviderExecuted: true,
			Dynamic:          opened.dynamic,
		},
	}
}

func (s *streamState) webSearchResultParts(block *responseWebSearchToolResult) []model.StreamPart {
	result, sources := s.converter.webSearchResult(block)

	parts := []model.StreamPart{
		result,
	}

	for _, source := range sources {
		s.hasWebSearchResults = true
		parts = append(parts, source)
	}

	return parts
}

// blockDelta appends to an open content block.
func (s *streamState) blockDelta(event *contentBlockDeltaEvent) []model.StreamPart {
	index := event.Index
	delta := event.Delta

	switch delta.Type {
	case deltaText:
		if s.converter.usesJSONResponseTool {
			return nil
		}

		return textDelta(index, delta.Text)

	case deltaThinking:
		return []model.StreamPart{
			model.ReasoningDelta{
				ID:    blockID(index),
				Delta: delta.Thinking,
			},
		}

	case deltaSignature:
		if !s.inThinkingBlock {
			return nil
		}

		return []model.StreamPart{
			model.ReasoningDelta{
				ID:               blockID(index),
				ProviderMetadata: signatureMetadata(delta.Signature),
			},
		}

	case deltaCompaction:
		if delta.Content == nil {
			return nil
		}

		return textDelta(index, *delta.Content)

	case deltaInputJSON:
		return s.inputDelta(index, delta.PartialJSON)

	case deltaCitations:
		return s.citationDelta(index, *delta.Citation)

	default:
		return nil
	}
}

func textDelta(index int, text string) []model.StreamPart {
	return []model.StreamPart{
		model.TextDelta{
			ID:    blockID(index),
			Delta: text,
		},
	}
}

// inputDelta appends tool input. JSON response tool input streams as text.
// Empty deltas are skipped so that the first real delta can carry the
// injected input type.
func (s *streamState) inputDelta(index int, partialJSON string) []model.StreamPart {
	if partialJSON == "" {
		return nil
	}

	block := s.blocks[index]

	if s.jsonFromTool {
		if block == nil || block.kind != streamText {
			return nil
		}

		return textDelta(index, partialJSON)
	}

	if block == nil || block.kind != streamToolCall {
		return nil
	}

	if block.toolsetName != "" {
		block.input += partialJSON

		return nil
	}

	delta := partialJSON
	if block.firstDelta && block.inputType != "" {
		delta = `{"type": "` + block.inputType + `",` + delta[1:]
	}

	block.input += delta
	block.firstDelta = false

	return []model.StreamPart{
		model.ToolInputDelta{
			ID:    block.toolCallID,
			Delta: delta,
		},
	}
}

// citationDelta collects a citation for its text block. Web search
// citations become sources when the message stops; others become sources
// at once.
func (s *streamState) citationDelta(index int, citation Citation) []model.StreamPart {
	if block := s.blocks[index]; block != nil && block.kind == streamText {
		block.citations = append(block.citations, citation)
	}

	if citation.Type == CitationWebSearchResultLocation {
		s.fallbackWebCitations = append(s.fallbackWebCitations, citation)

		return nil
	}

	source, ok := citationSource(citation, s.converter.documents, s.converter.generateID)
	if !ok {
		return nil
	}

	return []model.StreamPart{
		source,
	}
}

// blockStop closes a content block. A tool call block emits the complete
// call.
func (s *streamState) blockStop(event *contentBlockStopEvent) []model.StreamPart {
	s.inThinkingBlock = false

	block, found := s.blocks[event.Index]
	if !found {
		return nil
	}

	delete(s.blocks, event.Index)
	id := blockID(event.Index)

	switch block.kind {
	case streamText:
		return []model.StreamPart{
			model.TextEnd{
				ID:               id,
				Citations:        mapCitations(block.citations, s.converter.documents),
				ProviderMetadata: citationsMetadata(block.citations),
			},
		}

	case streamReasoning:
		return []model.StreamPart{
			model.ReasoningEnd{
				ID: id,
			},
		}

	default:
		return s.finishToolCall(block)
	}
}

// finishToolCall ends a tool call's input and emits the call.
func (s *streamState) finishToolCall(block *streamBlock) []model.StreamPart {
	if s.converter.isJSONResponseTool(block.toolName) {
		return nil
	}

	var parts []model.StreamPart

	if block.toolsetName != "" {
		block.input = toolsetMemberDelta(block.memberName, block.input)
		parts = append(parts, model.ToolInputDelta{
			ID:    block.toolCallID,
			Delta: block.input,
		})
	}

	parts = append(parts, model.ToolInputEnd{
		ID: block.toolCallID,
	})

	input := block.input
	if input == "" {
		input = "{}"
	}

	if block.providerToolName == nameCodeExecution {
		input = programmaticInput(input)
	}

	return append(parts, model.ToolCall{
		ToolCallID:       block.toolCallID,
		ToolName:         block.toolName,
		Input:            input,
		ProviderExecuted: block.providerExecuted,
		Dynamic:          block.dynamic,
		ProviderMetadata: streamToolCallMetadata(block),
	})
}

// toolsetMemberDelta returns the buffered input of a toolset member call
// with the member name as its action. JSON that is not an object becomes an
// object holding only the action, as upstream does; unparseable input is
// kept as is.
func toolsetMemberDelta(memberName string, input string) string {
	if input == "" {
		input = "{}"
	}

	object, err := jsonobject.Parse([]byte(input))
	if errors.Is(err, jsonobject.ErrNotObject) {
		object = jsonobject.Object{}
	} else if err != nil {
		return input
	}

	withAction, err := object.WithFirst("action", jsonobject.EncodeString(memberName)).Encode()
	if err != nil {
		return input
	}

	return string(withAction)
}

func streamToolCallMetadata(block *streamBlock) model.ProviderMetadata {
	if block.caller == nil && block.toolsetName == "" {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: ToolCallMetadata{
			ToolsetName: block.toolsetName,
			Caller:      block.caller,
		},
	}
}
