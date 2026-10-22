package openai

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// responsesOutputMapper converts the output items of a response into
// content.
type responsesOutputMapper struct {
	context responsesCallContext

	content  []model.Content
	logprobs [][]ResponsesLogprob

	// hasFunctionCall reports client-executed tool calls, which turn the
	// finish reason into a tool call finish.
	hasFunctionCall bool

	hostedToolSearchCallIDs []string
	retrievedFileIDs        map[string]bool

	// Citation sources are added only when the search items do not already
	// provide them.
	hasWebSearchActionSources bool
	hasFileSearchResults      bool
}

func newResponsesOutputMapper(callContext responsesCallContext) *responsesOutputMapper {
	return &responsesOutputMapper{
		context:          callContext,
		retrievedFileIDs: make(map[string]bool),
	}
}

// mapOutput converts every output item in order.
func (m *responsesOutputMapper) mapOutput(output responsesOutput) error {
	for _, item := range output {
		switch item := item.(type) {
		case responsesWebSearchCallItem:
			if responsesHasURLSources(item) {
				m.hasWebSearchActionSources = true
			}
		case responsesFileSearchCallItem:
			if len(item.Results) > 0 {
				m.hasFileSearchResults = true
			}
		}
	}

	for _, item := range output {
		if err := m.item(item); err != nil {
			return err
		}
	}

	return nil
}

func (m *responsesOutputMapper) item(item responsesOutputItem) error {
	c := m.context

	switch item := item.(type) {
	case responsesReasoningOutput:
		m.reasoning(item)

	case responsesImageGenerationCallItem:
		m.add(
			c.imageGenerationCall(item.ID),
			c.imageGenerationResult(item.ID, item.Result, false),
		)

	case responsesToolSearchCallItem:
		m.toolSearchCall(item)

	case responsesToolSearchOutputItem:
		m.toolSearchOutput(item)

	case responsesLocalShellCallItem:
		m.add(c.localShellCall(item))

	case responsesShellCallItem:
		m.add(c.shellCall(item))

	case responsesShellCallOutputItem:
		m.add(c.shellResult(item))

	case responsesMessageItem:
		return m.message(item)

	case responsesFunctionCallItem:
		m.functionCall(item)

	case responsesProgramItem:
		m.add(c.programCall(item))

	case responsesProgramOutputItem:
		m.add(c.programResult(item))

	case responsesCustomToolCallItem:
		m.hasFunctionCall = true
		m.add(c.customToolCall(item, item.Async))

	case responsesWebSearchCallItem:
		m.add(c.webSearchCall(item.ID), c.webSearchResult(item))
		for _, source := range c.webSearchSources(item) {
			m.add(source)
		}

	case responsesMCPCallItem:
		toolCallID := c.mcpToolCallID(item, nil)
		m.add(
			c.mcpCall(toolCallID, item.Name, item.Arguments),
			c.mcpResult(item, toolCallID),
		)

	case responsesMCPApprovalRequestItem:
		call, request := c.mcpApproval(item)
		m.add(call, request)

	case responsesComputerCallItem:
		m.computerCall(item)

	case responsesFileSearchCallItem:
		m.add(c.fileSearchCall(item.ID), c.fileSearchResult(item))
		for _, source := range c.fileSearchSources(item, m.retrievedFileIDs) {
			m.add(source)
		}

	case responsesCodeInterpreterCallItem:
		m.add(
			c.codeInterpreterCall(item.ID, item.Code, item.ContainerID),
			c.codeInterpreterResult(item),
		)

	case responsesApplyPatchCallItem:
		m.hasFunctionCall = true
		m.add(c.applyPatchCall(item, item.CallID))

	case responsesCompactionItem:
		m.add(c.compaction(item))
	}

	return nil
}

// reasoning adds one reasoning part per summary, or an empty part when the
// item has no summary.
func (m *responsesOutputMapper) reasoning(item responsesReasoningOutput) {
	summaries := item.Summary
	if len(summaries) == 0 {
		summaries = []responsesSummaryText{
			{
				Type: "summary_text",
			},
		}
	}

	for _, summary := range summaries {
		m.add(model.Reasoning{
			Text:             summary.Text,
			ProviderMetadata: m.context.reasoningMetadata(item.ID, item.EncryptedContent),
		})
	}
}

// message adds the text parts of a message with their citations and
// sources.
func (m *responsesOutputMapper) message(item responsesMessageItem) error {
	c := m.context

	for _, part := range item.Content {
		if part.Type != "output_text" {
			return fmt.Errorf("%w: unsupported message content type %q", model.ErrInvalidResponse, part.Type)
		}

		if c.collectLogprobs && part.Logprobs != nil {
			m.logprobs = append(m.logprobs, part.Logprobs)
		}

		m.add(model.Text{
			Text:             part.Text,
			Citations:        c.citations(part.Annotations),
			ProviderMetadata: c.textMetadata(item.ID, item.Phase, part.Annotations),
		})

		for _, annotation := range part.Annotations {
			if m.retrievalProvides(annotation) {
				continue
			}

			m.add(c.annotationSource(annotation, c.generateID()))
		}
	}

	return nil
}

// retrievalProvides reports whether search items already provide the
// source of a citation.
func (m *responsesOutputMapper) retrievalProvides(annotation ResponsesAnnotation) bool {
	switch annotation.Type {
	case ResponsesURLCitation:
		return m.hasWebSearchActionSources
	case ResponsesFileCitation:
		return m.hasFileSearchResults
	default:
		return false
	}
}

// functionCall adds a function call, expanding parallel wrapper calls.
func (m *responsesOutputMapper) functionCall(item responsesFunctionCallItem) {
	m.hasFunctionCall = true

	if calls, ok := m.context.expandParallelCall(item); ok {
		for _, call := range calls {
			m.add(call)
		}

		return
	}

	m.add(m.context.functionCall(item, item.Async))
}

// computerCall adds a computer call for the application, or a legacy
// computer use call and result.
func (m *responsesOutputMapper) computerCall(item responsesComputerCallItem) {
	c := m.context

	if item.CallID == nil {
		m.add(c.computerUseCall(item.ID), c.computerUseResult(item))

		return
	}

	m.hasFunctionCall = true
	m.add(c.computerCall(item, *item.CallID))
}

// toolSearchCall adds a tool search call. Hosted searches are remembered so
// that their outputs, which lack a call ID, can be paired with them.
func (m *responsesOutputMapper) toolSearchCall(item responsesToolSearchCallItem) {
	toolCallID := item.ID
	if item.CallID != nil {
		toolCallID = *item.CallID
	}

	if item.Execution == "server" {
		m.hostedToolSearchCallIDs = append(m.hostedToolSearchCallIDs, toolCallID)
	}

	m.add(m.context.toolSearchCall(item, toolCallID, item.CallID))
}

func (m *responsesOutputMapper) toolSearchOutput(item responsesToolSearchOutputItem) {
	toolCallID := responsesToolSearchOutputCallID(item, &m.hostedToolSearchCallIDs)

	m.add(m.context.toolSearchResult(item, toolCallID))
}

func (m *responsesOutputMapper) add(content ...model.Content) {
	m.content = append(m.content, content...)
}

// responsesToolSearchOutputCallID pairs a tool search output with its call:
// by call ID, else with the oldest unpaired hosted call, else by item ID.
func responsesToolSearchOutputCallID(item responsesToolSearchOutputItem, hostedCallIDs *[]string) string {
	if item.CallID != nil {
		return *item.CallID
	}

	if len(*hostedCallIDs) > 0 {
		toolCallID := (*hostedCallIDs)[0]
		*hostedCallIDs = (*hostedCallIDs)[1:]

		return toolCallID
	}

	return item.ID
}
