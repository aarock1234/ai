package perplexity

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// Stream event types.
const (
	eventCreated          = "response.created"
	eventInProgress       = "response.in_progress"
	eventTextDelta        = "response.output_text.delta"
	eventTextDone         = "response.output_text.done"
	eventReasoningStarted = "response.reasoning.started"
	eventSearchQueries    = "response.reasoning.search_queries"
	eventFetchURLQueries  = "response.reasoning.fetch_url_queries"
	eventSearchResults    = "response.reasoning.search_results"
	eventFetchURLResults  = "response.reasoning.fetch_url_results"
	eventReasoningStopped = "response.reasoning.stopped"
	eventOutputItemDone   = "response.output_item.done"
	eventCompleted        = "response.completed"
	eventIncomplete       = "response.incomplete"
	eventFailed           = "response.failed"
)

// Stream block IDs and the raw finish reason of failed responses.
const (
	defaultTextBlockID      = "text"
	reasoningBlockPrefix    = "reasoning-"
	failedFinishReasonValue = "failed"
)

// errResponseFailed reports a failed stream without an error object.
var errResponseFailed = errors.New("perplexity response failed")

// agentStreamSettings configures one stream.
type agentStreamSettings struct {
	includeRawChunks bool
	generateID       func() string
}

// textState is the progress of one streamed text block.
type textState struct {
	text  string
	ended bool
}

// agentStream converts Agent API events into stream parts. Sources without
// a search result ID are held back, because a later search result can
// supply the citation ID.
type agentStream struct {
	settings agentStreamSettings

	finishReason    model.FinishReason
	usage           *agentUsage
	hasFunctionCall bool
	sentMetadata    bool
	activeReasoning string

	texts     map[string]*textState
	textOrder []string

	emittedSources map[string]bool
	pending        map[string]model.Source
	pendingOrder   []string
	seenCalls      map[string]bool
}

func newAgentStream(settings agentStreamSettings) *agentStream {
	return &agentStream{
		settings: settings,
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
		texts:          make(map[string]*textState),
		emittedSources: make(map[string]bool),
		pending:        make(map[string]model.Source),
		seenCalls:      make(map[string]bool),
	}
}

// Process converts one server-sent event into stream parts.
func (s *agentStream) Process(event sse.Event) ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if s.settings.includeRawChunks {
		if raw, ok := chatsse.Raw(event); ok {
			parts = append(parts, raw)
		}
	}

	var chunk agentChunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		decodeErr := fmt.Errorf("%w: decode agent stream event: %w", model.ErrInvalidResponse, err)

		return append(parts, s.Fail(decodeErr)), nil
	}

	return append(parts, s.chunkParts(chunk)...), nil
}

func (s *agentStream) chunkParts(chunk agentChunk) []model.StreamPart {
	switch chunk.Type {
	case eventCreated, eventInProgress:
		return s.metadataParts(chunk.Response)

	case eventTextDelta:
		if chunk.Delta == nil {
			return nil
		}

		return s.textDelta(textID(chunk.ItemID, chunk.OutputIndex, chunk.ContentIndex), *chunk.Delta)

	case eventTextDone:
		return s.finishText(textID(chunk.ItemID, chunk.OutputIndex, chunk.ContentIndex), chunk.Text)

	case eventReasoningStarted:
		return s.startReasoning(chunk)

	case eventSearchQueries, eventFetchURLQueries:
		return s.reasoningThought(chunk.Thought)

	case eventSearchResults:
		parts := s.reasoningThought(chunk.Thought)
		for _, result := range chunk.Results {
			parts = append(parts, s.emitSource(searchSource(result, s.settings.generateID))...)
		}

		return parts

	case eventFetchURLResults:
		parts := s.reasoningThought(chunk.Thought)
		for _, fetched := range chunk.Contents {
			parts = append(parts, s.emitSource(fetchedSource(fetched, s.settings.generateID))...)
		}

		return parts

	case eventReasoningStopped:
		parts := s.reasoningThought(chunk.Thought)

		return append(parts, s.endReasoning()...)

	case eventOutputItemDone:
		if chunk.Item == nil {
			return nil
		}

		return s.itemParts(*chunk.Item, chunk.OutputIndex)

	case eventCompleted, eventIncomplete:
		return s.complete(chunk.Response)

	case eventFailed:
		s.finishReason = model.FinishReason{
			Unified: model.FinishError,
			Raw:     failedFinishReasonValue,
		}

		return []model.StreamPart{
			model.StreamError{
				Err: failedStreamError(chunk.Error),
			},
		}

	default:
		return nil
	}
}

// failedStreamError returns the error of a failed response, or a generic
// failure when the event has none.
func failedStreamError(responseErr *responseError) error {
	if responseErr == nil {
		return errResponseFailed
	}

	return responseErr
}

// metadataParts emits the response metadata once.
func (s *agentStream) metadataParts(response *agentResponse) []model.StreamPart {
	if s.sentMetadata || response == nil {
		return nil
	}

	s.sentMetadata = true

	return []model.StreamPart{
		responseMetadata(response),
	}
}

// complete finishes the output of the final response and records its usage
// and finish reason.
func (s *agentStream) complete(response *agentResponse) []model.StreamPart {
	if response == nil {
		return nil
	}

	parts := s.metadataParts(response)

	for index, item := range response.Output {
		parts = append(parts, s.itemParts(item, new(index))...)
	}

	s.usage = response.Usage
	s.finishReason = finishReason(response, s.hasFunctionCall)

	return parts
}

// itemParts finishes the item's text and emits its sources and function
// call.
func (s *agentStream) itemParts(item outputItem, outputIndex *int) []model.StreamPart {
	var parts []model.StreamPart

	if item.Message != nil {
		for contentIndex, part := range item.Message.Content {
			if part.Type != "output_text" {
				continue
			}

			textBlockID := textID(item.Message.ID, outputIndex, new(contentIndex))
			parts = append(parts, s.finishText(textBlockID, part.Text)...)
		}

		for _, part := range item.Message.Content {
			for _, source := range annotationSources(part.Annotations, s.settings.generateID) {
				parts = append(parts, s.emitSource(source)...)
			}
		}
	}

	if item.SearchResult != nil {
		for _, result := range item.SearchResult.Results {
			parts = append(parts, s.emitSource(searchSource(result, s.settings.generateID))...)
		}
	}

	if item.FetchResults != nil {
		for _, fetched := range item.FetchResults.Contents {
			parts = append(parts, s.emitSource(fetchedSource(fetched, s.settings.generateID))...)
		}
	}

	return append(parts, s.functionCallParts(item.FunctionCall)...)
}

// textID names a text block after its item, or its output index, and its
// content index.
func textID(itemID *string, outputIndex *int, contentIndex *int) string {
	textBlockID := defaultTextBlockID

	switch {
	case itemID != nil:
		textBlockID = *itemID

	case outputIndex != nil:
		textBlockID = strconv.Itoa(*outputIndex)
	}

	if contentIndex == nil || *contentIndex == 0 {
		return textBlockID
	}

	return textBlockID + ":" + strconv.Itoa(*contentIndex)
}

func (s *agentStream) textDelta(textBlockID string, delta string) []model.StreamPart {
	state, ok := s.texts[textBlockID]
	if ok && state.ended {
		return nil
	}

	var parts []model.StreamPart
	if !ok {
		state = &textState{}
		s.texts[textBlockID] = state
		s.textOrder = append(s.textOrder, textBlockID)
		parts = append(parts, model.TextStart{
			ID: textBlockID,
		})
	}

	state.text += delta

	return append(parts, model.TextDelta{
		ID:    textBlockID,
		Delta: delta,
	})
}

// finishText emits terminal text that arrived without deltas, or the rest
// of a partially streamed part, and ends the block.
func (s *agentStream) finishText(textBlockID string, text *string) []model.StreamPart {
	state, ok := s.texts[textBlockID]
	if ok && state.ended {
		return nil
	}

	emitted := ""
	if ok {
		emitted = state.text
	}

	var parts []model.StreamPart

	hasRemainder := text != nil && strings.HasPrefix(*text, emitted) && len(*text) > len(emitted)
	if hasRemainder {
		parts = s.textDelta(textBlockID, (*text)[len(emitted):])
	}

	finalState, started := s.texts[textBlockID]
	if !started {
		return parts
	}

	finalState.ended = true

	return append(parts, model.TextEnd{
		ID: textBlockID,
	})
}

// startReasoning ends an open reasoning block and opens a new one named
// after the event's sequence number.
func (s *agentStream) startReasoning(chunk agentChunk) []model.StreamPart {
	parts := s.endReasoning()

	suffix := s.settings.generateID()
	if chunk.SequenceNumber != nil {
		suffix = strconv.FormatFloat(*chunk.SequenceNumber, 'f', -1, 64)
	}

	s.activeReasoning = reasoningBlockPrefix + suffix
	parts = append(parts, model.ReasoningStart{
		ID: s.activeReasoning,
	})

	return append(parts, s.reasoningThought(chunk.Thought)...)
}

func (s *agentStream) reasoningThought(thought *string) []model.StreamPart {
	if s.activeReasoning == "" || thought == nil {
		return nil
	}

	return []model.StreamPart{
		model.ReasoningDelta{
			ID:    s.activeReasoning,
			Delta: *thought,
		},
	}
}

func (s *agentStream) endReasoning() []model.StreamPart {
	if s.activeReasoning == "" {
		return nil
	}

	reasoningID := s.activeReasoning
	s.activeReasoning = ""

	return []model.StreamPart{
		model.ReasoningEnd{
			ID: reasoningID,
		},
	}
}

// emitSource emits a source with a search result ID once per URL and holds
// back other sources until the stream ends.
func (s *agentStream) emitSource(cited citedSource) []model.StreamPart {
	url := cited.source.URL
	if s.emittedSources[url] {
		return nil
	}

	if cited.hasResultID {
		delete(s.pending, url)
		s.emittedSources[url] = true

		return []model.StreamPart{
			cited.source,
		}
	}

	if _, ok := s.pending[url]; !ok {
		s.pending[url] = cited.source
		s.pendingOrder = append(s.pendingOrder, url)
	}

	return nil
}

// functionCallParts emits a complete function call once.
func (s *agentStream) functionCallParts(call *functionCallItem) []model.StreamPart {
	if !call.complete() || s.seenCalls[*call.CallID] {
		return nil
	}

	s.seenCalls[*call.CallID] = true
	s.hasFunctionCall = true

	return []model.StreamPart{
		model.ToolInputStart{
			ID:       *call.CallID,
			ToolName: *call.Name,
		},
		model.ToolInputDelta{
			ID:    *call.CallID,
			Delta: *call.Arguments,
		},
		model.ToolInputEnd{
			ID: *call.CallID,
		},
		functionToolCall(*call),
	}
}

// Fail records an error finish reason and returns the error part.
func (s *agentStream) Fail(err error) model.StreamPart {
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}

	return model.StreamError{
		Err: err,
	}
}

// Flush emits the held-back sources, closes the open blocks and reports the
// finish.
func (s *agentStream) Flush() ([]model.StreamPart, error) {
	var parts []model.StreamPart

	for _, url := range s.pendingOrder {
		if source, ok := s.pending[url]; ok {
			parts = append(parts, source)
		}
	}

	parts = append(parts, s.endReasoning()...)

	for _, textBlockID := range s.textOrder {
		if !s.texts[textBlockID].ended {
			parts = append(parts, model.TextEnd{
				ID: textBlockID,
			})
		}
	}

	parts = append(parts, model.Finish{
		Usage:            convertUsage(s.usage),
		FinishReason:     s.finishReason,
		ProviderMetadata: agentMetadata(s.usage),
	})

	return parts, nil
}
