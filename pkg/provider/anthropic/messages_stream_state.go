package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// streamState is the state machine that converts stream events into stream
// parts. Each handler returns the parts its event produces.
type streamState struct {
	converter *contentConverter
	url       string
	headers   http.Header

	request *preparedRequest

	blocks map[int]*streamBlock

	// inThinkingBlock reports that a thinking block is being streamed;
	// signature deltas apply only to thinking blocks.
	inThinkingBlock bool

	jsonFromTool         bool
	hasWebSearchResults  bool
	fallbackWebCitations []Citation

	finishReason         model.FinishReason
	usage                usage
	rawUsage             jsonobject.Object
	encodedUsage         json.RawMessage
	stopSequence         *string
	stopDetails          *StopDetails
	inputTransformations []InputTransformation
	safeguardResults     []SafeguardResult
	container            *Container
	contextManagement    *AppliedContextManagement

	messageOpen     bool
	activeMessageID *string

	// invalidSequence stops processing after a message_start for a new
	// message arrives while another message is open.
	invalidSequence bool

	finished      bool
	reportedError bool
}

func newStreamState(call *preparedCall, headers http.Header) *streamState {
	return &streamState{
		converter: newContentConverter(call.request, call.documents, call.generateID),
		url:       call.api.URL,
		headers:   headers,
		request:   call.request,
		blocks:    make(map[int]*streamBlock),
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
	}
}

func (s *streamState) handle(event streamEvent) []model.StreamPart {
	switch typed := event.(type) {
	case *messageStartEvent:
		return s.messageStart(typed)
	case *contentBlockStartEvent:
		return s.blockStart(typed)
	case *contentBlockDeltaEvent:
		return s.blockDelta(typed)
	case *contentBlockStopEvent:
		return s.blockStop(typed)
	case *messageDeltaEvent:
		return s.messageDelta(typed)
	case *messageStopEvent:
		return s.messageStop()
	case *errorEvent:
		s.reportedError = true

		return []model.StreamPart{
			model.StreamError{
				Err: streamError(typed, s.url, s.headers, 0),
			},
		}
	default:
		return nil
	}
}

// messageStart opens a message. A repeated message_start for the open
// message is ignored; one for another message is an error that ends
// processing.
func (s *streamState) messageStart(event *messageStartEvent) []model.StreamPart {
	message := event.Message

	if s.messageOpen {
		if sameMessageID(s.activeMessageID, message.ID) {
			return nil
		}

		s.invalidSequence = true
		s.reportedError = true

		return []model.StreamPart{
			model.StreamError{
				Err: fmt.Errorf(
					"%w: received message_start for message %s while message %s is still open",
					model.ErrInvalidResponse,
					quotedID(message.ID),
					quotedID(s.activeMessageID),
				),
			},
		}
	}

	s.messageOpen = true
	s.activeMessageID = message.ID

	var started startUsage
	if err := json.Unmarshal(message.Usage, &started); err != nil {
		return streamFailure(fmt.Errorf("%w: message_start usage: %w", model.ErrInvalidResponse, err))
	}

	rawUsage, err := jsonobject.Parse(message.Usage)
	if err != nil {
		return streamFailure(fmt.Errorf("%w: message_start usage: %w", model.ErrInvalidResponse, err))
	}

	if err := s.setRawUsage(rawUsage); err != nil {
		return streamFailure(fmt.Errorf("%w: message_start usage: %w", model.ErrInvalidResponse, err))
	}

	s.usage.InputTokens = started.InputTokens
	s.usage.CacheReadInputTokens = new(valueOrZero(started.CacheReadInputTokens))
	s.usage.CacheCreationInputTokens = new(valueOrZero(started.CacheCreationInputTokens))

	if message.InputTransformations != nil {
		s.inputTransformations = message.InputTransformations
	}

	if message.Container != nil {
		s.container = &Container{
			ExpiresAt: message.Container.ExpiresAt,
			ID:        message.Container.ID,
		}
	}

	if message.StopReason != nil {
		s.finishReason = finishReason(message.StopReason, s.jsonFromTool)
	}

	parts := []model.StreamPart{
		model.ResponseMetadata{
			ID:      valueOrZero(message.ID),
			ModelID: valueOrZero(message.Model),
		},
		model.Custom{
			Kind: messageStartKind,
			ProviderMetadata: model.ProviderMetadata{
				Namespace: MessageStartMetadata{
					ID:    message.ID,
					Model: message.Model,
					Usage: s.encodedUsage,
				},
			},
		},
	}

	// Programmatic tool calling may deliver deferred tool calls complete in
	// message_start.
	for _, call := range message.Content {
		if call.Type != blockToolUse {
			parts = append(parts, model.StreamError{
				Err: fmt.Errorf("%w: message_start content has unexpected block type %q", model.ErrInvalidResponse, call.Type),
			})

			continue
		}

		parts = append(parts, completeToolCall(call.responseToolUse)...)
	}

	return parts
}

func sameMessageID(open *string, started *string) bool {
	if open == nil || started == nil {
		return open == started
	}

	return *open == *started
}

func quotedID(messageID *string) string {
	if messageID == nil {
		return "undefined"
	}

	return fmt.Sprintf("%q", *messageID)
}

// completeToolCall streams a tool call whose input is already complete.
func completeToolCall(call responseToolUse) []model.StreamPart {
	input := compactInput(call.Input)

	return []model.StreamPart{
		model.ToolInputStart{
			ID:       call.ID,
			ToolName: call.Name,
		},
		model.ToolInputDelta{
			ID:    call.ID,
			Delta: input,
		},
		model.ToolInputEnd{
			ID: call.ID,
		},
		model.ToolCall{
			ToolCallID:       call.ID,
			ToolName:         call.Name,
			Input:            input,
			ProviderMetadata: callerMetadata(call.Caller),
		},
	}
}

// messageDelta records the cumulative usage and the stop information of the
// open message.
func (s *streamState) messageDelta(event *messageDeltaEvent) []model.StreamPart {
	var reported deltaUsage
	if err := json.Unmarshal(event.Usage, &reported); err != nil {
		return streamFailure(fmt.Errorf("%w: message_delta usage: %w", model.ErrInvalidResponse, err))
	}

	rawUsage, err := jsonobject.Parse(event.Usage)
	if err != nil {
		return streamFailure(fmt.Errorf("%w: message_delta usage: %w", model.ErrInvalidResponse, err))
	}

	if err := s.setRawUsage(s.rawUsage.Merge(rawUsage)); err != nil {
		return streamFailure(fmt.Errorf("%w: message_delta usage: %w", model.ErrInvalidResponse, err))
	}

	s.applyDeltaUsage(reported)

	delta := event.Delta
	s.finishReason = finishReason(delta.StopReason, s.jsonFromTool)
	s.stopSequence = delta.StopSequence
	s.stopDetails = stopDetailsOf(delta.StopDetails)
	s.container = containerOf(delta.Container)

	if event.ContextManagement != nil {
		s.contextManagement = appliedContextManagementOf(event.ContextManagement)
	}

	if event.InputTransformations != nil {
		s.inputTransformations = event.InputTransformations
	}

	// Earlier deltas may carry no verdicts while the classifier still runs;
	// the last reported verdicts are final.
	if delta.SafeguardResults != nil {
		s.safeguardResults = delta.SafeguardResults
	}

	return nil
}

func (s *streamState) applyDeltaUsage(reported deltaUsage) {
	if reported.InputTokens != nil {
		s.usage.InputTokens = *reported.InputTokens
	}

	s.usage.OutputTokens = reported.OutputTokens

	if reported.OutputTokensDetails != nil {
		s.usage.OutputTokensDetails = reported.OutputTokensDetails
	}

	if reported.CacheReadInputTokens != nil {
		s.usage.CacheReadInputTokens = reported.CacheReadInputTokens
	}

	if reported.CacheCreationInputTokens != nil {
		s.usage.CacheCreationInputTokens = reported.CacheCreationInputTokens
	}

	if reported.Iterations != nil {
		s.usage.Iterations = reported.Iterations
	}
}

// messageStop closes the message and finishes it. Web search citations
// become sources here when no search results announced their own.
func (s *streamState) messageStop() []model.StreamPart {
	s.messageOpen = false
	s.activeMessageID = nil

	var parts []model.StreamPart

	if !s.hasWebSearchResults {
		for _, citation := range s.fallbackWebCitations {
			if source, ok := citationSource(citation, s.converter.documents, s.converter.generateID); ok {
				parts = append(parts, source)
			}
		}
	}

	s.fallbackWebCitations = nil
	s.finished = true

	return append(parts, model.Finish{
		Usage:            convertUsage(s.usage, s.encodedRawUsage()),
		FinishReason:     s.finishReason,
		ProviderMetadata: s.providerMetadata(),
	})
}

// endParts finishes a stream that ended without closing its last message,
// reporting the truncation unless an error was already reported.
func (s *streamState) endParts() []model.StreamPart {
	if s.finished && !s.messageOpen {
		return nil
	}

	var parts []model.StreamPart
	if !s.reportedError {
		parts = append(parts, model.StreamError{
			Err: fmt.Errorf("%w: stream ended before message_stop", model.ErrInvalidResponse),
		})
	}

	return append(parts, model.Finish{
		Usage: convertUsage(s.usage, s.encodedRawUsage()),
		FinishReason: model.FinishReason{
			Unified: model.FinishError,
			Raw:     s.finishReason.Raw,
		},
		ProviderMetadata: s.providerMetadata(),
	})
}

// setRawUsage records the cumulative raw usage and its encoding.
func (s *streamState) setRawUsage(rawUsage jsonobject.Object) error {
	encoded, err := rawUsage.Encode()
	if err != nil {
		return err
	}

	s.rawUsage = rawUsage
	s.encodedUsage = encoded

	return nil
}

func (s *streamState) encodedRawUsage() json.RawMessage {
	if s.rawUsage == nil {
		return nil
	}

	return s.encodedUsage
}

func (s *streamState) providerMetadata() model.ProviderMetadata {
	rawUsage := s.encodedRawUsage()
	if rawUsage == nil {
		rawUsage = json.RawMessage("null")
	}

	return s.request.providerMetadata(MessagesMetadata{
		Usage:                rawUsage,
		StopSequence:         s.stopSequence,
		StopDetails:          s.stopDetails,
		InputTransformations: s.inputTransformations,
		SafeguardResults:     s.safeguardResults,
		Iterations:           convertIterations(s.usage.Iterations),
		Container:            s.container,
		ContextManagement:    s.contextManagement,
	})
}

func streamFailure(err error) []model.StreamPart {
	return []model.StreamPart{
		model.StreamError{
			Err: err,
		},
	}
}
