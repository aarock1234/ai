package cartesia

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	// realtimeDefaultItemID identifies items of events without a request ID.
	realtimeDefaultItemID = "cartesia-realtime"

	// realtimeDefaultErrorMessage describes error events without a message.
	realtimeDefaultErrorMessage = "Unknown Cartesia realtime error"
)

// Realtime event types beyond those shared with streaming transcription.
const (
	realtimeConnected  = "connected"
	realtimeTurnStart  = "turn.start"
	realtimeTurnResume = "turn.resume"
)

// realtimeEvent is a Cartesia realtime event; fields beyond the type depend
// on it.
type realtimeEvent struct {
	Type       string   `json:"type"`
	RequestID  *string  `json:"request_id"`
	Transcript *string  `json:"transcript"`
	Text       *string  `json:"text"`
	IsFinal    *bool    `json:"is_final"`
	Duration   *float64 `json:"duration"`
	Message    *string  `json:"message"`
	ErrorCode  *string  `json:"error_code"`
}

// itemID returns the request ID, or a fixed ID when the event has none.
func (e realtimeEvent) itemID() string {
	if e.RequestID == nil {
		return realtimeDefaultItemID
	}

	return *e.RequestID
}

// requestID returns the request ID, or empty when the event has none.
func (e realtimeEvent) requestID() string {
	if e.RequestID == nil {
		return ""
	}

	return *e.RequestID
}

// manualTranscriptRaw is the raw message of a transcript assembled from the
// final transcript events of a session without turn detection.
type manualTranscriptRaw struct {
	Event            json.RawMessage   `json:"event"`
	TranscriptEvents []json.RawMessage `json:"transcriptEvents"`
	Duration         float64           `json:"duration"`
}

// realtimeEventParser maps the events of one connection. Without turn
// detection, it assembles the final transcript events until Cartesia
// flushes them.
type realtimeEventParser struct {
	mu sync.Mutex

	sessionCreated bool

	transcriptText     string
	transcriptDuration float64
	transcriptEvents   []json.RawMessage
}

func newRealtimeEventParser() *realtimeEventParser {
	return &realtimeEventParser{}
}

// parse maps one Cartesia event. The first event reports the session as
// created unless it is the connected event, which does so itself.
func (p *realtimeEventParser) parse(raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	var event realtimeEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		return nil, fmt.Errorf("%w: decode cartesia realtime event: %w", model.ErrInvalidResponse, err)
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if event.Type == realtimeConnected {
		p.sessionCreated = true

		return []model.RealtimeServerEvent{
			model.RealtimeSessionCreated{
				SessionID: event.requestID(),
				Raw:       raw,
			},
		}, nil
	}

	mapped, err := p.mapEvent(event, raw)
	if err != nil {
		return nil, err
	}

	return p.withSessionCreated(mapped, event, raw), nil
}

// mapEvent maps an event other than connected.
func (p *realtimeEventParser) mapEvent(event realtimeEvent, raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	switch event.Type {
	case realtimeTurnStart:
		return []model.RealtimeServerEvent{
			model.RealtimeSpeechStarted{
				ItemID: event.itemID(),
				Raw:    raw,
			},
		}, nil

	case realtimeTurnResume:
		return []model.RealtimeServerEvent{
			model.RealtimeSpeechStarted{
				ItemID: event.itemID(),
				Raw:    raw,
			},
			customRealtimeEvent(event, raw),
		}, nil

	case streamingTurnEnd:
		return turnEndEvents(event, raw), nil

	case streamingTranscript:
		p.recordTranscript(event, raw)

		return []model.RealtimeServerEvent{
			customRealtimeEvent(event, raw),
		}, nil

	case streamingFlushDone:
		return p.finishTranscript(event, raw)

	case streamingDone:
		return p.doneEvents(event, raw)

	case streamingError:
		return []model.RealtimeServerEvent{
			realtimeErrorEvent(event, raw),
		}, nil

	default:
		return []model.RealtimeServerEvent{
			customRealtimeEvent(event, raw),
		}, nil
	}
}

// turnEndEvents report the end of a detected turn with its transcript.
func turnEndEvents(event realtimeEvent, raw json.RawMessage) []model.RealtimeServerEvent {
	transcript := ""
	if event.Transcript != nil {
		transcript = *event.Transcript
	}

	return []model.RealtimeServerEvent{
		model.RealtimeSpeechStopped{
			ItemID: event.itemID(),
			Raw:    raw,
		},
		model.RealtimeInputTranscriptionCompleted{
			ItemID:     event.itemID(),
			Transcript: transcript,
			Raw:        raw,
		},
	}
}

// recordTranscript keeps a final transcript event for the next flush.
func (p *realtimeEventParser) recordTranscript(event realtimeEvent, raw json.RawMessage) {
	isFinal := event.IsFinal != nil && *event.IsFinal
	if !isFinal {
		return
	}

	if event.Text != nil {
		p.transcriptText += *event.Text
	}

	if event.Duration != nil {
		p.transcriptDuration += *event.Duration
	}

	p.transcriptEvents = append(p.transcriptEvents, raw)
}

// doneEvents report the end of the session, completing a pending
// transcript first.
func (p *realtimeEventParser) doneEvents(event realtimeEvent, raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	custom := customRealtimeEvent(event, raw)

	if p.transcriptText == "" {
		return []model.RealtimeServerEvent{
			custom,
		}, nil
	}

	completed, err := p.finishTranscript(event, raw)
	if err != nil {
		return nil, err
	}

	return append(completed, custom), nil
}

// finishTranscript completes the assembled transcript as a committed input
// buffer and resets it. Without transcript text, the event is reported as a
// custom event.
func (p *realtimeEventParser) finishTranscript(event realtimeEvent, raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	text := p.transcriptText
	transcriptRaw := manualTranscriptRaw{
		Event:            raw,
		TranscriptEvents: p.transcriptEvents,
		Duration:         p.transcriptDuration,
	}

	p.transcriptText = ""
	p.transcriptDuration = 0
	p.transcriptEvents = nil

	if text == "" {
		return []model.RealtimeServerEvent{
			customRealtimeEvent(event, raw),
		}, nil
	}

	encodedRaw, err := json.Marshal(transcriptRaw)
	if err != nil {
		return nil, fmt.Errorf("encode cartesia realtime transcript: %w", err)
	}

	return []model.RealtimeServerEvent{
		model.RealtimeAudioCommitted{
			ItemID: event.requestID(),
			Raw:    raw,
		},
		model.RealtimeInputTranscriptionCompleted{
			ItemID:     event.itemID(),
			Transcript: text,
			Raw:        encodedRaw,
		},
	}, nil
}

// withSessionCreated reports the session as created before the first
// event of a connection.
func (p *realtimeEventParser) withSessionCreated(events []model.RealtimeServerEvent, event realtimeEvent, raw json.RawMessage) []model.RealtimeServerEvent {
	if p.sessionCreated {
		return events
	}

	p.sessionCreated = true

	created := model.RealtimeSessionCreated{
		SessionID: event.itemID(),
		Raw:       raw,
	}

	return append([]model.RealtimeServerEvent{
		created,
	}, events...)
}

func realtimeErrorEvent(event realtimeEvent, raw json.RawMessage) model.RealtimeError {
	realtimeError := model.RealtimeError{
		Message: realtimeDefaultErrorMessage,
		Raw:     raw,
	}

	if event.Message != nil {
		realtimeError.Message = *event.Message
	}

	if event.ErrorCode != nil {
		realtimeError.Code = *event.ErrorCode
	}

	return realtimeError
}

func customRealtimeEvent(event realtimeEvent, raw json.RawMessage) model.RealtimeCustom {
	return model.RealtimeCustom{
		RawType: event.Type,
		Raw:     raw,
	}
}
