package google

import (
	"encoding/json"
	"fmt"
	"strconv"
	"sync"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// inputTranscriptionBoundary records what ended the latest response, which
// decides whether the next input transcription starts a new utterance.
type inputTranscriptionBoundary int

const (
	boundaryNone inputTranscriptionBoundary = iota
	boundaryInterrupted
	boundaryTurnComplete
)

// realtimeServerMessage is one Live API server message. Exactly one of the
// fields is normally set.
type realtimeServerMessage struct {
	SetupComplete           json.RawMessage        `json:"setupComplete"`
	ToolCall                *realtimeToolCall      `json:"toolCall"`
	ToolCallCancellation    json.RawMessage        `json:"toolCallCancellation"`
	ServerContent           *realtimeServerContent `json:"serverContent"`
	InputTranscription      *realtimeTranscription `json:"inputTranscription"`
	GoAway                  json.RawMessage        `json:"goAway"`
	SessionResumptionUpdate json.RawMessage        `json:"sessionResumptionUpdate"`
}

type realtimeToolCall struct {
	FunctionCalls []realtimeFunctionCall `json:"functionCalls"`
}

type realtimeFunctionCall struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type realtimeServerContent struct {
	GenerationComplete  bool                   `json:"generationComplete"`
	InteractionStatus   json.RawMessage        `json:"interactionStatus"`
	Interrupted         bool                   `json:"interrupted"`
	ModelTurn           *realtimeModelTurn     `json:"modelTurn"`
	OutputTranscription *realtimeTranscription `json:"outputTranscription"`
	InputTranscription  *realtimeTranscription `json:"inputTranscription"`
	TurnComplete        bool                   `json:"turnComplete"`
	WaitingForInput     bool                   `json:"waitingForInput"`
}

type realtimeModelTurn struct {
	Parts []realtimeModelPart `json:"parts"`
}

type realtimeModelPart struct {
	InlineData *struct {
		Data string `json:"data"`
	} `json:"inlineData"`

	Text string `json:"text"`
}

type realtimeTranscription struct {
	Text     string `json:"text"`
	Finished *bool  `json:"finished"`
}

// realtimeEventMapper maps the messages of one connection. Live API
// messages carry no response or item IDs, so the mapper tracks turns to
// synthesize stable ones. It is safe for concurrent use.
type realtimeEventMapper struct {
	mu sync.Mutex

	turnCounter int
	hasAudio    bool
	hasText     bool

	hasTranscript bool
	turnClosed    bool

	inputTranscriptionCounter  int
	inputTranscriptionBuffer   string
	inputTranscriptionBoundary inputTranscriptionBoundary

	// preserveInputAcrossTurnComplete keeps the utterance of a barge-in
	// open across the interrupted response's trailing turn completion.
	preserveInputAcrossTurnComplete bool
}

func newRealtimeEventMapper() *realtimeEventMapper {
	return &realtimeEventMapper{}
}

func (m *realtimeEventMapper) responseID() string {
	return "google-resp-" + strconv.Itoa(m.turnCounter)
}

func (m *realtimeEventMapper) itemID() string {
	return "google-item-" + strconv.Itoa(m.turnCounter)
}

// beginTurnIfClosed rolls over to the next turn once its model content
// arrives. A turn completion only closes the turn, so that an output
// transcript arriving after it stays with its turn.
func (m *realtimeEventMapper) beginTurnIfClosed() {
	if !m.turnClosed {
		return
	}

	m.turnCounter++
	m.hasAudio = false
	m.hasText = false
	m.hasTranscript = false
	m.turnClosed = false
	m.preserveInputAcrossTurnComplete = false
}

// parse maps one server message to one or more events.
func (m *realtimeEventMapper) parse(raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	var message realtimeServerMessage
	if err := json.Unmarshal(raw, &message); err != nil {
		return nil, fmt.Errorf("%w: decode realtime server message: %w", model.ErrInvalidResponse, err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	switch {
	case wirejson.HasValue(message.SetupComplete):
		return realtimeEvents(model.RealtimeSessionCreated{
			Raw: raw,
		}), nil

	case message.ToolCall != nil:
		return m.toolCall(*message.ToolCall, raw), nil

	case wirejson.HasValue(message.ToolCallCancellation):
		return realtimeCustom("toolCallCancellation", raw), nil

	case wirejson.HasValue(message.GoAway):
		return realtimeCustom("goAway", raw), nil

	case wirejson.HasValue(message.SessionResumptionUpdate):
		return realtimeCustom("sessionResumptionUpdate", raw), nil

	case message.ServerContent != nil:
		return m.serverContent(*message.ServerContent, raw), nil
	}

	if message.InputTranscription != nil {
		if event, ok := m.inputTranscription(*message.InputTranscription, raw); ok {
			return realtimeEvents(event), nil
		}
	}

	return realtimeCustom(firstMemberKey(raw), raw), nil
}

func (m *realtimeEventMapper) toolCall(call realtimeToolCall, raw json.RawMessage) []model.RealtimeServerEvent {
	m.beginTurnIfClosed()

	events := make([]model.RealtimeServerEvent, 0, 2*len(call.FunctionCalls))

	for _, functionCall := range call.FunctionCalls {
		arguments := compactArguments(functionCall.Args)

		events = append(events,
			model.RealtimeFunctionCallArgumentsDelta{
				ResponseID: m.responseID(),
				ItemID:     m.itemID(),
				CallID:     functionCall.ID,
				Delta:      arguments,
				Raw:        raw,
			},
			model.RealtimeFunctionCallArgumentsDone{
				ResponseID: m.responseID(),
				ItemID:     m.itemID(),
				CallID:     functionCall.ID,
				Name:       functionCall.Name,
				Arguments:  arguments,
				Raw:        raw,
			},
		)
	}

	return events
}

func (m *realtimeEventMapper) serverContent(content realtimeServerContent, raw json.RawMessage) []model.RealtimeServerEvent {
	var events []model.RealtimeServerEvent

	if content.Interrupted {
		// A barge-in ends the current utterance; the interrupted response's
		// trailing turn completion must not split the next one.
		m.inputTranscriptionBoundary = boundaryInterrupted
		m.preserveInputAcrossTurnComplete = true
		m.turnClosed = true

		events = append(events, model.RealtimeSpeechStarted{
			Raw: raw,
		})
	}

	if content.ModelTurn != nil && content.ModelTurn.Parts != nil {
		m.beginTurnIfClosed()
		events = append(events, m.modelTurn(content.ModelTurn.Parts, raw)...)
	}

	if content.OutputTranscription != nil && content.OutputTranscription.Text != "" {
		m.hasTranscript = true

		events = append(events, model.RealtimeAudioTranscriptDelta{
			ResponseID: m.responseID(),
			ItemID:     m.itemID(),
			Delta:      content.OutputTranscription.Text,
			Raw:        raw,
		})
	}

	if content.InputTranscription != nil {
		if event, ok := m.inputTranscription(*content.InputTranscription, raw); ok {
			events = append(events, event)
		}
	}

	// Generation can stop while playback and the turn stay open, so it is
	// reported apart from the response end.
	if content.GenerationComplete {
		events = append(events, realtimeCustom("generationComplete", raw)...)
	}

	// The interaction status is the definitive activity signal of
	// background reasoning models.
	if wirejson.HasValue(content.InteractionStatus) {
		events = append(events, realtimeCustom("interactionStatus", raw)...)
	}

	// Proactive audio models yield the floor while they wait for input.
	if content.WaitingForInput {
		events = append(events, realtimeCustom("waitingForInput", raw)...)
	}

	if content.TurnComplete {
		events = append(events, m.completeTurn(raw)...)
	}

	if len(events) == 0 {
		return realtimeCustom("serverContent", raw)
	}

	return events
}

func (m *realtimeEventMapper) modelTurn(parts []realtimeModelPart, raw json.RawMessage) []model.RealtimeServerEvent {
	var events []model.RealtimeServerEvent

	for _, part := range parts {
		if part.InlineData != nil && part.InlineData.Data != "" {
			m.hasAudio = true

			events = append(events, model.RealtimeAudioDelta{
				ResponseID: m.responseID(),
				ItemID:     m.itemID(),
				Delta:      part.InlineData.Data,
				Raw:        raw,
			})
		}

		if part.Text != "" {
			m.hasText = true

			events = append(events, model.RealtimeTextDelta{
				ResponseID: m.responseID(),
				ItemID:     m.itemID(),
				Delta:      part.Text,
				Raw:        raw,
			})
		}
	}

	return events
}

// completeTurn ends the outputs of the turn and the response, and closes
// the turn. The counter advances when the next response begins.
func (m *realtimeEventMapper) completeTurn(raw json.RawMessage) []model.RealtimeServerEvent {
	var events []model.RealtimeServerEvent

	if m.hasAudio {
		events = append(events, model.RealtimeAudioDone{
			ResponseID: m.responseID(),
			ItemID:     m.itemID(),
			Raw:        raw,
		})
	}

	if m.hasText {
		events = append(events, model.RealtimeTextDone{
			ResponseID: m.responseID(),
			ItemID:     m.itemID(),
			Raw:        raw,
		})
	}

	if m.hasTranscript {
		events = append(events, model.RealtimeAudioTranscriptDone{
			ResponseID: m.responseID(),
			ItemID:     m.itemID(),
			Raw:        raw,
		})
	}

	events = append(events, model.RealtimeResponseDone{
		ResponseID: m.responseID(),
		Status:     "completed",
		Raw:        raw,
	})

	m.turnClosed = true

	switch {
	case m.preserveInputAcrossTurnComplete:
		m.preserveInputAcrossTurnComplete = false

	case m.inputTranscriptionBoundary != boundaryInterrupted:
		// Without a finished signal, a completed response is the fallback
		// boundary of the next utterance.
		m.inputTranscriptionBoundary = boundaryTurnComplete
	}

	return events
}

// inputTranscription accumulates the fragments of one user utterance into
// a running transcript with a stable synthetic ID. The finished signal is
// the authoritative utterance boundary; response boundaries are the
// fallback for messages without it.
func (m *realtimeEventMapper) inputTranscription(transcription realtimeTranscription, raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	finished := transcription.Finished != nil && *transcription.Finished

	if transcription.Text == "" {
		if finished {
			m.finishInputTranscription()
		}

		return nil, false
	}

	hasFinishedSignal := transcription.Finished != nil
	startsUtterance := m.inputTranscriptionBoundary == boundaryInterrupted ||
		(m.inputTranscriptionBoundary == boundaryTurnComplete && !hasFinishedSignal)

	if m.inputTranscriptionBuffer != "" && startsUtterance {
		m.finishInputTranscription()
	}

	m.inputTranscriptionBoundary = boundaryNone
	m.inputTranscriptionBuffer += transcription.Text

	event := model.RealtimeInputTranscriptionCompleted{
		ItemID:     "google-input-" + strconv.Itoa(m.inputTranscriptionCounter),
		Transcript: m.inputTranscriptionBuffer,
		Raw:        raw,
	}

	if finished {
		m.finishInputTranscription()
	}

	return event, true
}

func (m *realtimeEventMapper) finishInputTranscription() {
	if m.inputTranscriptionBuffer == "" {
		return
	}

	m.inputTranscriptionCounter++
	m.inputTranscriptionBuffer = ""
}

func realtimeEvents(events ...model.RealtimeServerEvent) []model.RealtimeServerEvent {
	return events
}

func realtimeCustom(rawType string, raw json.RawMessage) []model.RealtimeServerEvent {
	return realtimeEvents(model.RealtimeCustom{
		RawType: rawType,
		Raw:     raw,
	})
}

// firstMemberKey returns the first key of a message, or "undefined" for an
// empty message, as upstream names unknown events.
func firstMemberKey(raw json.RawMessage) string {
	object, err := jsonobject.Parse(raw)
	if err != nil || len(object) == 0 {
		return "undefined"
	}

	return object[0].Key
}
