package cartesia

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// streamingDefaultErrorMessage describes error events without a message.
const streamingDefaultErrorMessage = "Cartesia streaming transcription error"

// Streaming event types.
const (
	streamingTurnUpdate   = "turn.update"
	streamingTurnEagerEnd = "turn.eager_end"
	streamingTurnEnd      = "turn.end"
	streamingTranscript   = "transcript"
	streamingFlushDone    = "flush_done"
	streamingDone         = "done"
	streamingError        = "error"
)

// Client commands. With turn detection, the end of the audio closes the
// session; without it, the audio is finalized first and the session closes
// once the flush is done.
const (
	streamingFinalizeCommand = "finalize"
	streamingCloseCommand    = "close"
)

// streamingCloseMessage ends the audio of a turn detection session.
type streamingCloseMessage struct {
	Type string `json:"type"`
}

// streamingEvent is a server event; fields beyond the type depend on it.
type streamingEvent struct {
	Type       string   `json:"type"`
	RequestID  string   `json:"request_id"`
	Transcript string   `json:"transcript"`
	Text       string   `json:"text"`
	IsFinal    *bool    `json:"is_final"`
	Duration   *float64 `json:"duration"`
	Message    *string  `json:"message"`
	ErrorCode  *string  `json:"error_code"`
}

// streamingProtocol is the Cartesia streaming transcription protocol. Raw
// audio rides BINARY frames from the moment the connection opens.
type streamingProtocol struct {
	warnings      []model.Warning
	turnDetection bool

	finalTexts []string
	duration   float64
}

// Open starts the stream and the audio.
func (p *streamingProtocol) Open(session *livetranscription.TranscriptionSession) {
	session.Emit(model.StreamStart{
		Warnings: p.warnings,
	})
	session.StartAudio()
}

// HandleMessage handles one server event.
func (p *streamingProtocol) HandleMessage(session *livetranscription.TranscriptionSession, message []byte) {
	var event streamingEvent
	if err := json.Unmarshal(message, &event); err != nil {
		session.Fail(fmt.Errorf("%w: decode cartesia streaming event: %w", model.ErrInvalidResponse, err))

		return
	}

	switch event.Type {
	case streamingTurnUpdate, streamingTurnEagerEnd:
		session.Emit(model.TranscriptPartial{
			ID:   event.RequestID,
			Text: event.Transcript,
		})

	case streamingTurnEnd:
		p.finalTexts = append(p.finalTexts, event.Transcript)

		session.Emit(model.TranscriptFinal{
			ID:   event.RequestID,
			Text: event.Transcript,
		})

	case streamingTranscript:
		p.handleTranscript(session, event)

	case streamingFlushDone:
		session.Send(livetranscription.TextFrame([]byte(streamingCloseCommand)))

	case streamingDone:
		p.finish(session)

	case streamingError:
		session.Fail(&model.APICallError{
			Message:      streamingErrorMessage(event),
			ResponseBody: string(message),
		})
	}
}

// handleTranscript emits a transcript of a session without turn
// detection. Final transcripts add their duration to the total.
func (p *streamingProtocol) handleTranscript(session *livetranscription.TranscriptionSession, event streamingEvent) {
	if event.IsFinal != nil && *event.IsFinal {
		p.finalTexts = append(p.finalTexts, event.Text)

		if event.Duration != nil {
			p.duration += *event.Duration
		}

		session.Emit(model.TranscriptFinal{
			ID:   event.RequestID,
			Text: event.Text,
		})

		return
	}

	partial := model.TranscriptPartial{
		ID:   event.RequestID,
		Text: event.Text,
	}

	if event.Duration != nil {
		partial.Duration = new(seconds(*event.Duration))
	}

	session.Emit(partial)
}

func streamingErrorMessage(event streamingEvent) string {
	if event.Message != nil {
		return *event.Message
	}

	if event.ErrorCode != nil {
		return *event.ErrorCode
	}

	return streamingDefaultErrorMessage
}

// EncodeAudio sends the raw audio as a BINARY frame.
func (p *streamingProtocol) EncodeAudio(chunk []byte, _ bool) (livetranscription.Frame, error) {
	return livetranscription.BinaryFrame(chunk), nil
}

// AudioEnded closes a turn detection session, or finalizes the audio.
func (p *streamingProtocol) AudioEnded(session *livetranscription.TranscriptionSession, _ int64) {
	if !p.turnDetection {
		session.Send(livetranscription.TextFrame([]byte(streamingFinalizeCommand)))

		return
	}

	session.SendJSON(streamingCloseMessage{
		Type: streamingCloseCommand,
	})
}

// TimerFired is unused: Cartesia reports when it is done.
func (p *streamingProtocol) TimerFired(*livetranscription.TranscriptionSession) {}

// Closed finishes the stream with the transcripts so far, as upstream.
func (p *streamingProtocol) Closed(session *livetranscription.TranscriptionSession, _ *websocket.CloseError) {
	p.finish(session)
}

func (p *streamingProtocol) finish(session *livetranscription.TranscriptionSession) {
	separator := ""
	if p.turnDetection {
		separator = " "
	}

	finish := model.TranscriptFinish{
		Text:     strings.Join(p.finalTexts, separator),
		Language: streamingLanguage,
	}

	if p.duration > 0 {
		finish.Duration = seconds(p.duration)
	}

	session.Finish(finish)
}
