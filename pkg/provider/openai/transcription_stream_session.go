package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// Realtime event types.
const (
	realtimeAppendEvent         = "input_audio_buffer.append"
	realtimeCommitEvent         = "input_audio_buffer.commit"
	realtimeDeltaEvent          = "conversation.item.input_audio_transcription.delta"
	realtimeCompletedEvent      = "conversation.item.input_audio_transcription.completed"
	realtimeErrorEvent          = "error"
	realtimeDefaultErrorMessage = "OpenAI realtime error"
)

// realtimeAudioAppend sends one chunk of base64 audio.
type realtimeAudioAppend struct {
	Type  string `json:"type"`
	Audio string `json:"audio"`
}

// realtimeClientEvent is a client event without fields beyond its type,
// such as an audio commit.
type realtimeClientEvent struct {
	Type string `json:"type"`
}

// realtimeServerEvent is a server event; fields beyond the type depend on
// it.
type realtimeServerEvent struct {
	Type       string               `json:"type"`
	ItemID     string               `json:"item_id"`
	Delta      string               `json:"delta"`
	Transcript string               `json:"transcript"`
	Error      *realtimeErrorDetail `json:"error"`
}

type realtimeErrorDetail struct {
	Message string `json:"message"`
}

// realtimeTranscriptionProtocol is the realtime transcription protocol. The
// session is configured and the audio starts as soon as the connection
// opens; the audio is committed once it ends, and the completed transcript
// finishes the stream.
type realtimeTranscriptionProtocol struct {
	warnings      []model.Warning
	sessionUpdate []byte
	language      string
}

// Open starts the stream, configures the session and starts the audio.
func (p *realtimeTranscriptionProtocol) Open(session *livetranscription.TranscriptionSession) {
	session.Emit(model.StreamStart{
		Warnings: p.warnings,
	})
	session.Send(livetranscription.TextFrame(p.sessionUpdate))
	session.StartAudio()
}

// HandleMessage handles one server event.
func (p *realtimeTranscriptionProtocol) HandleMessage(session *livetranscription.TranscriptionSession, message []byte) {
	var event realtimeServerEvent
	if err := json.Unmarshal(message, &event); err != nil {
		session.Fail(fmt.Errorf("%w: decode realtime event: %w", model.ErrInvalidResponse, err))

		return
	}

	switch event.Type {
	case realtimeDeltaEvent:
		session.Emit(model.TranscriptDelta{
			ID:    event.ItemID,
			Delta: event.Delta,
		})

	case realtimeCompletedEvent:
		p.complete(session, event)

	case realtimeErrorEvent:
		session.Fail(realtimeEventError(event.Error, message))
	}
}

// complete emits the final transcript and finishes the stream.
func (p *realtimeTranscriptionProtocol) complete(session *livetranscription.TranscriptionSession, event realtimeServerEvent) {
	if event.ItemID != "" {
		session.Emit(model.TranscriptFinal{
			ID:   event.ItemID,
			Text: event.Transcript,
		})
	}

	session.Finish(model.TranscriptFinish{
		Text:     event.Transcript,
		Language: p.language,
	})
}

// EncodeAudio sends one chunk of base64 audio.
func (p *realtimeTranscriptionProtocol) EncodeAudio(chunk []byte, _ bool) (livetranscription.Frame, error) {
	return livetranscription.JSONFrame(realtimeAudioAppend{
		Type:  realtimeAppendEvent,
		Audio: base64.StdEncoding.EncodeToString(chunk),
	})
}

// AudioEnded commits the audio.
func (p *realtimeTranscriptionProtocol) AudioEnded(session *livetranscription.TranscriptionSession, _ int64) {
	session.SendJSON(realtimeClientEvent{
		Type: realtimeCommitEvent,
	})
}

// TimerFired is unused: the completed transcript finishes the stream.
func (p *realtimeTranscriptionProtocol) TimerFired(*livetranscription.TranscriptionSession) {}

// Closed ends the stream without an error, as upstream; the missing finish
// part tells the caller that no transcript arrived.
func (p *realtimeTranscriptionProtocol) Closed(session *livetranscription.TranscriptionSession, _ *websocket.CloseError) {
	session.End()
}

// realtimeEventError converts the error of an error event.
func realtimeEventError(detail *realtimeErrorDetail, message []byte) error {
	errorMessage := realtimeDefaultErrorMessage
	if detail != nil && detail.Message != "" {
		errorMessage = detail.Message
	}

	return &model.APICallError{
		Message:      errorMessage,
		ResponseBody: string(message),
	}
}
