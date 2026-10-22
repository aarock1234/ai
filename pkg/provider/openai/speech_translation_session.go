package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// Realtime translation event types.
const (
	translationAppendEvent          = "session.input_audio_buffer.append"
	translationCloseEvent           = "session.close"
	translationOutputAudioEvent     = "session.output_audio.delta"
	translationOutputTranscription  = "session.output_transcript.delta"
	translationInputTranscription   = "session.input_transcript.delta"
	translationClosedEvent          = "session.closed"
	translationUnexpectedCloseError = "openai realtime translation websocket closed unexpectedly before finishing"
)

// translationServerEvent is a translation server event; fields beyond the
// type depend on it.
type translationServerEvent struct {
	Type  string               `json:"type"`
	Delta string               `json:"delta"`
	Error *realtimeErrorDetail `json:"error"`
}

// realtimeTranslationProtocol is the realtime translation protocol. The
// session is configured and the audio starts as soon as the connection
// opens; the session is closed once the audio ends, and the server's
// session.closed event finishes the stream. Error events are reported
// without ending the stream, as upstream.
type realtimeTranslationProtocol struct {
	warnings      []model.Warning
	sessionUpdate []byte

	sourceText      strings.Builder
	translationText strings.Builder
}

// Open starts the stream, configures the session and starts the audio.
func (p *realtimeTranslationProtocol) Open(session *livetranscription.TranslationSession) {
	session.Emit(model.StreamStart{
		Warnings: p.warnings,
	})
	session.Send(livetranscription.TextFrame(p.sessionUpdate))
	session.StartAudio()
}

// HandleMessage handles one server event.
func (p *realtimeTranslationProtocol) HandleMessage(session *livetranscription.TranslationSession, message []byte) {
	var event translationServerEvent
	if err := json.Unmarshal(message, &event); err != nil {
		session.Fail(fmt.Errorf("%w: decode realtime translation event: %w", model.ErrInvalidResponse, err))

		return
	}

	switch event.Type {
	case translationOutputAudioEvent:
		p.outputAudio(session, event.Delta)

	case translationOutputTranscription:
		p.translationText.WriteString(event.Delta)

		session.Emit(model.OutputTextDelta{
			Delta: event.Delta,
		})

	case translationInputTranscription:
		p.sourceText.WriteString(event.Delta)

		session.Emit(model.SourceTranscriptDelta{
			Delta: event.Delta,
		})

	case translationClosedEvent:
		p.finish(session)

	case realtimeErrorEvent:
		session.Report(realtimeEventError(event.Error, message))
	}
}

// outputAudio emits decoded audio. Empty deltas carry no audio and are
// skipped; audio that is not base64 is reported without ending the stream.
func (p *realtimeTranslationProtocol) outputAudio(session *livetranscription.TranslationSession, delta string) {
	if delta == "" {
		return
	}

	audio, err := base64.StdEncoding.DecodeString(delta)
	if err != nil {
		session.Report(fmt.Errorf("%w: decode translated audio: %w", model.ErrInvalidResponse, err))

		return
	}

	session.Emit(model.TranslationAudio{
		Audio: audio,
	})
}

// finish emits the final texts and ends the stream with them.
func (p *realtimeTranslationProtocol) finish(session *livetranscription.TranslationSession) {
	sourceText := p.sourceText.String()
	translationText := p.translationText.String()

	if sourceText != "" {
		session.Emit(model.SourceTranscriptFinal{
			Text: sourceText,
		})
	}

	if translationText != "" {
		session.Emit(model.OutputTextFinal{
			Text: translationText,
		})
	}

	session.Finish(model.TranslationFinish{
		SourceText: sourceText,
		OutputText: translationText,
	})
}

// EncodeAudio sends one chunk of base64 audio.
func (p *realtimeTranslationProtocol) EncodeAudio(chunk []byte, _ bool) (livetranscription.Frame, error) {
	return livetranscription.JSONFrame(realtimeAudioAppend{
		Type:  translationAppendEvent,
		Audio: base64.StdEncoding.EncodeToString(chunk),
	})
}

// AudioEnded asks the provider to finish the translation of the audio sent.
func (p *realtimeTranslationProtocol) AudioEnded(session *livetranscription.TranslationSession, _ int64) {
	session.SendJSON(realtimeClientEvent{
		Type: translationCloseEvent,
	})
}

// TimerFired is unused: the session.closed event finishes the stream.
func (p *realtimeTranslationProtocol) TimerFired(*livetranscription.TranslationSession) {}

// Closed fails the stream: a close before the session closed is an
// abnormal end, reported with the close diagnostics.
func (p *realtimeTranslationProtocol) Closed(session *livetranscription.TranslationSession, closeErr *websocket.CloseError) {
	session.Fail(fmt.Errorf(
		"%s (code %d, reason: %q): %w",
		translationUnexpectedCloseError,
		closeErr.Code,
		closeErr.Reason,
		closeErr,
	))
}
