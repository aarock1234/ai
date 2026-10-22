package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/transcriptionstream"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// errTranscriptionStreamClosed reports a stream that closed without a
// finish part or an error part.
var errTranscriptionStreamClosed = errors.New("AI Gateway transcription stream closed before a finish part was received")

// serverErrorPayload is the payload of a server error part.
type serverErrorPayload struct {
	Message *string `json:"message"`
	Type    *string `json:"type"`
}

// transcriptionStreamProtocol is the client side of the transcription
// stream envelope. Server frames are stream parts, passed through as they
// arrive; error parts are not terminal until the server closes.
type transcriptionStreamProtocol struct {
	startFrame     []byte
	audioDoneFrame []byte
	method         authMethod

	// serverErr is the latest error part.
	serverErr *transcriptionstream.RemoteError
}

// Open sends the start frame and starts the audio.
func (p *transcriptionStreamProtocol) Open(session *livetranscription.TranscriptionSession) {
	session.Send(livetranscription.TextFrame(p.startFrame))
	session.StartAudio()
}

// HandleMessage passes one server part through. Unknown and malformed parts
// are ignored, as the envelope requires.
func (p *transcriptionStreamProtocol) HandleMessage(session *livetranscription.TranscriptionSession, message []byte) {
	part, ok := transcriptionstream.ParsePart(message)
	if !ok {
		return
	}

	switch typed := part.(type) {
	case model.TranscriptFinish:
		session.Finish(typed)

	case model.StreamError:
		p.handleServerError(session, typed)

	default:
		session.Emit(part)
	}
}

// handleServerError emits an error part and stops the audio: error parts
// end the stream, while the server may hold the connection open, for
// example for its final billing flush.
func (p *transcriptionStreamProtocol) handleServerError(session *livetranscription.TranscriptionSession, part model.StreamError) {
	if remote, ok := errors.AsType[*transcriptionstream.RemoteError](part.Err); ok {
		p.serverErr = remote
	}

	session.StopAudio()
	session.Emit(part)
}

// EncodeAudio sends the raw audio as a BINARY frame.
func (p *transcriptionStreamProtocol) EncodeAudio(chunk []byte, _ bool) (livetranscription.Frame, error) {
	return livetranscription.BinaryFrame(chunk), nil
}

// AudioEnded sends the audio-done frame.
func (p *transcriptionStreamProtocol) AudioEnded(session *livetranscription.TranscriptionSession, _ int64) {
	session.Send(livetranscription.TextFrame(p.audioDoneFrame))
}

// TimerFired is unused: the server finishes the stream.
func (p *transcriptionStreamProtocol) TimerFired(*livetranscription.TranscriptionSession) {}

// Closed fails the stream with the server's error, or reports that it
// closed without finishing.
func (p *transcriptionStreamProtocol) Closed(session *livetranscription.TranscriptionSession, _ *websocket.CloseError) {
	if p.serverErr == nil {
		session.Fail(errTranscriptionStreamClosed)

		return
	}

	session.Fail(p.serverPartError(p.serverErr))
}

// serverPartError converts an error part into the gateway Error of its
// type. Unknown types keep the server's message.
func (p *transcriptionStreamProtocol) serverPartError(remote *transcriptionstream.RemoteError) error {
	var payload serverErrorPayload
	if json.Unmarshal(remote.Value, &payload) != nil || payload.Message == nil || payload.Type == nil {
		return fmt.Errorf("AI Gateway transcription stream failed: %w", remote)
	}

	statusCode, known := serverErrorStatus(ErrorType(*payload.Type))
	if !known {
		return fmt.Errorf("AI Gateway transcription stream failed: %w", remote)
	}

	body, err := json.Marshal(errorBody{
		Error: &errorDetail{
			Message: payload.Message,
			Type:    payload.Type,
		},
	})
	if err != nil {
		return fmt.Errorf("AI Gateway transcription stream failed: %w", remote)
	}

	return errorFromResponse(errorResponseSettings{
		body:       body,
		statusCode: statusCode,
		cause:      remote,
		method:     p.method,
	})
}

// serverErrorStatus returns the status code that an error part's type
// implies; the WebSocket has no HTTP status.
func serverErrorStatus(errorType ErrorType) (int, bool) {
	switch errorType {
	case ErrorAuthentication:
		return http.StatusUnauthorized, true
	case ErrorFailedDependency:
		return http.StatusFailedDependency, true
	case ErrorForbidden:
		return http.StatusForbidden, true
	case ErrorInternalServer:
		return http.StatusInternalServerError, true
	case ErrorInvalidRequest:
		return http.StatusBadRequest, true
	case ErrorModelNotFound:
		return http.StatusNotFound, true
	case ErrorRateLimit:
		return http.StatusTooManyRequests, true
	default:
		return 0, false
	}
}
