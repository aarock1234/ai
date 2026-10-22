// Package livetranscription runs streaming transcriptions and speech
// translations over provider WebSocket APIs. It owns the concurrency of a
// live stream: one goroutine reads server messages, another sends the
// caller's audio, and the consumer's goroutine runs the provider protocol
// and yields the parts. Providers own the URL, authentication and the
// message protocol.
package livetranscription

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// Frame is one message sent to the server.
type Frame struct {
	Type websocket.MessageType
	Data []byte
}

// TextFrame returns a TEXT frame of data.
func TextFrame(data []byte) Frame {
	return Frame{
		Type: websocket.TextMessage,
		Data: data,
	}
}

// BinaryFrame returns a BINARY frame of data.
func BinaryFrame(data []byte) Frame {
	return Frame{
		Type: websocket.BinaryMessage,
		Data: data,
	}
}

// JSONFrame encodes value as a TEXT frame.
func JSONFrame(value any) (Frame, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return Frame{}, fmt.Errorf("encode websocket message: %w", err)
	}

	return TextFrame(encoded), nil
}

// Protocol is a provider's half of a live stream whose parts are of type
// Part. Every method except EncodeAudio runs on the consumer's goroutine,
// one call at a time, so a protocol keeps its state without locks. A method
// that ends the stream calls Session.Finish, Session.Fail or Session.End.
type Protocol[Part any] interface {
	// Open runs once, right after the stream starts.
	Open(session *Session[Part])

	// HandleMessage handles one server message. Messages that are not JSON
	// are ignored before they reach the protocol.
	HandleMessage(session *Session[Part], message []byte)

	// EncodeAudio returns the frame of one chunk of audio. It runs on the
	// audio goroutine and must not keep chunk, whose buffer is reused.
	EncodeAudio(chunk []byte, first bool) (Frame, error)

	// AudioEnded runs once the audio reached io.EOF and every chunk was
	// sent, with the number of audio bytes sent. It does not run when the
	// audio was stopped.
	AudioEnded(session *Session[Part], totalBytes int64)

	// TimerFired runs when the timer set with Session.SetTimer expires.
	TimerFired(session *Session[Part])

	// Closed runs when the server closes the connection before the stream
	// ended. The stream fails with closeErr unless Closed ends it.
	Closed(session *Session[Part], closeErr *websocket.CloseError)
}

// TranscriptionSession is the session of a streaming transcription.
type TranscriptionSession = Session[model.TranscriptionStreamPart]

// TranslationSession is the session of a streaming speech translation.
type TranslationSession = Session[model.SpeechTranslationStreamPart]

// Setup is everything a live stream needs once its connection is open.
type Setup[Part any] struct {
	Conn     *websocket.Conn
	Protocol Protocol[Part]

	// Audio is the caller's audio. The stream never closes it.
	Audio io.Reader

	// AudioChunkBytes bounds the audio of one frame. Zero selects 32 KiB.
	AudioChunkBytes int

	// IncludeRawChunks emits every JSON server message as a Raw part
	// before the protocol handles it.
	IncludeRawChunks bool

	// KeepPendingTimer makes Session.SetTimer keep a pending timer instead
	// of restarting it, for protocols whose quiet window starts once.
	KeepPendingTimer bool

	// Name prefixes connection errors, such as "elevenlabs realtime
	// transcription".
	Name string
}

// TranscriptionSetup is the setup of a streaming transcription.
type TranscriptionSetup = Setup[model.TranscriptionStreamPart]

// TranslationSetup is the setup of a streaming speech translation.
type TranslationSetup = Setup[model.SpeechTranslationStreamPart]

// NewResult returns the stream of a live transcription. Ranging over its
// parts runs the transcription; releasing it stops the audio and closes the
// connection, normally once the transcript finished.
func NewResult(ctx context.Context, setup TranscriptionSetup) *model.TranscriptionStreamResult {
	session := newSession(setup, partKinds[model.TranscriptionStreamPart]{
		errorPart: transcriptionError,
		rawPart:   transcriptionRaw,
	})

	return model.NewTranscriptionStreamResult(session.parts(ctx), session.close)
}

// NewTranslationResult returns the stream of a live speech translation,
// like NewResult.
func NewTranslationResult(ctx context.Context, setup TranslationSetup) *model.SpeechTranslationStreamResult {
	session := newSession(setup, partKinds[model.SpeechTranslationStreamPart]{
		errorPart: translationError,
		rawPart:   translationRaw,
	})

	return model.NewSpeechTranslationStreamResult(session.parts(ctx), session.close)
}

// partKinds builds the parts that the session itself emits.
type partKinds[Part any] struct {
	errorPart func(err error) Part
	rawPart   func(message json.RawMessage) Part
}

func transcriptionError(err error) model.TranscriptionStreamPart {
	return model.StreamError{
		Err: err,
	}
}

func transcriptionRaw(message json.RawMessage) model.TranscriptionStreamPart {
	return model.Raw{
		Value: message,
	}
}

func translationError(err error) model.SpeechTranslationStreamPart {
	return model.StreamError{
		Err: err,
	}
}

func translationRaw(message json.RawMessage) model.SpeechTranslationStreamPart {
	return model.Raw{
		Value: message,
	}
}
