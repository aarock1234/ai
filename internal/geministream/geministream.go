// Package geministream starts the Gemini Live API streaming transcriptions
// that the google and googlevertex providers share once their endpoint,
// authentication and setup are prepared.
package geministream

import (
	"context"
	"encoding/json"
	"time"

	"github.com/aarock1234/ai/internal/geminilive"
	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/pkg/model"
)

// Transcription is a prepared Live API streaming transcription.
type Transcription struct {
	ModelID string

	// Dial describes the WebSocket connection.
	Dial livetranscription.DialOptions

	// Setup is the encoded session setup.
	Setup geminilive.EncodedSetup

	// Options are the call's stream options.
	Options model.TranscriptionStreamOptions

	// Timestamp is the start of the call.
	Timestamp time.Time

	// DefaultErrorMessage describes server errors without a message, such
	// as "Google Live API error".
	DefaultErrorMessage string

	// Name names the session in errors, such as "google live
	// transcription".
	Name string

	// Metadata returns the provider metadata of the reported usage.
	Metadata func(usageMetadata json.RawMessage) model.ProviderMetadata
}

// Start dials the Live API and returns the streaming result, which owns the
// connection.
func Start(ctx context.Context, transcription Transcription) (*model.TranscriptionStreamResult, error) {
	conn, err := livetranscription.Dial(ctx, transcription.Dial)
	if err != nil {
		return nil, err
	}

	protocol := geminilive.NewProtocol(geminilive.ProtocolSetup{
		SetupMessage:        transcription.Setup.Message,
		DefaultErrorMessage: transcription.DefaultErrorMessage,
		Metadata:            transcription.Metadata,
	})

	result := livetranscription.NewResult(ctx, livetranscription.TranscriptionSetup{
		Conn:             conn,
		Protocol:         protocol,
		Audio:            transcription.Options.Audio,
		IncludeRawChunks: transcription.Options.IncludeRawChunks,
		Name:             transcription.Name,
	})

	result.Request = model.RequestInfo{
		Body: transcription.Setup.Body,
	}

	result.Response = model.ResponseInfo{
		Timestamp: transcription.Timestamp,
		ModelID:   transcription.ModelID,
		Headers:   conn.ResponseHeader(),
	}

	return result, nil
}
