// Package transcriptionstream implements the experimental transcription
// stream WebSocket envelope (v1): the standard serialization of a streaming
// transcription over a WebSocket, shared by clients and servers so the two
// sides cannot drift.
//
// The rules of the envelope:
//
//  1. The client sends exactly one start TEXT frame first.
//  2. Audio rides BINARY frames of raw bytes in the declared input format.
//  3. The client ends the audio with the audio-done TEXT frame; a plain
//     close without it is an abort.
//  4. Every server TEXT frame is one JSON transcription stream part.
//     Timestamps are ISO 8601 strings, and errors are {name, message}.
//  5. The server closes with code 1000 after the finish part; on failure it
//     sends an error part and closes with another code. A close without a
//     prior finish is an error.
//  6. Unknown frame and part types are ignored in both directions.
//  7. Servers may limit frame sizes; clients should send audio frames of at
//     most 64 KiB.
//  8. Connection establishment (URL, authentication) is out of scope.
//
// The envelope validates frame shapes only; server policy such as accepted
// audio formats layers on top. Go's JSON decoding has no prototype
// pollution, so the upstream secure parsing needs no counterpart.
package transcriptionstream

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// Client frame types.
const (
	// StartFrameType is the type of the first client TEXT frame.
	StartFrameType = "transcription-stream.start"

	// AudioDoneFrameType is the type of the client TEXT frame that ends the
	// audio input.
	AudioDoneFrameType = "transcription-stream.audio-done"
)

// MaxAudioFrameBytes is the largest audio frame clients should send.
const MaxAudioFrameBytes = 64 << 10

// StartFrame is the client's session start frame.
type StartFrame struct {
	Type             string                     `json:"type"`
	InputAudioFormat AudioFormat                `json:"inputAudioFormat"`
	ProviderOptions  map[string]json.RawMessage `json:"providerOptions,omitempty"`
	IncludeRawChunks *bool                      `json:"includeRawChunks,omitempty"`
}

// AudioFormat is the wire form of model.AudioFormat.
type AudioFormat struct {
	Type string `json:"type"`
	Rate *int   `json:"rate,omitempty"`
}

// audioDoneFrame is the client frame that ends the audio input.
type audioDoneFrame struct {
	Type string `json:"type"`
}

// ClientFrameKind classifies a client TEXT frame.
type ClientFrameKind int

// Client frame kinds.
const (
	// ClientFrameUnknown is an unrecognized frame type, ignored for forward
	// compatibility.
	ClientFrameUnknown ClientFrameKind = iota

	ClientFrameStart
	ClientFrameAudioDone

	// ClientFrameInvalid is malformed JSON or a recognized frame with an
	// invalid shape.
	ClientFrameInvalid
)

// ClientFrame is a parsed client TEXT frame. Start is set for start frames
// and Message for invalid ones.
type ClientFrame struct {
	Kind    ClientFrameKind
	Start   StartFrame
	Message string
}

// startFrameFields are the fields of a start frame, decoded one at a time
// so that each invalid field gets its own message.
type startFrameFields struct {
	InputAudioFormat json.RawMessage `json:"inputAudioFormat"`
	ProviderOptions  json.RawMessage `json:"providerOptions"`
	IncludeRawChunks json.RawMessage `json:"includeRawChunks"`
}

// audioFormatFields are the fields of a start frame's audio format.
type audioFormatFields struct {
	Type *string         `json:"type"`
	Rate json.RawMessage `json:"rate"`
}

// EncodeStartFrame returns the start frame for options.
func EncodeStartFrame(options model.TranscriptionStreamOptions) ([]byte, error) {
	frame := StartFrame{
		Type: StartFrameType,
		InputAudioFormat: AudioFormat{
			Type: options.InputAudioFormat.Type,
		},
	}

	if options.InputAudioFormat.Rate != 0 {
		frame.InputAudioFormat.Rate = new(options.InputAudioFormat.Rate)
	}

	if options.IncludeRawChunks {
		frame.IncludeRawChunks = new(true)
	}

	providerOptions, err := encodeProviderOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	frame.ProviderOptions = providerOptions

	encoded, err := json.Marshal(frame)
	if err != nil {
		return nil, fmt.Errorf("encode transcription stream start frame: %w", err)
	}

	return encoded, nil
}

// encodeProviderOptions encodes each namespace of options as JSON.
func encodeProviderOptions(options model.ProviderOptions) (map[string]json.RawMessage, error) {
	if len(options) == 0 {
		return nil, nil
	}

	encoded := make(map[string]json.RawMessage, len(options))
	for namespace, value := range options {
		raw, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("%w: encode %s provider options: %w", model.ErrInvalidArgument, namespace, err)
		}

		encoded[namespace] = raw
	}

	return encoded, nil
}

// EncodeAudioDoneFrame returns the frame that ends the audio input.
func EncodeAudioDoneFrame() ([]byte, error) {
	encoded, err := json.Marshal(audioDoneFrame{
		Type: AudioDoneFrameType,
	})
	if err != nil {
		return nil, fmt.Errorf("encode transcription stream audio-done frame: %w", err)
	}

	return encoded, nil
}

// ParseClientFrame classifies a client TEXT frame on the server side. It
// validates the envelope shape only.
func ParseClientFrame(text []byte) ClientFrame {
	var header struct {
		Type json.RawMessage `json:"type"`
	}

	if !json.Valid(text) {
		return invalidClientFrame("malformed JSON")
	}

	if !decodeObject(text, &header) {
		return invalidClientFrame("frame must be a JSON object")
	}

	var frameType string
	if !decodeNonNull(header.Type, &frameType) {
		return invalidClientFrame("frame type must be a string")
	}

	switch frameType {
	case StartFrameType:
		return parseStartFrame(text)

	case AudioDoneFrameType:
		return ClientFrame{
			Kind: ClientFrameAudioDone,
		}

	default:
		return ClientFrame{
			Kind: ClientFrameUnknown,
		}
	}
}

// parseStartFrame validates each field of a start frame.
func parseStartFrame(text []byte) ClientFrame {
	var fields startFrameFields
	if err := json.Unmarshal(text, &fields); err != nil {
		return invalidClientFrame("malformed JSON")
	}

	format, message := parseAudioFormat(fields.InputAudioFormat)
	if message != "" {
		return invalidClientFrame(message)
	}

	frame := StartFrame{
		Type:             StartFrameType,
		InputAudioFormat: format,
	}

	if wirejson.IsSet(fields.ProviderOptions) {
		if !decodeObject(fields.ProviderOptions, &frame.ProviderOptions) {
			return invalidClientFrame("providerOptions must be an object when present")
		}
	}

	if wirejson.IsSet(fields.IncludeRawChunks) {
		var include bool
		if !decodeNonNull(fields.IncludeRawChunks, &include) {
			return invalidClientFrame("includeRawChunks must be a boolean when present")
		}

		frame.IncludeRawChunks = &include
	}

	return ClientFrame{
		Kind:  ClientFrameStart,
		Start: frame,
	}
}

// parseAudioFormat validates the audio format of a start frame. It returns
// a message describing the problem, or an empty one when it is valid.
func parseAudioFormat(raw json.RawMessage) (AudioFormat, string) {
	const formatMessage = "start frame must have an inputAudioFormat object with a string type"

	var fields audioFormatFields
	if !decodeObject(raw, &fields) || fields.Type == nil {
		return AudioFormat{}, formatMessage
	}

	format := AudioFormat{
		Type: *fields.Type,
	}

	if wirejson.IsSet(fields.Rate) {
		var rate int
		if !decodeNonNull(fields.Rate, &rate) {
			return AudioFormat{}, "inputAudioFormat.rate must be a number when present"
		}

		format.Rate = &rate
	}

	return format, ""
}

// decodeObject decodes raw into target when it holds a JSON object, and
// reports whether it did.
func decodeObject(raw []byte, target any) bool {
	if !isJSONObject(raw) {
		return false
	}

	return json.Unmarshal(raw, target) == nil
}

// decodeNonNull decodes raw into target when it holds a value other than
// null, and reports whether it did.
func decodeNonNull(raw json.RawMessage, target any) bool {
	if wirejson.IsNull(raw) {
		return false
	}

	return json.Unmarshal(raw, target) == nil
}

func invalidClientFrame(message string) ClientFrame {
	return ClientFrame{
		Kind:    ClientFrameInvalid,
		Message: message,
	}
}

// isJSONObject reports whether raw holds a JSON object.
func isJSONObject(raw []byte) bool {
	trimmed := bytes.TrimSpace(raw)

	return len(trimmed) > 0 && trimmed[0] == '{'
}
