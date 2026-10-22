// Package bridge is the host side of the bridge transport shared by
// bridge-backed harness adapters: adapters that run their agent runtime
// inside the sandbox behind a small bridge process and talk to it over a
// WebSocket. It ports the bridge protocol of upstream @ai-sdk/harness
// (harness-v1-bridge-protocol) and the host utilities of
// @ai-sdk/harness/utils: the reconnecting Channel, the readiness handshake,
// bridge tokens, diagnostics and user message submission.
//
// The in-sandbox bridge runtime (upstream @ai-sdk/harness/bridge) is
// JavaScript that runs in the sandbox's Node.js runtime and is not ported;
// Go adapters ship their bridge bundle as files of their bootstrap recipe.
package bridge

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/model"
)

// ErrInvalidFrame reports a bridge frame that does not match the protocol.
var ErrInvalidFrame = errors.New("invalid bridge frame")

// Message is one frame a bridge sends to the host: a StreamMessage carrying
// a harness event, one of the control frames Hello, UserMessageResponse,
// Stop, Thread, SandboxLog and DebugEvent, or an adapter-specific
// CustomMessage.
type Message interface {
	isMessage()
}

// StreamMessage carries one harness.StreamPart.
type StreamMessage struct {
	Part harness.StreamPart
}

// Hello is sent as soon as the bridge accepts an authenticated connection.
// The host waits for it before sending start or resume, because some
// sandboxes complete the WebSocket handshake before the connection reaches
// the bridge process.
type Hello struct {
	// State is the bridge's lifecycle state.
	State string

	// LastSeq is the highest event sequence number the bridge emitted, and
	// HasLastSeq reports whether the bridge sent one.
	LastSeq    int64
	HasLastSeq bool

	// UserMessageResponses reports that the bridge answers user messages
	// with UserMessageResponse frames.
	UserMessageResponses bool
}

// UserMessageResponse answers a UserMessageCommand. This frame is
// experimental.
type UserMessageResponse struct {
	MessageID string
	Accepted  bool

	// Error explains a rejection. It is empty when accepted.
	Error string
}

// Stop answers a StopCommand with the adapter-specific data the host stores
// in lifecycle state.
type Stop struct {
	Data json.RawMessage
}

// Thread announces a resume coordinate, such as Codex's thread ID, before
// the bridge stops.
type Thread struct {
	ThreadID string
}

// SandboxLog is one console line captured inside the sandbox.
type SandboxLog struct {
	Source string

	// Stream is "stdout" or "stderr".
	Stream string
	Line   string
}

// DebugEvent is a structured diagnostic the bridge emitted.
type DebugEvent struct {
	Level     harness.DebugLevel
	Subsystem string
	Message   string
	Attrs     map[string]any
	Error     *harness.DiagnosticError
}

// CustomMessage is a frame of an adapter-specific type the adapter listed
// in ChannelOptions.CustomFrameTypes. Data is the complete frame.
type CustomMessage struct {
	Type string
	Data json.RawMessage
}

func (StreamMessage) isMessage()       {}
func (Hello) isMessage()               {}
func (UserMessageResponse) isMessage() {}
func (Stop) isMessage()                {}
func (Thread) isMessage()              {}
func (SandboxLog) isMessage()          {}
func (DebugEvent) isMessage()          {}
func (CustomMessage) isMessage()       {}

// Frame is one decoded bridge frame with its sequence number.
type Frame struct {
	Message Message

	// Seq is the frame's sequence number, and HasSeq reports whether it had
	// one. Events carry one; control frames may not.
	Seq    int64
	HasSeq bool
}

// RemoteError is an error a bridge reported in an error frame. Value is the
// error as the bridge serialized it.
type RemoteError struct {
	Value json.RawMessage
}

// Error returns the bridge's error message, or its JSON when it has none.
func (e *RemoteError) Error() string {
	var serialized struct {
		Name    string `json:"name"`
		Message string `json:"message"`
	}

	if err := json.Unmarshal(e.Value, &serialized); err == nil && serialized.Message != "" {
		if serialized.Name != "" {
			return serialized.Name + ": " + serialized.Message
		}

		return serialized.Message
	}

	var text string
	if err := json.Unmarshal(e.Value, &text); err == nil {
		return text
	}

	return string(e.Value)
}

// DecodeFrame decodes and validates one bridge frame.
func DecodeFrame(data []byte) (Frame, error) {
	var envelope struct {
		Type string   `json:"type"`
		Seq  *float64 `json:"seq"`
	}

	if err := json.Unmarshal(data, &envelope); err != nil {
		return Frame{}, fmt.Errorf("%w: %w", ErrInvalidFrame, err)
	}

	message, err := decodeMessage(envelope.Type, data)
	if err != nil {
		return Frame{}, err
	}

	frame := Frame{
		Message: message,
	}

	if envelope.Seq != nil {
		frame.Seq = int64(*envelope.Seq)
		frame.HasSeq = true
	}

	return frame, nil
}

// decodeCustomFrame keeps a frame of one of the adapter-specific types whole.
// It reports false for every other frame.
func decodeCustomFrame(data []byte, frameTypes []string) (Frame, bool) {
	if len(frameTypes) == 0 {
		return Frame{}, false
	}

	var envelope struct {
		Type string   `json:"type"`
		Seq  *float64 `json:"seq"`
	}

	if err := json.Unmarshal(data, &envelope); err != nil || !slices.Contains(frameTypes, envelope.Type) {
		return Frame{}, false
	}

	frame := Frame{
		Message: CustomMessage{
			Type: envelope.Type,
			Data: slices.Clone(data),
		},
	}

	if envelope.Seq != nil {
		frame.Seq = int64(*envelope.Seq)
		frame.HasSeq = true
	}

	return frame, true
}

// decodeMessage decodes a frame of a known type.
func decodeMessage(frameType string, data []byte) (Message, error) {
	if decode, ok := controlDecoders[frameType]; ok {
		return decode(data)
	}

	part, err := decodeStreamPart(frameType, data)
	if err != nil {
		return nil, err
	}

	return StreamMessage{
		Part: part,
	}, nil
}

// controlDecoders decode the control frames by type.
var controlDecoders = map[string]func(data []byte) (Message, error){
	"bridge-hello":          decodeHello,
	"user-message-response": decodeUserMessageResponse,
	"bridge-stop":           decodeStop,
	"bridge-thread":         decodeThread,
	"sandbox-log":           decodeSandboxLog,
	"debug-event":           decodeDebugEvent,
}

// decodeHello decodes a bridge-hello frame.
func decodeHello(data []byte) (Message, error) {
	var wire struct {
		State        string   `json:"state"`
		LastSeq      *float64 `json:"lastSeq"`
		Capabilities struct {
			UserMessageResponses bool `json:"experimental_userMessageResponses"`
		} `json:"capabilities"`
	}

	if err := decodeWire(data, &wire); err != nil {
		return nil, err
	}

	hello := Hello{
		State:                wire.State,
		UserMessageResponses: wire.Capabilities.UserMessageResponses,
	}

	if wire.LastSeq != nil {
		hello.LastSeq = int64(*wire.LastSeq)
		hello.HasLastSeq = true
	}

	return hello, nil
}

// decodeUserMessageResponse decodes a user-message-response frame.
func decodeUserMessageResponse(data []byte) (Message, error) {
	var wire struct {
		MessageID *string `json:"messageId"`
		Accepted  *bool   `json:"accepted"`
		Error     *struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := decodeWire(data, &wire); err != nil {
		return nil, err
	}

	if wire.MessageID == nil || wire.Accepted == nil {
		return nil, missingField("user-message-response", "messageId and accepted")
	}

	response := UserMessageResponse{
		MessageID: *wire.MessageID,
		Accepted:  *wire.Accepted,
	}

	if wire.Error != nil {
		response.Error = wire.Error.Message
	}

	return response, nil
}

// decodeStop decodes a bridge-stop frame.
func decodeStop(data []byte) (Message, error) {
	var wire struct {
		Data json.RawMessage `json:"data"`
	}

	if err := decodeWire(data, &wire); err != nil {
		return nil, err
	}

	return Stop{
		Data: wire.Data,
	}, nil
}

// decodeThread decodes a bridge-thread frame.
func decodeThread(data []byte) (Message, error) {
	var wire struct {
		ThreadID *string `json:"threadId"`
	}

	if err := decodeWire(data, &wire); err != nil {
		return nil, err
	}

	if wire.ThreadID == nil {
		return nil, missingField("bridge-thread", "threadId")
	}

	return Thread{
		ThreadID: *wire.ThreadID,
	}, nil
}

// decodeSandboxLog decodes a sandbox-log frame.
func decodeSandboxLog(data []byte) (Message, error) {
	var wire struct {
		Source *string `json:"source"`
		Stream *string `json:"stream"`
		Line   *string `json:"line"`
	}

	if err := decodeWire(data, &wire); err != nil {
		return nil, err
	}

	hasFields := wire.Source != nil && wire.Stream != nil && wire.Line != nil
	if !hasFields || !isProcessStream(*wire.Stream) {
		return nil, missingField("sandbox-log", "source, stream and line")
	}

	return SandboxLog{
		Source: *wire.Source,
		Stream: *wire.Stream,
		Line:   *wire.Line,
	}, nil
}

// isProcessStream reports the name of a process output stream.
func isProcessStream(stream string) bool {
	return stream == processStreamStdout || stream == processStreamStderr
}

// decodeDebugEvent decodes a debug-event frame.
func decodeDebugEvent(data []byte) (Message, error) {
	var wire struct {
		Level     *harness.DebugLevel      `json:"level"`
		Subsystem *string                  `json:"subsystem"`
		Message   *string                  `json:"message"`
		Attrs     map[string]any           `json:"attrs"`
		Error     *harness.DiagnosticError `json:"error"`
	}

	if err := decodeWire(data, &wire); err != nil {
		return nil, err
	}

	if wire.Level == nil || !isDebugLevel(*wire.Level) || wire.Subsystem == nil || wire.Message == nil {
		return nil, missingField("debug-event", "level, subsystem and message")
	}

	return DebugEvent{
		Level:     *wire.Level,
		Subsystem: *wire.Subsystem,
		Message:   *wire.Message,
		Attrs:     wire.Attrs,
		Error:     wire.Error,
	}, nil
}

// DiagnosticFromFrame normalizes a sandbox-log or debug-event frame into the
// harness.Diagnostic an adapter reports: captured stderr lines become
// warnings and stdout lines information.
func DiagnosticFromFrame(message Message, sessionID string, timestamp time.Time) (harness.Diagnostic, bool) {
	switch typed := message.(type) {
	case SandboxLog:
		level := harness.DebugInfo
		if typed.Stream == "stderr" {
			level = harness.DebugWarn
		}

		return harness.Diagnostic{
			Level:     level,
			Message:   typed.Line,
			Subsystem: "sandbox.log." + typed.Source,
			Kind:      harness.DiagnosticLog,
			Source:    typed.Source,
			Stream:    typed.Stream,
			SessionID: sessionID,
			Timestamp: timestamp,
		}, true
	case DebugEvent:
		return harness.Diagnostic{
			Level:     typed.Level,
			Message:   typed.Message,
			Subsystem: typed.Subsystem,
			Kind:      harness.DiagnosticEvent,
			Attrs:     typed.Attrs,
			Error:     typed.Error,
			SessionID: sessionID,
			Timestamp: timestamp,
		}, true
	default:
		return harness.Diagnostic{}, false
	}
}

// decodeWire decodes a frame into its wire struct.
func decodeWire(data []byte, wire any) error {
	if err := json.Unmarshal(data, wire); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidFrame, err)
	}

	return nil
}

// missingField reports a frame without its required fields.
func missingField(frameType string, fields string) error {
	return fmt.Errorf("%w: %s frame requires %s", ErrInvalidFrame, frameType, fields)
}

// isDebugLevel reports a known debug level.
func isDebugLevel(level harness.DebugLevel) bool {
	switch level {
	case harness.DebugError, harness.DebugWarn, harness.DebugInfo, harness.DebugDebug, harness.DebugTrace:
		return true
	default:
		return false
	}
}

// metadataJSON is the wire form of harness metadata.
type metadataJSON = map[string]any

// toMetadata converts wire metadata.
func toMetadata(value metadataJSON) harness.Metadata {
	if value == nil {
		return nil
	}

	return harness.Metadata(value)
}

// toProviderMetadata converts wire provider metadata.
func toProviderMetadata(value metadataJSON) model.ProviderMetadata {
	if value == nil {
		return nil
	}

	return model.ProviderMetadata(value)
}

// trimmedLine removes a trailing carriage return and surrounding spaces.
func trimmedLine(line string) string {
	return strings.TrimSpace(strings.TrimSuffix(line, "\r"))
}
