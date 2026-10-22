package model

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
)

// RealtimeModelKind names realtime models in errors and registries.
const RealtimeModelKind Kind = "realtime model"

// RealtimeModel converses with audio and text over a bidirectional
// connection. It is a codec: it maps normalized client events to the
// provider's messages and the provider's messages to normalized server
// events, while the caller owns the connection. The realtime contract is
// experimental and may change.
//
// Connection setup is optional: models implement
// RealtimeServerWebSocketConfigurer, RealtimeClientSecretCreator with
// RealtimeWebSocketConfigurer, or RealtimeWebRTCSessionCreator with
// RealtimeWebRTCConfigurer, according to the connections they support.
type RealtimeModel interface {
	// Provider returns the provider identifier, such as "openai.realtime".
	Provider() string

	ModelID() string

	// Capabilities declares the conversation semantics and the supported
	// connections. The zero value declares nothing, which selects the
	// legacy behavior described on each field.
	Capabilities() RealtimeCapabilities

	// ParseServerEvent maps one JSON message received from the provider to
	// one or more normalized events.
	ParseServerEvent(raw json.RawMessage) ([]RealtimeServerEvent, error)

	// SerializeClientEvent maps a normalized event to the frame sent to the
	// provider. A nil frame means nothing is sent.
	SerializeClientEvent(event RealtimeClientEvent) (*RealtimeFrame, error)

	// BuildSessionConfig returns the provider's session configuration
	// payload for a normalized configuration.
	BuildSessionConfig(config RealtimeSessionConfig) (json.RawMessage, error)
}

// RealtimeFrameType is the WebSocket message type of a realtime frame.
type RealtimeFrameType string

// Realtime frame types.
const (
	// RealtimeTextFrame carries UTF-8 text, such as a JSON event or a plain
	// command.
	RealtimeTextFrame RealtimeFrameType = "text"

	// RealtimeBinaryFrame carries raw bytes, such as audio.
	RealtimeBinaryFrame RealtimeFrameType = "binary"
)

// RealtimeFrame is one message sent to a realtime provider, mirroring the
// string, binary or JSON values upstream codecs return.
type RealtimeFrame struct {
	Type RealtimeFrameType
	Data []byte
}

// NewRealtimeJSONFrame returns a text frame carrying a JSON message, or nil
// when the message is nil.
func NewRealtimeJSONFrame(message json.RawMessage) *RealtimeFrame {
	if message == nil {
		return nil
	}

	return &RealtimeFrame{
		Type: RealtimeTextFrame,
		Data: message,
	}
}

// NewRealtimeTextFrame returns a text frame carrying plain text.
func NewRealtimeTextFrame(text string) *RealtimeFrame {
	return &RealtimeFrame{
		Type: RealtimeTextFrame,
		Data: []byte(text),
	}
}

// NewRealtimeBinaryFrame returns a binary frame carrying data.
func NewRealtimeBinaryFrame(data []byte) *RealtimeFrame {
	return &RealtimeFrame{
		Type: RealtimeBinaryFrame,
		Data: data,
	}
}

// RealtimeProvider is implemented by providers that offer realtime models.
// It is separate from Provider because most providers have none.
type RealtimeProvider interface {
	RealtimeModel(modelID string) (RealtimeModel, error)
}

// RealtimeServerEventParser parses the messages of one connection. It may
// keep state between messages.
type RealtimeServerEventParser func(raw json.RawMessage) ([]RealtimeServerEvent, error)

// RealtimeServerEventParserFactory is implemented by models whose parsing
// keeps state per connection. Callers create one parser per connection and
// discard it on disconnect instead of calling ParseServerEvent.
type RealtimeServerEventParserFactory interface {
	NewServerEventParser() RealtimeServerEventParser
}

// RealtimeHealthChecker is implemented by models with a keepalive protocol,
// such as ping and pong messages.
type RealtimeHealthChecker interface {
	// HealthCheckResponse returns the message to send back in reply to a
	// received message, and reports false when no reply is needed. It is
	// called before ParseServerEvent.
	HealthCheckResponse(raw json.RawMessage) (json.RawMessage, bool)
}

// RealtimeServerWebSocketConfigurer is implemented by models that accept
// WebSocket connections from a server holding long-lived credentials.
type RealtimeServerWebSocketConfigurer interface {
	// ServerWebSocketConfig returns the URL and the headers to connect with.
	// The headers can contain credentials and must not reach clients.
	ServerWebSocketConfig(ctx context.Context) (*RealtimeServerWebSocketConfig, error)
}

// RealtimeServerWebSocketConfig is the connection setup of a server
// WebSocket.
type RealtimeServerWebSocketConfig struct {
	URL     string
	Headers http.Header
}

// RealtimeClientSecretCreator is implemented by models that issue
// short-lived client secrets, which are safe to hand to clients that
// connect with RealtimeWebSocketConfigurer.
type RealtimeClientSecretCreator interface {
	CreateClientSecret(ctx context.Context, options RealtimeClientSecretOptions) (*RealtimeClientSecret, error)
}

// RealtimeClientSecretOptions are the inputs of CreateClientSecret.
type RealtimeClientSecretOptions struct {
	// ExpiresAfter is the lifetime of the secret, or zero for the provider
	// default.
	ExpiresAfter time.Duration

	// SessionConfig is embedded in the secret request. Some providers
	// require the full configuration when the secret is created.
	SessionConfig *RealtimeSessionConfig
}

// RealtimeClientSecret is a short-lived credential for one connection.
type RealtimeClientSecret struct {
	// Token authenticates the connection.
	Token string

	// URL is the WebSocket URL, including provider-specific query
	// parameters such as the model ID.
	URL string

	// ExpiresAt is zero when the provider did not report it.
	ExpiresAt time.Time
}

// RealtimeWebSocketConfigurer is implemented by models that clients connect
// to with a client secret.
type RealtimeWebSocketConfigurer interface {
	// WebSocketConfig returns the URL and subprotocols that carry the
	// client secret.
	WebSocketConfig(secret RealtimeClientSecret) RealtimeWebSocketConfig
}

// RealtimeWebSocketConfig is the connection setup of a client WebSocket.
type RealtimeWebSocketConfig struct {
	URL       string
	Protocols []string
}

// RealtimeWebRTCConfigurer is implemented by models that support WebRTC
// connections.
type RealtimeWebRTCConfigurer interface {
	WebRTCConfig() RealtimeWebRTCConfig
}

// RealtimeWebRTCConfig is the client setup of a WebRTC event channel.
type RealtimeWebRTCConfig struct {
	DataChannelLabel string
}

// RealtimeWebRTCSessionCreator is implemented by models whose WebRTC
// sessions start with a server-side SDP exchange.
type RealtimeWebRTCSessionCreator interface {
	// CreateWebRTCSession exchanges the client's SDP offer for the
	// provider's answer. Only the answer and the session ID should reach
	// the client.
	CreateWebRTCSession(ctx context.Context, options RealtimeWebRTCSessionOptions) (*RealtimeWebRTCSession, error)
}

// RealtimeWebRTCSessionOptions are the inputs of CreateWebRTCSession.
type RealtimeWebRTCSessionOptions struct {
	// SDP is the client's offer.
	SDP string

	SessionConfig RealtimeSessionConfig
}

// RealtimeWebRTCSession is a WebRTC session created by the provider.
type RealtimeWebRTCSession struct {
	SessionID string

	// SDP is the provider's answer.
	SDP string
}

// RealtimeCapabilities declares the behavior of a realtime model. Empty
// fields were not declared.
type RealtimeCapabilities struct {
	// Conversation is the conversation semantics.
	Conversation RealtimeConversation

	// Transports are the supported transports. Nil declares none.
	Transports []RealtimeTransport

	// Connections are the supported connection kinds. Nil keeps the legacy
	// client secret WebSocket connection.
	Connections []RealtimeConnection

	// Startup is the event that starts a session. Empty starts it with a
	// session update.
	Startup RealtimeStartup

	// Finalization is how a session ends. Empty finalizes it when the
	// transport closes.
	Finalization RealtimeFinalization
}

// RealtimeConversation is the conversation semantics of a realtime model.
type RealtimeConversation string

// Realtime conversation semantics.
const (
	// RealtimeContinuous streams audio both ways at all times.
	RealtimeContinuous RealtimeConversation = "continuous"

	// RealtimeTurnBased alternates user and model turns.
	RealtimeTurnBased RealtimeConversation = "turn-based"
)

// RealtimeTransport is a realtime transport.
type RealtimeTransport string

// Realtime transports.
const (
	RealtimeWebSocketTransport RealtimeTransport = "websocket"
	RealtimeWebRTCTransport    RealtimeTransport = "webrtc"
)

// RealtimeConnection is a kind of realtime connection.
type RealtimeConnection string

// Realtime connection kinds.
const (
	RealtimeClientSecretWebSocket RealtimeConnection = "client-secret-websocket"
	RealtimeServerWebSocket       RealtimeConnection = "server-websocket"
	RealtimeWebRTC                RealtimeConnection = "webrtc"
)

// RealtimeStartup is the client event that starts a session.
type RealtimeStartup string

// Realtime startup events.
const (
	RealtimeStartupSessionStart  RealtimeStartup = "session-start"
	RealtimeStartupSessionUpdate RealtimeStartup = "session-update"
)

// RealtimeFinalization is how a realtime session ends.
type RealtimeFinalization string

// Realtime finalizations.
const (
	// RealtimeFinalizeSessionClose ends the session with a session close
	// event that the provider acknowledges with the final usage.
	RealtimeFinalizeSessionClose RealtimeFinalization = "session-close"

	// RealtimeFinalizeTransportClose ends the session when the transport
	// closes.
	RealtimeFinalizeTransportClose RealtimeFinalization = "transport-close"
)

// RealtimeSessionConfig is the provider-neutral configuration of a realtime
// session. Empty fields leave the provider default.
type RealtimeSessionConfig struct {
	// Instructions are the system instructions.
	Instructions string

	// Voice is the voice of the audio output.
	Voice string

	// OutputModalities selects text output, audio output or both.
	OutputModalities []RealtimeModality

	InputAudioFormat  *AudioFormat
	OutputAudioFormat *AudioFormat

	// InputAudioTranscription enables transcripts of the user's audio,
	// reported as input transcription events.
	InputAudioTranscription *RealtimeTranscriptionConfig

	// OutputAudioTranscription makes transcripts of the model's audio
	// explicit, reported as audio transcript events.
	OutputAudioTranscription *RealtimeTranscriptionConfig

	// TurnDetection configures voice activity detection.
	TurnDetection *RealtimeTurnDetection

	// Tools are the functions the model may call.
	Tools []RealtimeToolDefinition

	ProviderOptions ProviderOptions
}

// RealtimeModality is an output modality.
type RealtimeModality string

// Realtime output modalities.
const (
	RealtimeText  RealtimeModality = "text"
	RealtimeAudio RealtimeModality = "audio"
)

// RealtimeTranscriptionConfig configures the transcription of input or
// output audio. Empty fields leave the provider default.
type RealtimeTranscriptionConfig struct {
	// Model is the provider-specific transcription model.
	Model string

	// Language is a hint for the language of the audio.
	Language string

	// Prompt guides the transcription.
	Prompt string
}

// RealtimeTurnDetectionType selects how turns are detected.
type RealtimeTurnDetectionType string

// Turn detection types.
const (
	// RealtimeServerVAD detects turns by voice activity on the server.
	RealtimeServerVAD RealtimeTurnDetectionType = "server-vad"

	// RealtimeSemanticVAD detects turns from the meaning of speech.
	RealtimeSemanticVAD RealtimeTurnDetectionType = "semantic-vad"

	// RealtimeTurnDetectionDisabled turns detection off, for push-to-talk.
	RealtimeTurnDetectionDisabled RealtimeTurnDetectionType = "disabled"
)

// RealtimeTurnDetection configures voice activity detection. Nil fields
// leave the provider default.
type RealtimeTurnDetection struct {
	Type RealtimeTurnDetectionType

	// Threshold is the activation threshold between 0 and 1. Higher values
	// require louder audio.
	Threshold *float64

	// SilenceDuration is the silence that ends a turn.
	SilenceDuration *time.Duration

	// PrefixPadding is the audio kept before the detected start of speech.
	PrefixPadding *time.Duration
}

// RealtimeToolDefinition is a function the model may call during a
// session.
type RealtimeToolDefinition struct {
	// Name is unique within the session.
	Name string

	// Description tells the model when to call the function.
	Description string

	// Parameters is the JSON Schema of the arguments.
	Parameters json.RawMessage
}
