package openai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// liveSessionsPath creates Live API sessions.
	liveSessionsPath = "/live/sessions"

	// liveDataChannelLabel is the WebRTC data channel of Live API events.
	liveDataChannelLabel = "oai-events"
)

// LiveModel is an OpenAI Live API model, such as gpt-live-1, for continuous
// conversations: audio streams both ways at all times, and the model
// delegates work to the client. Servers connect over a WebSocket with the
// API key; browsers connect over WebRTC after a server-side SDP exchange.
// The realtime contract is experimental and may change.
type LiveModel struct {
	modelID string
	config  config
}

func newLiveModel(modelID string, c config) *LiveModel {
	return &LiveModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as "openai.live".
func (m *LiveModel) Provider() string {
	return m.config.modelProvider("live")
}

// ModelID returns the model identifier.
func (m *LiveModel) ModelID() string {
	return m.modelID
}

// Capabilities declares continuous conversations over server WebSockets
// and WebRTC, started with a session start event and finalized with a
// session close event.
func (m *LiveModel) Capabilities() model.RealtimeCapabilities {
	return model.RealtimeCapabilities{
		Conversation: model.RealtimeContinuous,
		Transports: []model.RealtimeTransport{
			model.RealtimeWebSocketTransport,
			model.RealtimeWebRTCTransport,
		},
		Connections: []model.RealtimeConnection{
			model.RealtimeServerWebSocket,
			model.RealtimeWebRTC,
		},
		Startup:      model.RealtimeStartupSessionStart,
		Finalization: model.RealtimeFinalizeSessionClose,
	}
}

// WebRTCConfig returns the label of the events data channel.
func (m *LiveModel) WebRTCConfig() model.RealtimeWebRTCConfig {
	return model.RealtimeWebRTCConfig{
		DataChannelLabel: liveDataChannelLabel,
	}
}

// ServerWebSocketConfig returns the sessions WebSocket URL with the
// provider's authentication headers, which must stay on the server.
func (m *LiveModel) ServerWebSocketConfig(ctx context.Context) (*model.RealtimeServerWebSocketConfig, error) {
	request, err := m.config.request(ctx, liveSessionsPath, nil)
	if err != nil {
		return nil, err
	}

	target, err := livetranscription.WebSocketURL(request.URL)
	if err != nil {
		return nil, err
	}

	return &model.RealtimeServerWebSocketConfig{
		URL:     target.String(),
		Headers: request.Headers,
	}, nil
}

// liveWebRTCRequest is the body of a WebRTC session creation.
type liveWebRTCRequest struct {
	Session   json.RawMessage `json:"session"`
	Transport liveTransport   `json:"transport"`
}

// liveTransport is the WebRTC transport of a session.
type liveTransport struct {
	Type string `json:"type"`
	SDP  string `json:"sdp"`
}

// liveWebRTCResponse is a created WebRTC session.
type liveWebRTCResponse struct {
	Session struct {
		ID string `json:"id"`
	} `json:"session"`

	Transport liveTransport `json:"transport"`
}

// CreateWebRTCSession exchanges the client's SDP offer for the API's
// answer.
func (m *LiveModel) CreateWebRTCSession(ctx context.Context, options model.RealtimeWebRTCSessionOptions) (*model.RealtimeWebRTCSession, error) {
	if options.SDP == "" {
		return nil, fmt.Errorf("%w: the WebRTC session requires an SDP offer", model.ErrInvalidArgument)
	}

	session, err := buildLiveSession(options.SessionConfig, m.modelID, liveWebRTC)
	if err != nil {
		return nil, err
	}

	request, err := m.config.request(ctx, liveSessionsPath, nil)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, liveWebRTCRequest{
		Session: session,
		Transport: liveTransport{
			Type: "webrtc",
			SDP:  options.SDP,
		},
	})
	if err != nil {
		return nil, err
	}

	var created liveWebRTCResponse
	if err := apicall.DecodeJSON(response.Body, &created); err != nil {
		return nil, err
	}

	if created.Session.ID == "" || created.Transport.Type != "webrtc" || created.Transport.SDP == "" {
		return nil, fmt.Errorf("%w: the WebRTC session response has no session ID or SDP answer", model.ErrInvalidResponse)
	}

	return &model.RealtimeWebRTCSession{
		SessionID: created.Session.ID,
		SDP:       created.Transport.SDP,
	}, nil
}

// NewServerEventParser returns the stateless event parser.
func (m *LiveModel) NewServerEventParser() model.RealtimeServerEventParser {
	return m.ParseServerEvent
}

// ParseServerEvent maps one Live API event. Malformed known events become
// error events with the code "invalid_server_event".
func (m *LiveModel) ParseServerEvent(raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	return []model.RealtimeServerEvent{
		parseLiveServerEvent(raw),
	}, nil
}

// SerializeClientEvent maps a normalized event to a Live API event. Live
// sessions are configured once at startup and take context instead of
// voice-turn commands, so session updates and turn commands are rejected.
func (m *LiveModel) SerializeClientEvent(event model.RealtimeClientEvent) (*model.RealtimeFrame, error) {
	message, err := serializeLiveClientEvent(event, m.modelID)
	if err != nil {
		return nil, err
	}

	return model.NewRealtimeJSONFrame(message), nil
}

// BuildSessionConfig returns the Live API session configuration for a
// WebSocket connection.
func (m *LiveModel) BuildSessionConfig(config model.RealtimeSessionConfig) (json.RawMessage, error) {
	return buildLiveSession(config, m.modelID, liveWebSocket)
}
