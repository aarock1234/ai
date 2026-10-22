package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// realtimeClientSecretsPath creates ephemeral client secrets.
const realtimeClientSecretsPath = "/realtime/client_secrets"

// liveModelIDs are the models RealtimeModel routes to the Live API.
var liveModelIDs = []string{
	"gpt-live-1",
}

// RealtimeModel is an OpenAI Realtime API model, such as gpt-realtime, for
// turn-based voice and text conversations. Clients connect with a
// short-lived client secret. The realtime contract is experimental and may
// change.
type RealtimeModel struct {
	modelID string
	config  config
}

func newRealtimeModel(modelID string, c config) *RealtimeModel {
	return &RealtimeModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as "openai.realtime".
func (m *RealtimeModel) Provider() string {
	return m.config.modelProvider("realtime")
}

// ModelID returns the model identifier.
func (m *RealtimeModel) ModelID() string {
	return m.modelID
}

// Capabilities declares nothing, which keeps the client secret WebSocket
// connection and session update startup.
func (m *RealtimeModel) Capabilities() model.RealtimeCapabilities {
	return model.RealtimeCapabilities{}
}

// realtimeClientSecretRequest is the body of a client secret request.
type realtimeClientSecretRequest struct {
	Session      json.RawMessage             `json:"session"`
	ExpiresAfter *realtimeSecretExpiresAfter `json:"expires_after,omitempty"`
}

// realtimeSecretExpiresAfter is the lifetime of a client secret. The
// anchor is required by the endpoint.
type realtimeSecretExpiresAfter struct {
	Anchor  string `json:"anchor"`
	Seconds int64  `json:"seconds"`
}

// realtimeClientSecretResponse is a created client secret.
type realtimeClientSecretResponse struct {
	Value     string `json:"value"`
	ExpiresAt *int64 `json:"expires_at"`
}

// CreateClientSecret creates an ephemeral client secret, configured with
// the session configuration when one is given.
func (m *RealtimeModel) CreateClientSecret(ctx context.Context, options model.RealtimeClientSecretOptions) (*model.RealtimeClientSecret, error) {
	session, err := m.clientSecretSession(options.SessionConfig)
	if err != nil {
		return nil, err
	}

	body := realtimeClientSecretRequest{
		Session: session,
	}

	if options.ExpiresAfter > 0 {
		body.ExpiresAfter = &realtimeSecretExpiresAfter{
			Anchor:  "created_at",
			Seconds: int64(options.ExpiresAfter / time.Second),
		}
	}

	request, err := m.config.request(ctx, realtimeClientSecretsPath, nil)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var secret realtimeClientSecretResponse
	if err := apicall.DecodeJSON(response.Body, &secret); err != nil {
		return nil, err
	}

	if secret.Value == "" {
		return nil, fmt.Errorf("%w: realtime client secret has no value", model.ErrInvalidResponse)
	}

	webSocketURL, err := m.webSocketURL()
	if err != nil {
		return nil, err
	}

	return &model.RealtimeClientSecret{
		Token:     secret.Value,
		URL:       webSocketURL,
		ExpiresAt: unixTime(secret.ExpiresAt),
	}, nil
}

// clientSecretSession returns the session of a client secret request: the
// full configuration, or only the type and the model.
func (m *RealtimeModel) clientSecretSession(config *model.RealtimeSessionConfig) (json.RawMessage, error) {
	if config != nil {
		return m.BuildSessionConfig(*config)
	}

	return json.Marshal(realtimeModelSession{
		Type:  "realtime",
		Model: m.modelID,
	})
}

// webSocketURL returns the Realtime API WebSocket URL on the base URL's
// host.
func (m *RealtimeModel) webSocketURL() (string, error) {
	base, err := url.Parse(m.config.baseURL)
	if err != nil {
		return "", fmt.Errorf("%w: parse base url: %w", model.ErrInvalidArgument, err)
	}

	return "wss://" + base.Host + "/v1/realtime?model=" + url.QueryEscape(m.modelID), nil
}

// WebSocketConfig carries the client secret in the API key subprotocol, as
// browser WebSockets cannot send headers.
func (m *RealtimeModel) WebSocketConfig(secret model.RealtimeClientSecret) model.RealtimeWebSocketConfig {
	return model.RealtimeWebSocketConfig{
		URL: secret.URL,
		Protocols: []string{
			realtimeProtocol,
			realtimeAPIKeyProtocol + secret.Token,
		},
	}
}

// ParseServerEvent maps one Realtime API event to a normalized event.
func (m *RealtimeModel) ParseServerEvent(raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	event, err := parseRealtimeServerEvent(raw)
	if err != nil {
		return nil, err
	}

	return []model.RealtimeServerEvent{
		event,
	}, nil
}

// SerializeClientEvent maps a normalized event to a Realtime API event.
// Session lifecycle, mute and context events have no Realtime API form.
func (m *RealtimeModel) SerializeClientEvent(event model.RealtimeClientEvent) (*model.RealtimeFrame, error) {
	message, err := m.serializeClientEvent(event)
	if err != nil {
		return nil, err
	}

	return model.NewRealtimeJSONFrame(message), nil
}

// BuildSessionConfig returns the Realtime API session configuration.
func (m *RealtimeModel) BuildSessionConfig(config model.RealtimeSessionConfig) (json.RawMessage, error) {
	return buildRealtimeSession(config, m.modelID)
}
