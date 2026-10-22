package xai

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

// realtimeSecretProtocol prefixes the client secret subprotocol.
const realtimeSecretProtocol = "xai-client-secret."

var (
	_ model.RealtimeModel               = (*RealtimeModel)(nil)
	_ model.RealtimeClientSecretCreator = (*RealtimeModel)(nil)
	_ model.RealtimeWebSocketConfigurer = (*RealtimeModel)(nil)
	_ model.RealtimeProvider            = (*Provider)(nil)
)

// Realtime returns a realtime voice model for turn-based voice and text
// conversations. Realtime models are experimental.
func (p *Provider) Realtime(modelID string) *RealtimeModel {
	return &RealtimeModel{
		modelID:  modelID,
		provider: p,
	}
}

// RealtimeModel returns a realtime model, as Realtime does.
func (p *Provider) RealtimeModel(modelID string) (model.RealtimeModel, error) {
	return p.Realtime(modelID), nil
}

// RealtimeModel is an xAI realtime voice model. Clients connect with a
// short-lived client secret. The realtime contract is experimental and may
// change.
type RealtimeModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "xai.realtime" by default.
func (m *RealtimeModel) Provider() string {
	return m.provider.name + ".realtime"
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
	ExpiresAfter *realtimeSecretExpiresAfter `json:"expires_after,omitempty"`
}

type realtimeSecretExpiresAfter struct {
	Seconds int64 `json:"seconds"`
}

// realtimeClientSecretResponse is a created client secret.
type realtimeClientSecretResponse struct {
	Value     string `json:"value"`
	ExpiresAt *int64 `json:"expires_at"`
}

// CreateClientSecret creates an ephemeral client secret. xAI does not take
// the session configuration with the secret; send it in a session update
// after connecting.
func (m *RealtimeModel) CreateClientSecret(ctx context.Context, options model.RealtimeClientSecretOptions) (*model.RealtimeClientSecret, error) {
	body := realtimeClientSecretRequest{}

	if options.ExpiresAfter > 0 {
		body.ExpiresAfter = &realtimeSecretExpiresAfter{
			Seconds: int64(options.ExpiresAfter / time.Second),
		}
	}

	request, err := m.provider.apiRequest(realtimeClientSecretsPath, nil)
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

	result := &model.RealtimeClientSecret{
		Token: secret.Value,
		URL:   webSocketURL,
	}

	if secret.ExpiresAt != nil {
		result.ExpiresAt = time.Unix(*secret.ExpiresAt, 0).UTC()
	}

	return result, nil
}

// webSocketURL returns the realtime WebSocket URL on the base URL's host.
// xAI selects the voice model from the model query parameter; without it
// the server silently falls back to its default model.
func (m *RealtimeModel) webSocketURL() (string, error) {
	base, err := url.Parse(m.provider.baseURL)
	if err != nil {
		return "", fmt.Errorf("%w: parse base url: %w", model.ErrInvalidArgument, err)
	}

	return "wss://" + base.Host + "/v1/realtime?model=" + url.QueryEscape(m.modelID), nil
}

// WebSocketConfig carries the client secret in a subprotocol, as browser
// WebSockets cannot send headers.
func (m *RealtimeModel) WebSocketConfig(secret model.RealtimeClientSecret) model.RealtimeWebSocketConfig {
	return model.RealtimeWebSocketConfig{
		URL: secret.URL,
		Protocols: []string{
			realtimeSecretProtocol + secret.Token,
		},
	}
}

// ParseServerEvent maps one xAI realtime event to a normalized event.
func (m *RealtimeModel) ParseServerEvent(raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	event, err := parseRealtimeServerEvent(raw)
	if err != nil {
		return nil, err
	}

	return []model.RealtimeServerEvent{
		event,
	}, nil
}

// SerializeClientEvent maps a normalized event to an xAI realtime event.
// Truncation is dropped, as xAI ignores it; session lifecycle, mute and
// context events have no xAI form.
func (m *RealtimeModel) SerializeClientEvent(event model.RealtimeClientEvent) (*model.RealtimeFrame, error) {
	message, err := serializeRealtimeClientEvent(event)
	if err != nil {
		return nil, err
	}

	return model.NewRealtimeJSONFrame(message), nil
}

// BuildSessionConfig returns the xAI session configuration.
func (m *RealtimeModel) BuildSessionConfig(config model.RealtimeSessionConfig) (json.RawMessage, error) {
	return buildRealtimeSession(config)
}
