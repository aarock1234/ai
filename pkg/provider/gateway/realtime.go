package gateway

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// realtimeModelPath is the realtime WebSocket endpoint.
	realtimeModelPath = "/realtime-model"

	// realtimeClientSecretsPath mints client secrets. It lives at the
	// gateway origin, not under the base URL's path.
	realtimeClientSecretsPath = "/v1/realtime/client-secrets"

	// realtimeSubprotocol lets the gateway echo a negotiated subprotocol.
	realtimeSubprotocol = "ai-gateway-realtime.v1"
)

var (
	_ model.RealtimeModel               = (*RealtimeModel)(nil)
	_ model.RealtimeClientSecretCreator = (*RealtimeModel)(nil)
	_ model.RealtimeWebSocketConfigurer = (*RealtimeModel)(nil)
	_ model.RealtimeProvider            = (*Provider)(nil)
)

// Realtime returns a realtime model, such as "openai/gpt-realtime-2".
// Realtime models are experimental.
func (p *Provider) Realtime(modelID string) *RealtimeModel {
	return &RealtimeModel{
		modelID:  modelID,
		provider: p,
	}
}

// RealtimeModel returns Realtime(modelID).
func (p *Provider) RealtimeModel(modelID string) (model.RealtimeModel, error) {
	return p.Realtime(modelID), nil
}

// RealtimeModel is an AI Gateway realtime model. The gateway speaks the
// normalized realtime protocol and translates it for the model's provider,
// so the model encodes and decodes the normalized events as they are; only
// the connection and its authentication are gateway-specific. The realtime
// contract is experimental and may change.
type RealtimeModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "gateway.realtime" by default.
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
	Model     string `json:"model"`
	ExpiresIn *int64 `json:"expiresIn,omitempty"`
}

// realtimeClientSecretResponse is a minted client secret. ExpiresAt is in
// Unix seconds.
type realtimeClientSecretResponse struct {
	Token     string   `json:"token"`
	ExpiresAt *float64 `json:"expiresAt"`
}

// CreateClientSecret mints a single-use, short-lived client secret with the
// gateway credential. Call it on a server and hand only the secret to the
// client. The session configuration is not part of the secret; send it in a
// session update after connecting.
func (m *RealtimeModel) CreateClientSecret(ctx context.Context, options model.RealtimeClientSecretOptions) (*model.RealtimeClientSecret, error) {
	body := realtimeClientSecretRequest{
		Model: m.modelID,
	}

	if options.ExpiresAfter > 0 {
		body.ExpiresIn = new(int64(options.ExpiresAfter / time.Second))
	}

	origin, err := m.provider.origin()
	if err != nil {
		return nil, err
	}

	request, method, err := m.provider.accountRequest(origin.String() + realtimeClientSecretsPath)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var secret realtimeClientSecretResponse
	if err := apicall.DecodeJSON(response.Body, &secret); err != nil {
		return nil, err
	}

	if secret.Token == "" {
		return nil, fmt.Errorf("%w: realtime client secret has no token", model.ErrInvalidResponse)
	}

	webSocketURL, err := m.webSocketURL()
	if err != nil {
		return nil, err
	}

	result := &model.RealtimeClientSecret{
		Token: secret.Token,
		URL:   webSocketURL,
	}

	if secret.ExpiresAt != nil {
		result.ExpiresAt = time.Unix(int64(*secret.ExpiresAt), 0).UTC()
	}

	return result, nil
}

// webSocketURL returns the realtime WebSocket URL. The model ID rides the
// ai-model-id query parameter, as browser WebSockets cannot send headers.
func (m *RealtimeModel) webSocketURL() (string, error) {
	endpoint := m.provider.baseURL + realtimeModelPath
	if rest, isHTTP := strings.CutPrefix(endpoint, "http"); isHTTP {
		endpoint = "ws" + rest
	}

	target, err := url.Parse(endpoint)
	if err != nil {
		return "", fmt.Errorf("%w: gateway realtime url: %w", model.ErrInvalidArgument, err)
	}

	query := target.Query()
	query.Set("ai-model-id", m.modelID)
	target.RawQuery = query.Encode()

	return target.String(), nil
}

// WebSocketConfig carries the client secret, and the team when one is
// configured, in subprotocols.
func (m *RealtimeModel) WebSocketConfig(secret model.RealtimeClientSecret) model.RealtimeWebSocketConfig {
	protocols := []string{
		realtimeSubprotocol,
		authSubprotocolPrefix + secret.Token,
	}

	if m.provider.teamIDOrSlug != "" {
		protocols = append(protocols, teamSubprotocolPrefix+base64.RawURLEncoding.EncodeToString([]byte(m.provider.teamIDOrSlug)))
	}

	return model.RealtimeWebSocketConfig{
		URL:       secret.URL,
		Protocols: protocols,
	}
}

// ParseServerEvent decodes one normalized server event.
func (m *RealtimeModel) ParseServerEvent(raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	event, err := decodeRealtimeServerEvent(raw)
	if err != nil {
		return nil, err
	}

	return []model.RealtimeServerEvent{
		event,
	}, nil
}

// SerializeClientEvent encodes a normalized client event.
func (m *RealtimeModel) SerializeClientEvent(event model.RealtimeClientEvent) (*model.RealtimeFrame, error) {
	message, err := encodeRealtimeClientEvent(event)
	if err != nil {
		return nil, err
	}

	return model.NewRealtimeJSONFrame(message), nil
}

// BuildSessionConfig encodes the normalized session configuration, which
// the gateway maps to the provider's session.
func (m *RealtimeModel) BuildSessionConfig(config model.RealtimeSessionConfig) (json.RawMessage, error) {
	return json.Marshal(encodeRealtimeSessionConfig(config))
}
