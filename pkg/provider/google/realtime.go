package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// realtimeWebSocketPath is the Live API service that accepts ephemeral
	// auth tokens.
	realtimeWebSocketPath = "google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContentConstrained"

	// realtimeDefaultOpenWindow is how long a token can open a session when
	// no lifetime is requested.
	realtimeDefaultOpenWindow = 60 * time.Second

	// realtimeSessionAllowance extends the token lifetime past its open
	// window so that the opened session can keep running.
	realtimeSessionAllowance = 30 * time.Minute

	// realtimeDefaultInputRate is the input audio rate until a session
	// update sets another.
	realtimeDefaultInputRate = 16000

	// isoTimestampLayout formats times like JavaScript's toISOString.
	isoTimestampLayout = "2006-01-02T15:04:05.000Z07:00"
)

var (
	_ model.RealtimeModel                    = (*RealtimeModel)(nil)
	_ model.RealtimeClientSecretCreator      = (*RealtimeModel)(nil)
	_ model.RealtimeWebSocketConfigurer      = (*RealtimeModel)(nil)
	_ model.RealtimeServerEventParserFactory = (*RealtimeModel)(nil)
	_ model.RealtimeProvider                 = (*Provider)(nil)
)

// RealtimeModel is a Gemini Live API model, such as gemini-3.8-live, for
// voice and text conversations. Clients connect with an ephemeral auth token
// created with the API key. The realtime contract is experimental and may
// change.
type RealtimeModel struct {
	modelID  string
	provider *Provider

	// inputAudioRate is the rate that input audio chunks advertise. Session
	// updates set it from their input audio format.
	inputAudioRate atomic.Int64

	// mapper parses events for ParseServerEvent, which shares one turn
	// state; connections use NewServerEventParser instead.
	mapper *realtimeEventMapper
}

// Realtime returns a Gemini Live API model. Realtime models are
// experimental.
func (p *Provider) Realtime(modelID string) *RealtimeModel {
	realtimeModel := &RealtimeModel{
		modelID:  modelID,
		provider: p,
		mapper:   newRealtimeEventMapper(),
	}

	realtimeModel.inputAudioRate.Store(realtimeDefaultInputRate)

	return realtimeModel
}

// RealtimeModel returns Realtime(modelID).
func (p *Provider) RealtimeModel(modelID string) (model.RealtimeModel, error) {
	return p.Realtime(modelID), nil
}

// Provider returns the provider identifier,
// "google.generative-ai.realtime" by default.
func (m *RealtimeModel) Provider() string {
	return m.provider.name + ".realtime"
}

// ModelID returns the model ID.
func (m *RealtimeModel) ModelID() string {
	return m.modelID
}

// Capabilities declares nothing, which keeps the client secret WebSocket
// connection and session update startup.
func (m *RealtimeModel) Capabilities() model.RealtimeCapabilities {
	return model.RealtimeCapabilities{}
}

// authTokenRequest creates an ephemeral auth token. Uses is always sent:
// zero places no limit on the sessions a token starts, which WebSocket
// reconnects within a session need.
type authTokenRequest struct {
	Uses                     int             `json:"uses"`
	ExpireTime               string          `json:"expireTime"`
	NewSessionExpireTime     string          `json:"newSessionExpireTime"`
	BidiGenerateContentSetup json.RawMessage `json:"bidiGenerateContentSetup"`
}

type authTokenResponse struct {
	Name       string  `json:"name"`
	ExpireTime *string `json:"expireTime"`
}

// CreateClientSecret creates an ephemeral auth token that carries the
// session configuration. ExpiresAfter bounds how long the token can open a
// session, 60 seconds by default; the token itself lives 30 minutes longer.
// The API key is required: it authenticates the request as a query
// parameter.
func (m *RealtimeModel) CreateClientSecret(ctx context.Context, options model.RealtimeClientSecretOptions) (*model.RealtimeClientSecret, error) {
	apiKey, err := m.provider.liveAPIKey(ctx, nil, "realtime token creation")
	if err != nil {
		return nil, err
	}

	setup, err := buildRealtimeSetup(options.SessionConfig, m.modelID)
	if err != nil {
		return nil, err
	}

	openWindow := options.ExpiresAfter
	if openWindow <= 0 {
		openWindow = realtimeDefaultOpenWindow
	}

	now := time.Now().UTC()

	baseURL, err := m.provider.currentBaseURL()
	if err != nil {
		return nil, err
	}

	tokensURL, err := realtimeBaseURL(baseURL)
	if err != nil {
		return nil, err
	}

	tokensURL.Path = strings.TrimSuffix(tokensURL.Path, "/") + "/v1alpha/auth_tokens"
	tokensURL.RawQuery = url.Values{
		"key": {
			apiKey,
		},
	}.Encode()

	request := apicall.Request{
		Client:     m.provider.httpClient,
		URL:        tokensURL.String(),
		ParseError: parseErrorMessage,
	}

	response, err := apicall.PostJSON(ctx, request, authTokenRequest{
		Uses:                     0,
		ExpireTime:               now.Add(openWindow + realtimeSessionAllowance).Format(isoTimestampLayout),
		NewSessionExpireTime:     now.Add(openWindow).Format(isoTimestampLayout),
		BidiGenerateContentSetup: setup,
	})
	if err != nil {
		return nil, err
	}

	var token authTokenResponse
	if err := apicall.DecodeJSON(response.Body, &token); err != nil {
		return nil, err
	}

	if token.Name == "" {
		return nil, fmt.Errorf("%w: realtime auth token has no name", model.ErrInvalidResponse)
	}

	webSocketURL, err := realtimeWebSocketURL(baseURL, realtimeWebSocketPath)
	if err != nil {
		return nil, err
	}

	return &model.RealtimeClientSecret{
		Token:     token.Name,
		URL:       webSocketURL.String(),
		ExpiresAt: tokenExpiry(token.ExpireTime),
	}, nil
}

// tokenExpiry parses the reported expiration time, truncated to seconds.
func tokenExpiry(expireTime *string) time.Time {
	if expireTime == nil {
		return time.Time{}
	}

	parsed, err := time.Parse(time.RFC3339Nano, *expireTime)
	if err != nil {
		return time.Time{}
	}

	return parsed.Truncate(time.Second)
}

// WebSocketConfig carries the auth token in the access_token query
// parameter.
func (m *RealtimeModel) WebSocketConfig(secret model.RealtimeClientSecret) model.RealtimeWebSocketConfig {
	return model.RealtimeWebSocketConfig{
		URL: secret.URL + "?access_token=" + url.QueryEscape(secret.Token),
	}
}

// NewServerEventParser returns a parser with its own turn state, which
// synthesizes the response and item IDs that Live API messages lack.
func (m *RealtimeModel) NewServerEventParser() model.RealtimeServerEventParser {
	return newRealtimeEventMapper().parse
}

// ParseServerEvent maps one Live API message with the model's shared turn
// state. Prefer NewServerEventParser, which keeps state per connection.
func (m *RealtimeModel) ParseServerEvent(raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	return m.mapper.parse(raw)
}

// SerializeClientEvent maps a normalized event to a Live API message. Turn
// commands the Live API handles itself, such as response creation, and
// session lifecycle events have no Live API form and send nothing.
func (m *RealtimeModel) SerializeClientEvent(event model.RealtimeClientEvent) (*model.RealtimeFrame, error) {
	if update, ok := event.(model.RealtimeSessionUpdate); ok && update.Config.InputAudioFormat != nil && update.Config.InputAudioFormat.Rate != 0 {
		m.inputAudioRate.Store(int64(update.Config.InputAudioFormat.Rate))
	}

	message, err := serializeRealtimeClientEvent(event, m.modelID, m.inputAudioRate.Load())
	if err != nil {
		return nil, err
	}

	return model.NewRealtimeJSONFrame(message), nil
}

// BuildSessionConfig returns the Live API setup of a session.
func (m *RealtimeModel) BuildSessionConfig(config model.RealtimeSessionConfig) (json.RawMessage, error) {
	return buildRealtimeSetup(&config, m.modelID)
}

// liveAPIKey returns the API key of a Live API call, which authenticates as
// a query parameter. The call headers may replace the provider's key.
func (p *Provider) liveAPIKey(ctx context.Context, callHeaders http.Header, purpose string) (string, error) {
	request, err := p.request(ctx, "", callHeaders)
	if err != nil {
		return "", err
	}

	return liveAPIKeyOf(request.Headers, purpose)
}

// liveAPIKeyOf returns the API key header of a Live API call. purpose names
// the call when the key is missing.
func liveAPIKeyOf(headers http.Header, purpose string) (string, error) {
	apiKey := headers.Get(apiKeyHeader)
	if apiKey == "" {
		return "", fmt.Errorf("%w: Google Generative AI API key is required for %s", model.ErrLoadAPIKey, purpose)
	}

	return apiKey, nil
}

// realtimeBaseURL returns the base URL of Live API endpoints, without a
// trailing v1beta or v1alpha segment: Live API paths carry their own
// version.
func realtimeBaseURL(baseURL string) (*url.URL, error) {
	target, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("%w: parse google base url: %w", model.ErrInvalidArgument, err)
	}

	segments := strings.Split(target.Path, "/")
	if last := segments[len(segments)-1]; last == "v1beta" || last == "v1alpha" {
		target.Path = strings.Join(segments[:len(segments)-1], "/")
		if target.Path == "" {
			target.Path = "/"
		}

		target.RawPath = ""
	}

	return target, nil
}

// realtimeWebSocketURL returns the ws or wss URL of a Live API service.
func realtimeWebSocketURL(baseURL string, servicePath string) (*url.URL, error) {
	target, err := realtimeBaseURL(baseURL)
	if err != nil {
		return nil, err
	}

	if target.Scheme == "https" {
		target.Scheme = "wss"
	} else {
		target.Scheme = "ws"
	}

	target.Path = strings.TrimSuffix(target.Path, "/") + "/ws/" + servicePath
	target.RawPath = ""

	return target, nil
}
