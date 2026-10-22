package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"sync/atomic"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// maxRealtimeSecretLifetime bounds the lifetime of a realtime client secret.
const maxRealtimeSecretLifetime = time.Hour

var (
	_ model.RealtimeModel                    = (*RealtimeModel)(nil)
	_ model.RealtimeClientSecretCreator      = (*RealtimeModel)(nil)
	_ model.RealtimeWebSocketConfigurer      = (*RealtimeModel)(nil)
	_ model.RealtimeServerEventParserFactory = (*RealtimeModel)(nil)
	_ model.RealtimeProvider                 = (*Provider)(nil)
)

// RealtimeModel is a realtime Ink speech-to-text model, such as "ink-2",
// over Cartesia's streaming transcription WebSockets. Audio is sent as
// binary frames; with turn detection disabled, committing the input buffer
// sends the plain "finalize" command. The realtime contract is experimental
// and may change.
type RealtimeModel struct {
	modelID  string
	provider *Provider

	// turnDetection is whether the connection uses the turn detection
	// WebSocket. WebSocketConfig sets it from the client secret URL.
	turnDetection atomic.Bool

	// parser parses events for ParseServerEvent, which shares one
	// transcript state; connections use NewServerEventParser instead.
	parser *realtimeEventParser
}

// Realtime returns a realtime Ink model. Realtime models are experimental.
func (p *Provider) Realtime(modelID string) *RealtimeModel {
	realtimeModel := &RealtimeModel{
		modelID:  modelID,
		provider: p,
		parser:   newRealtimeEventParser(),
	}

	realtimeModel.turnDetection.Store(true)

	return realtimeModel
}

// RealtimeModel returns Realtime(modelID).
func (p *Provider) RealtimeModel(modelID string) (model.RealtimeModel, error) {
	return p.Realtime(modelID), nil
}

// Provider returns the provider identifier, "cartesia.realtime" by default.
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

// CreateClientSecret creates a short-lived access token and the WebSocket URL
// it opens. ExpiresAfter is whole seconds up to an hour, or zero for
// Cartesia's default. The session configuration selects the endpoint: turn
// detection is used unless it is disabled. Its input audio format sets the
// encoding and sample rate, and its transcription language must be English.
func (m *RealtimeModel) CreateClientSecret(ctx context.Context, options model.RealtimeClientSecretOptions) (*model.RealtimeClientSecret, error) {
	expiresIn, err := realtimeSecretLifetime(options.ExpiresAfter)
	if err != nil {
		return nil, err
	}

	settings, err := m.realtimeURLSettings(options.SessionConfig)
	if err != nil {
		return nil, err
	}

	issuedAt := time.Now().Truncate(time.Second)

	token, err := m.provider.createAccessToken(ctx, accessTokenSettings{
		expiresIn: expiresIn,
	})
	if err != nil {
		return nil, err
	}

	target, err := m.provider.streamingURL(settings)
	if err != nil {
		return nil, err
	}

	secret := &model.RealtimeClientSecret{
		Token: token,
		URL:   target.String(),
	}

	if expiresIn != nil {
		secret.ExpiresAt = issuedAt.Add(options.ExpiresAfter)
	}

	return secret, nil
}

// realtimeSecretLifetime returns the requested lifetime in seconds, or nil
// for the default.
func realtimeSecretLifetime(expiresAfter time.Duration) (*int, error) {
	if expiresAfter == 0 {
		return nil, nil
	}

	wholeSeconds := expiresAfter%time.Second == 0
	inRange := expiresAfter > 0 && expiresAfter <= maxRealtimeSecretLifetime

	if !wholeSeconds || !inRange {
		return nil, fmt.Errorf("%w: Cartesia realtime client secrets must expire between 1 and 3600 seconds", model.ErrInvalidArgument)
	}

	seconds := int(expiresAfter / time.Second)

	return &seconds, nil
}

// realtimeURLSettings derives the WebSocket query settings of a session.
func (m *RealtimeModel) realtimeURLSettings(config *model.RealtimeSessionConfig) (streamingURLSettings, error) {
	settings := streamingURLSettings{
		modelID:       m.modelID,
		encoding:      StreamingPCMS16LE,
		turnDetection: true,
	}

	if config == nil {
		return settings, nil
	}

	language := realtimeLanguage(config)
	if language != nil && *language != streamingLanguage {
		return streamingURLSettings{}, fmt.Errorf("%w: Cartesia Ink 2 currently supports English only", model.ErrInvalidArgument)
	}

	settings.language = language
	settings.turnDetection = usesTurnDetection(config.TurnDetection)

	if format := config.InputAudioFormat; format != nil {
		encoding, err := realtimeEncoding(format.Type)
		if err != nil {
			return streamingURLSettings{}, err
		}

		settings.encoding = encoding
		settings.sampleRate = format.Rate
	}

	return settings, nil
}

// realtimeLanguage returns the transcription language hint, or nil when the
// configuration has none.
func realtimeLanguage(config *model.RealtimeSessionConfig) *string {
	transcription := config.InputAudioTranscription
	if transcription == nil || transcription.Language == "" {
		return nil
	}

	return &transcription.Language
}

// usesTurnDetection reports whether a session uses Cartesia's turn
// detection, which applies unless detection is disabled.
func usesTurnDetection(turnDetection *model.RealtimeTurnDetection) bool {
	return turnDetection == nil || turnDetection.Type != model.RealtimeTurnDetectionDisabled
}

// realtimeEncoding returns the encoding of an input audio media type, which
// defaults to 16-bit PCM.
func realtimeEncoding(mediaType string) (StreamingEncoding, error) {
	if mediaType == "" {
		return StreamingPCMS16LE, nil
	}

	encoding, err := inferredEncoding(mediaType)
	if err != nil {
		return "", fmt.Errorf("%w: unsupported Cartesia realtime audio format: %s", model.ErrInvalidArgument, mediaType)
	}

	return encoding, nil
}

// WebSocketConfig carries the access token in the access_token query
// parameter and records whether the URL uses turn detection, which decides
// how input audio commits are sent. A URL that does not parse is returned
// unchanged and fails to connect.
func (m *RealtimeModel) WebSocketConfig(secret model.RealtimeClientSecret) model.RealtimeWebSocketConfig {
	target, err := url.Parse(secret.URL)
	if err != nil {
		return model.RealtimeWebSocketConfig{
			URL: secret.URL,
		}
	}

	m.turnDetection.Store(strings.Contains(target.Path, turnsWebSocketPath))

	return model.RealtimeWebSocketConfig{
		URL: withAccessToken(*target, secret.Token),
	}
}

// NewServerEventParser returns a parser with its own session and transcript
// state.
func (m *RealtimeModel) NewServerEventParser() model.RealtimeServerEventParser {
	return newRealtimeEventParser().parse
}

// ParseServerEvent maps one Cartesia event with the model's shared state.
// Prefer NewServerEventParser, which keeps state per connection.
func (m *RealtimeModel) ParseServerEvent(raw json.RawMessage) ([]model.RealtimeServerEvent, error) {
	return m.parser.parse(raw)
}

// SerializeClientEvent maps a normalized event to a Cartesia frame. Input
// audio becomes a binary frame of the decoded audio. Without turn detection,
// a commit becomes the "finalize" text command; with it, Cartesia detects
// turns itself. Session updates, buffer clears and response requests send
// nothing, and other events are unsupported.
func (m *RealtimeModel) SerializeClientEvent(event model.RealtimeClientEvent) (*model.RealtimeFrame, error) {
	switch typed := event.(type) {
	case model.RealtimeSessionUpdate, model.RealtimeInputAudioClear, model.RealtimeResponseCreate:
		return nil, nil

	case model.RealtimeInputAudioAppend:
		return realtimeAudioFrame(typed.Audio)

	case model.RealtimeInputAudioCommit:
		if m.turnDetection.Load() {
			return nil, nil
		}

		return model.NewRealtimeTextFrame(streamingFinalizeCommand), nil

	default:
		return nil, fmt.Errorf("%w: Cartesia Ink 2 does not support realtime client event %T", model.ErrUnsupported, event)
	}
}

// realtimeAudioFrame decodes base64 input audio into a binary frame.
func realtimeAudioFrame(audio string) (*model.RealtimeFrame, error) {
	decoded, err := base64.StdEncoding.DecodeString(audio)
	if err != nil {
		return nil, fmt.Errorf("%w: decode realtime input audio: %w", model.ErrInvalidArgument, err)
	}

	return model.NewRealtimeBinaryFrame(decoded), nil
}

// BuildSessionConfig returns no configuration: Cartesia configures sessions
// through the WebSocket URL.
func (m *RealtimeModel) BuildSessionConfig(_ model.RealtimeSessionConfig) (json.RawMessage, error) {
	return nil, nil
}
