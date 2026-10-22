package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// realtimePath is the realtime WebSocket endpoint; the transcription
	// intent is added as a query parameter.
	realtimePath = "/realtime"

	realtimeIntent = "transcription"

	// realtimeProtocol is the WebSocket subprotocol of the realtime API.
	realtimeProtocol = "realtime"

	// realtimeAPIKeyProtocol prefixes the subprotocol that carries the API
	// key.
	realtimeAPIKeyProtocol = "openai-insecure-api-key."
)

// realtimeBearerPattern extracts the token of a bearer Authorization header.
// The scheme is case-insensitive.
var realtimeBearerPattern = regexp.MustCompile(`(?i)^bearer\s+(.+)$`)

// Realtime transcription delays, from the lowest latency to the highest
// accuracy.
var realtimeDelays = []string{
	"minimal",
	"low",
	"medium",
	"high",
	"xhigh",
}

// realtimeSessionUpdate configures a realtime transcription session.
type realtimeSessionUpdate struct {
	Type    string          `json:"type"`
	Session realtimeSession `json:"session"`
}

type realtimeSession struct {
	Type    string               `json:"type"`
	Audio   realtimeSessionAudio `json:"audio"`
	Include []string             `json:"include,omitempty"`
}

type realtimeSessionAudio struct {
	Input realtimeAudioInput `json:"input"`
}

type realtimeAudioInput struct {
	Format        realtimeAudioFormat           `json:"format"`
	Transcription realtimeTranscriptionSettings `json:"transcription"`

	// TurnDetection is always null: the audio is committed once it ends,
	// rather than split by voice activity.
	TurnDetection *realtimeTurnDetection `json:"turn_detection"`
}

type realtimeAudioFormat struct {
	Type string `json:"type"`
	Rate int    `json:"rate,omitempty"`
}

type realtimeTranscriptionSettings struct {
	Model    string `json:"model"`
	Language string `json:"language,omitempty"`
	Delay    string `json:"delay,omitempty"`
}

// realtimeTurnDetection is never sent; see realtimeAudioInput.
type realtimeTurnDetection struct{}

// StreamTranscribe transcribes live audio with a realtime model, such as
// gpt-realtime-whisper, over the realtime WebSocket API. Other models do
// not stream. The audio is committed once Audio reaches io.EOF, and the
// stream finishes with the completed transcript of that audio.
func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, options model.TranscriptionStreamOptions) (*model.TranscriptionStreamResult, error) {
	if !transcriptionIsRealtimeModel(m.modelID) {
		return nil, fmt.Errorf("%w: streaming transcription with %s", model.ErrUnsupported, m.modelID)
	}

	if options.Audio == nil {
		return nil, fmt.Errorf("%w: streaming transcription requires audio", model.ErrInvalidArgument)
	}

	timestamp := time.Now()

	openaiOptions, _, err := model.ProviderValue[TranscriptionOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := openaiOptions.validate(); err != nil {
		return nil, err
	}

	sessionUpdate, err := json.Marshal(m.realtimeSessionUpdate(options.InputAudioFormat, openaiOptions))
	if err != nil {
		return nil, fmt.Errorf("encode realtime session: %w", err)
	}

	conn, err := m.dialRealtime(ctx, options.Headers)
	if err != nil {
		return nil, err
	}

	protocol := &realtimeTranscriptionProtocol{
		warnings:      realtimeWarnings(openaiOptions),
		sessionUpdate: sessionUpdate,
		language:      openaiOptions.Language,
	}

	result := livetranscription.NewResult(ctx, livetranscription.TranscriptionSetup{
		Conn:             conn,
		Protocol:         protocol,
		Audio:            options.Audio,
		IncludeRawChunks: options.IncludeRawChunks,
		Name:             "openai realtime transcription",
	})
	result.Request = model.RequestInfo{
		Body: sessionUpdate,
	}
	result.Response = model.ResponseInfo{
		ResponseMetadata: model.ResponseMetadata{
			Timestamp: timestamp,
			ModelID:   m.modelID,
		},
		Headers: conn.ResponseHeader(),
	}

	return result, nil
}

// realtimeSessionUpdate returns the session.update event for the call.
func (m *TranscriptionModel) realtimeSessionUpdate(format model.AudioFormat, options TranscriptionOptions) realtimeSessionUpdate {
	session := realtimeSession{
		Type: "transcription",
		Audio: realtimeSessionAudio{
			Input: realtimeAudioInput{
				Format: realtimeAudioFormat{
					Type: format.Type,
					Rate: format.Rate,
				},
				Transcription: realtimeTranscriptionSettings{
					Model:    m.modelID,
					Language: options.Language,
				},
			},
		},
	}

	if options.Streaming != nil {
		session.Audio.Input.Transcription.Delay = options.Streaming.Delay
		session.Include = options.Streaming.Include
	}

	return realtimeSessionUpdate{
		Type:    "session.update",
		Session: session,
	}
}

// realtimeWarnings reports options that apply only to the REST
// transcription endpoint.
func realtimeWarnings(options TranscriptionOptions) []model.Warning {
	unsupported := []struct {
		name string
		set  bool
	}{
		{
			name: "include",
			set:  options.Include != nil,
		},
		{
			name: "prompt",
			set:  options.Prompt != "",
		},
		{
			name: "temperature",
			set:  options.Temperature != nil,
		},
		{
			name: "timestampGranularities",
			set:  options.TimestampGranularities != nil,
		},
		{
			name: "responseFormat",
			set:  options.ResponseFormat != "",
		},
		{
			name: "chunkingStrategy",
			set:  options.ChunkingStrategy != nil,
		},
	}

	var warnings []model.Warning

	for _, option := range unsupported {
		if !option.set {
			continue
		}

		warnings = append(warnings, model.UnsupportedWarning(
			"providerOptions.openai."+option.name,
			"OpenAI streaming transcription does not support "+option.name+".",
		))
	}

	return warnings
}

// dialRealtime opens the realtime WebSocket with the provider's
// authentication and headers.
func (m *TranscriptionModel) dialRealtime(ctx context.Context, callHeaders http.Header) (*websocket.Conn, error) {
	request, err := m.config.request(ctx, realtimePath, callHeaders)
	if err != nil {
		return nil, err
	}

	target, err := realtimeURL(request.URL)
	if err != nil {
		return nil, err
	}

	return dialRealtimeSocket(ctx, request, target)
}

// realtimeURL converts the http(s) endpoint URL to ws(s) and adds the
// transcription intent, keeping any query the provider configured.
func realtimeURL(endpoint string) (*url.URL, error) {
	target, err := livetranscription.WebSocketURL(endpoint)
	if err != nil {
		return nil, err
	}

	query := target.Query()
	query.Set("intent", realtimeIntent)
	target.RawQuery = query.Encode()

	return target, nil
}

// realtimeAuthentication moves a bearer token from the Authorization header
// to the API key subprotocol, as upstream does: OpenAI rejects handshakes
// that send both, and browser WebSockets can only use the subprotocol.
// Other authentication headers, such as Azure's api-key, stay headers.
func realtimeAuthentication(headers http.Header) ([]string, http.Header) {
	protocols := []string{
		realtimeProtocol,
	}

	match := realtimeBearerPattern.FindStringSubmatch(headers.Get("Authorization"))
	if match == nil {
		return protocols, headers
	}

	withoutAuthorization := headers.Clone()
	withoutAuthorization.Del("Authorization")

	return append(protocols, realtimeAPIKeyProtocol+match[1]), withoutAuthorization
}

// dialRealtimeSocket opens a realtime WebSocket at target with the
// request's client, headers and error parser. A rejected handshake is
// reported like a failed HTTP call; errors name the URL without its query,
// which a host may use for credentials.
func dialRealtimeSocket(ctx context.Context, request apicall.Request, target *url.URL) (*websocket.Conn, error) {
	protocols, headers := realtimeAuthentication(request.Headers)

	displayURL := *target
	displayURL.RawQuery = ""

	return livetranscription.Dial(ctx, livetranscription.DialOptions{
		URL:        target.String(),
		DisplayURL: displayURL.String(),
		Client:     request.Client,
		Header:     headers,
		Protocols:  protocols,
		ParseError: request.ParseError,
	})
}

// validate checks the realtime delay.
func (o *TranscriptionStreamingOptions) validate() error {
	if o == nil || o.Delay == "" || slices.Contains(realtimeDelays, o.Delay) {
		return nil
	}

	return transcriptionOptionError("unknown streaming.delay %q", o.Delay)
}
