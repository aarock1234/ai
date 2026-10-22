package cartesia

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// accessTokenPath creates short-lived access tokens.
	accessTokenPath = "/access-token"

	// turnsWebSocketPath streams transcription with turn detection.
	turnsWebSocketPath = "/stt/turns/websocket"

	// webSocketPath streams transcription without turn detection.
	webSocketPath = "/stt/websocket"

	// defaultStreamingSampleRate is the rate when the input format has none.
	defaultStreamingSampleRate = 24000

	// streamingLanguage is the only language of Ink 2.
	streamingLanguage = "en"
)

// accessTokenRequest asks for a token that may open a transcription stream.
type accessTokenRequest struct {
	Grants accessTokenGrants `json:"grants"`

	// ExpiresIn is the token lifetime in seconds, or nil for Cartesia's
	// default.
	ExpiresIn *int `json:"expires_in,omitempty"`
}

type accessTokenGrants struct {
	STT bool `json:"stt"`
}

type accessTokenResponse struct {
	Token string `json:"token"`
}

var _ model.StreamingTranscriptionModel = (*TranscriptionModel)(nil)

// StreamTranscribe transcribes live audio with Ink 2 and later models over
// Cartesia's streaming WebSocket. Ink Whisper models do not stream. The
// input audio must be audio/pcm, audio/pcmu or audio/pcma. Turn detection
// is used unless TranscriptionStreamingOptions.TurnDetection is false. Once
// Audio reaches io.EOF, the stream finishes when Cartesia reports that it is
// done.
func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, options model.TranscriptionStreamOptions) (*model.TranscriptionStreamResult, error) {
	if isBatchModel(m.modelID) {
		return nil, fmt.Errorf("%w: streaming transcription with %s", model.ErrUnsupported, m.modelID)
	}

	if options.Audio == nil {
		return nil, fmt.Errorf("%w: streaming transcription requires audio", model.ErrInvalidArgument)
	}

	timestamp := time.Now()

	transcriptionOptions, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	if transcriptionOptions.Language != nil && *transcriptionOptions.Language != streamingLanguage {
		return nil, fmt.Errorf("%w: Cartesia Ink 2 currently supports English only", model.ErrInvalidArgument)
	}

	encoding, warnings, err := streamingEncoding(options.InputAudioFormat.Type, transcriptionOptions)
	if err != nil {
		return nil, err
	}

	token, err := m.provider.createAccessToken(ctx, accessTokenSettings{
		callHeaders: options.Headers,
	})
	if err != nil {
		return nil, err
	}

	turnDetection := transcriptionOptions.Streaming == nil ||
		transcriptionOptions.Streaming.TurnDetection == nil ||
		*transcriptionOptions.Streaming.TurnDetection

	target, err := m.provider.streamingURL(streamingURLSettings{
		modelID:       m.modelID,
		encoding:      encoding,
		sampleRate:    options.InputAudioFormat.Rate,
		language:      transcriptionOptions.Language,
		turnDetection: turnDetection,
	})
	if err != nil {
		return nil, err
	}

	displayURL := target.String()
	authenticatedURL := withAccessToken(*target, token)

	requestBody, err := json.Marshal(displayURL)
	if err != nil {
		return nil, fmt.Errorf("encode streaming transcription url: %w", err)
	}

	conn, err := livetranscription.Dial(ctx, livetranscription.DialOptions{
		URL:        authenticatedURL,
		DisplayURL: displayURL,
		Client:     m.provider.httpClient,
		ParseError: parseError,
	})
	if err != nil {
		return nil, err
	}

	result := livetranscription.NewResult(ctx, livetranscription.TranscriptionSetup{
		Conn: conn,
		Protocol: &streamingProtocol{
			warnings:      warnings,
			turnDetection: turnDetection,
		},
		Audio:            options.Audio,
		IncludeRawChunks: options.IncludeRawChunks,
		Name:             "cartesia streaming transcription",
	})
	result.Request = model.RequestInfo{
		Body: requestBody,
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

// streamingEncoding returns the encoding to send: the override, or the
// encoding of the declared media type. Widening generic audio/pcm to another
// linear PCM encoding is the purpose of the override; other overrides
// contradict the declared format and are reported, then sent.
func streamingEncoding(mediaType string, options TranscriptionOptions) (StreamingEncoding, []model.Warning, error) {
	var warnings []model.Warning

	if options.TimestampGranularities != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"providerOptions.cartesia.timestampGranularities",
			"Cartesia streaming transcription does not support timestamp granularities.",
		))
	}

	inferred, err := inferredEncoding(mediaType)
	if err != nil {
		return "", nil, err
	}

	if options.Streaming == nil || options.Streaming.Encoding == "" {
		return inferred, warnings, nil
	}

	encoding := options.Streaming.Encoding
	widensPCM := inferred == StreamingPCMS16LE && isLinearPCM(encoding)

	if encoding != inferred && !widensPCM {
		warnings = append(warnings, model.Warning{
			Type: model.WarningOther,
			Message: fmt.Sprintf(
				"providerOptions.cartesia.streaming.encoding '%s' contradicts inputAudioFormat.type '%s' (inferred '%s'); sending '%s'.",
				encoding,
				mediaType,
				inferred,
				encoding,
			),
		})
	}

	return encoding, warnings, nil
}

func inferredEncoding(mediaType string) (StreamingEncoding, error) {
	switch mediaType {
	case "audio/pcm":
		return StreamingPCMS16LE, nil
	case "audio/pcmu":
		return StreamingPCMMulaw, nil
	case "audio/pcma":
		return StreamingPCMAlaw, nil
	default:
		return "", fmt.Errorf("%w: unsupported Cartesia streaming audio format: %s", model.ErrInvalidArgument, mediaType)
	}
}

// isLinearPCM reports whether an encoding is linear PCM rather than one of
// the two G.711 companding laws.
func isLinearPCM(encoding StreamingEncoding) bool {
	return encoding != StreamingPCMMulaw && encoding != StreamingPCMAlaw
}

// accessTokenSettings are the inputs of an access token request.
type accessTokenSettings struct {
	callHeaders http.Header

	// expiresIn is the lifetime in seconds, or nil for Cartesia's default.
	expiresIn *int
}

// createAccessToken returns a short-lived token that may open a
// transcription stream; the WebSocket carries no API key.
func (p *Provider) createAccessToken(ctx context.Context, settings accessTokenSettings) (string, error) {
	request, err := p.apiRequest(accessTokenPath, settings.callHeaders)
	if err != nil {
		return "", err
	}

	response, err := apicall.PostJSON(ctx, request, accessTokenRequest{
		Grants: accessTokenGrants{
			STT: true,
		},
		ExpiresIn: settings.expiresIn,
	})
	if err != nil {
		return "", err
	}

	var decoded accessTokenResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return "", err
	}

	if decoded.Token == "" {
		return "", fmt.Errorf("%w: cartesia access token is empty", model.ErrInvalidResponse)
	}

	return decoded.Token, nil
}

// streamingURLSettings are the query settings of a transcription stream.
type streamingURLSettings struct {
	modelID       string
	encoding      StreamingEncoding
	sampleRate    int
	language      *string
	turnDetection bool
}

// streamingURL returns the WebSocket URL of a stream, without its access
// token. The WebSocket path replaces any path of the base URL.
func (p *Provider) streamingURL(settings streamingURLSettings) (*url.URL, error) {
	base, err := livetranscription.WebSocketURL(p.baseURL)
	if err != nil {
		return nil, err
	}

	path := turnsWebSocketPath
	if !settings.turnDetection {
		path = webSocketPath
	}

	target := base.ResolveReference(&url.URL{
		Path: path,
	})

	sampleRate := settings.sampleRate
	if sampleRate == 0 {
		sampleRate = defaultStreamingSampleRate
	}

	query := target.Query()
	query.Set("model", settings.modelID)
	query.Set("encoding", string(settings.encoding))
	query.Set("sample_rate", strconv.Itoa(sampleRate))
	query.Set("cartesia_version", p.version)

	if !settings.turnDetection && settings.language != nil {
		query.Set("language", *settings.language)
	}

	target.RawQuery = query.Encode()

	return target, nil
}

func withAccessToken(target url.URL, token string) string {
	query := target.Query()
	query.Set("access_token", token)
	target.RawQuery = query.Encode()

	return target.String()
}
