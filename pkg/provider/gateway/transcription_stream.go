package gateway

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/transcriptionstream"
	"github.com/aarock1234/ai/pkg/model"
)

// WebSocket subprotocols of streaming transcription. The marker lets the
// gateway echo a negotiated subprotocol; the others carry the bearer token
// and the optional team, which browser WebSockets cannot send as headers.
const (
	transcriptionSubprotocol = "ai-gateway-transcription.v1"
	authSubprotocolPrefix    = "ai-gateway-auth."
	teamSubprotocolPrefix    = "ai-gateway-team."
)

var _ model.StreamingTranscriptionModel = (*TranscriptionModel)(nil)

// StreamTranscribe transcribes live audio through the gateway over the
// transcription stream WebSocket envelope: the gateway forwards the audio to
// the model's provider and streams its parts back. The input audio format
// and provider options are passed through unchanged.
func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, options model.TranscriptionStreamOptions) (*model.TranscriptionStreamResult, error) {
	if options.Audio == nil {
		return nil, fmt.Errorf("%w: streaming transcription requires audio", model.ErrInvalidArgument)
	}

	timestamp := time.Now()

	request, method, err := m.provider.modelRequest(
		transcriptionModelPath,
		options.Headers,
		modelConfigHeaders("transcription", m.modelID),
	)
	if err != nil {
		return nil, err
	}

	startFrame, err := transcriptionstream.EncodeStartFrame(options)
	if err != nil {
		return nil, err
	}

	audioDoneFrame, err := transcriptionstream.EncodeAudioDoneFrame()
	if err != nil {
		return nil, err
	}

	target, err := m.transcriptionStreamURL()
	if err != nil {
		return nil, err
	}

	conn, err := livetranscription.Dial(ctx, livetranscription.DialOptions{
		URL:        target,
		Client:     request.Client,
		Header:     request.Headers,
		Protocols:  transcriptionProtocols(request.Headers),
		ParseError: request.ParseError,
	})
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	protocol := &transcriptionStreamProtocol{
		startFrame:     startFrame,
		audioDoneFrame: audioDoneFrame,
		method:         method,
	}

	result := livetranscription.NewResult(ctx, livetranscription.TranscriptionSetup{
		Conn:            conn,
		Protocol:        protocol,
		Audio:           options.Audio,
		AudioChunkBytes: transcriptionstream.MaxAudioFrameBytes,
		Name:            "AI Gateway transcription stream",
	})
	result.Request = model.RequestInfo{
		Body: startFrame,
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

// transcriptionStreamURL returns the WebSocket URL of the transcription
// endpoint. The model ID rides the ai-model-id query parameter, which is
// safe for qualified IDs such as openai/gpt-realtime-whisper.
func (m *TranscriptionModel) transcriptionStreamURL() (string, error) {
	endpoint := m.provider.baseURL + transcriptionModelPath
	if rest, isHTTP := strings.CutPrefix(endpoint, "http"); isHTTP {
		endpoint = "ws" + rest
	}

	target, err := url.Parse(endpoint)
	if err != nil {
		return "", fmt.Errorf("%w: gateway transcription stream url: %w", model.ErrInvalidArgument, err)
	}

	query := target.Query()
	query.Set("ai-model-id", m.modelID)
	target.RawQuery = query.Encode()

	return target.String(), nil
}

// transcriptionProtocols returns the subprotocols that carry the bearer
// token and team of the request headers.
func transcriptionProtocols(headers http.Header) []string {
	protocols := []string{
		transcriptionSubprotocol,
	}

	token, isBearer := strings.CutPrefix(headers.Get("Authorization"), "Bearer ")
	if !isBearer {
		return protocols
	}

	protocols = append(protocols, authSubprotocolPrefix+token)

	if team := headers.Get(teamHeader); team != "" {
		protocols = append(protocols, teamSubprotocolPrefix+base64.RawURLEncoding.EncodeToString([]byte(team)))
	}

	return protocols
}
