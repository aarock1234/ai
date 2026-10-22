package xai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// Streaming option limits, in milliseconds.
const (
	maxEndpointingMs      = 5000
	minSmartTurnTimeoutMs = 1
	maxSmartTurnTimeoutMs = 5000
)

// defaultStreamingEncoding is the encoding of input formats without a known
// xAI encoding.
const defaultStreamingEncoding = "pcm"

// streamingEncodings maps input audio media types onto xAI encodings.
var streamingEncodings = map[string]string{
	"audio/pcm":  "pcm",
	"audio/pcmu": "mulaw",
	"audio/pcma": "alaw",
	"audio/opus": "opus",
}

var _ model.StreamingTranscriptionModel = (*TranscriptionModel)(nil)

// TranscriptionStreamingOptions configure StreamTranscribe.
type TranscriptionStreamingOptions struct {
	// InterimResults emits interim transcripts while speech is processed.
	InterimResults *bool `json:"interimResults,omitempty"`

	// Endpointing is the silence in milliseconds, up to 5000, before an
	// utterance ends.
	Endpointing *int `json:"endpointing,omitempty"`

	// SmartTurn is the end-of-turn detection threshold between 0 and 1.
	// Setting it enables Smart Turn.
	SmartTurn *float64 `json:"smartTurn,omitempty"`

	// SmartTurnTimeout is the silence in milliseconds, from 1 to 5000,
	// before Smart Turn forces the end of speech.
	SmartTurnTimeout *int `json:"smartTurnTimeout,omitempty"`
}

func (o TranscriptionStreamingOptions) validate() error {
	if o.Endpointing != nil && (*o.Endpointing < 0 || *o.Endpointing > maxEndpointingMs) {
		return fmt.Errorf("streaming.endpointing must be between 0 and %d", maxEndpointingMs)
	}

	if o.SmartTurn != nil && (*o.SmartTurn < 0 || *o.SmartTurn > 1) {
		return errors.New("streaming.smartTurn must be between 0 and 1")
	}

	if o.SmartTurnTimeout != nil && (*o.SmartTurnTimeout < minSmartTurnTimeoutMs || *o.SmartTurnTimeout > maxSmartTurnTimeoutMs) {
		return fmt.Errorf("streaming.smartTurnTimeout must be between %d and %d", minSmartTurnTimeoutMs, maxSmartTurnTimeoutMs)
	}

	return nil
}

// StreamTranscribe transcribes live audio over the speech-to-text
// WebSocket. The audio is sent once the server created the transcript, and
// after Audio reaches io.EOF the stream finishes when every channel is
// done. Multichannel audio requires the channels option.
func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, options model.TranscriptionStreamOptions) (*model.TranscriptionStreamResult, error) {
	if options.Audio == nil {
		return nil, fmt.Errorf("%w: streaming transcription requires audio", model.ErrInvalidArgument)
	}

	timestamp := time.Now()

	transcriptionOptions, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	multichannel := isTrue(transcriptionOptions.Multichannel)
	if multichannel && transcriptionOptions.Channels == nil {
		return nil, fmt.Errorf(
			"%w: providerOptions.xai.channels is required when providerOptions.xai.multichannel is true",
			model.ErrInvalidArgument,
		)
	}

	request, err := m.provider.apiRequest(transcriptionPath, options.Headers)
	if err != nil {
		return nil, err
	}

	target, err := m.streamingURL(request.URL, options.InputAudioFormat, transcriptionOptions)
	if err != nil {
		return nil, err
	}

	requestBody, err := json.Marshal(target)
	if err != nil {
		return nil, fmt.Errorf("encode streaming transcription url: %w", err)
	}

	conn, err := livetranscription.Dial(ctx, livetranscription.DialOptions{
		URL:        target,
		Client:     request.Client,
		Header:     request.Headers,
		ParseError: request.ParseError,
	})
	if err != nil {
		return nil, err
	}

	expectedChannels := 1
	if multichannel {
		expectedChannels = *transcriptionOptions.Channels
	}

	protocol := newStreamingProtocol(streamingProtocolSetup{
		warnings:         streamingWarnings(options.InputAudioFormat, transcriptionOptions),
		language:         optional.Or(transcriptionOptions.Language, ""),
		expectedChannels: expectedChannels,
	})

	result := livetranscription.NewResult(ctx, livetranscription.TranscriptionSetup{
		Conn:             conn,
		Protocol:         protocol,
		Audio:            options.Audio,
		IncludeRawChunks: options.IncludeRawChunks,
		Name:             "xai streaming transcription",
	})
	result.Request = model.RequestInfo{
		Body: requestBody,
	}
	result.Response = model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   conn.ResponseHeader(),
	}

	return result, nil
}

// streamingURL returns the WebSocket URL with the session settings as
// query parameters. The sample rate falls back to the input format's rate
// and the encoding to the input format's encoding.
func (m *TranscriptionModel) streamingURL(endpoint string, format model.AudioFormat, options TranscriptionOptions) (string, error) {
	target, err := livetranscription.WebSocketURL(endpoint)
	if err != nil {
		return "", err
	}

	query := target.Query()

	if m.modelID != "" {
		query.Set("model", m.modelID)
	}

	sampleRate := format.Rate
	if options.SampleRate != nil {
		sampleRate = *options.SampleRate
	}

	if sampleRate != 0 {
		query.Set("sample_rate", strconv.Itoa(sampleRate))
	}

	if options.Language != nil {
		query.Set("language", *options.Language)
	}

	setBool(query, "multichannel", options.Multichannel)
	setInt(query, "channels", options.Channels)
	setBool(query, "diarize", options.Diarize)
	setBool(query, "filler_words", options.FillerWords)
	setFloat(query, "vad_threshold", options.VadThreshold)

	for _, term := range options.Keyterm {
		query.Add("keyterm", term)
	}

	query.Set("encoding", streamingEncoding(format, options))

	if streaming := options.Streaming; streaming != nil {
		setBool(query, "interim_results", streaming.InterimResults)
		setInt(query, "endpointing", streaming.Endpointing)
		setFloat(query, "smart_turn", streaming.SmartTurn)
		setInt(query, "smart_turn_timeout", streaming.SmartTurnTimeout)
	}

	target.RawQuery = query.Encode()

	return target.String(), nil
}

// streamingEncoding returns the audioFormat option, the encoding of the
// input format, or raw PCM.
func streamingEncoding(format model.AudioFormat, options TranscriptionOptions) string {
	if options.AudioFormat != nil {
		return *options.AudioFormat
	}

	if encoding, ok := streamingEncodings[strings.ToLower(format.Type)]; ok {
		return encoding
	}

	return defaultStreamingEncoding
}

// streamingWarnings reports the format option, which only Transcribe
// supports, and input formats without a known encoding.
func streamingWarnings(format model.AudioFormat, options TranscriptionOptions) []model.Warning {
	var warnings []model.Warning

	if options.Format != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"providerOptions.xai.format",
			"xAI streaming transcription does not support format.",
		))
	}

	if _, known := streamingEncodings[strings.ToLower(format.Type)]; options.AudioFormat == nil && !known {
		warnings = append(warnings, model.Warning{
			Type: model.WarningOther,
			Message: fmt.Sprintf(
				"Unrecognized inputAudioFormat.type %q; falling back to raw PCM encoding. Use audio/pcm, audio/pcmu, audio/pcma, audio/opus, or set providerOptions.xai.audioFormat explicitly.",
				format.Type,
			),
		})
	}

	return warnings
}

func setBool(query url.Values, name string, value *bool) {
	if value != nil {
		query.Set(name, strconv.FormatBool(*value))
	}
}

func setInt(query url.Values, name string, value *int) {
	if value != nil {
		query.Set(name, strconv.Itoa(*value))
	}
}

func setFloat(query url.Values, name string, value *float64) {
	if value != nil {
		query.Set(name, strconv.FormatFloat(*value, 'f', -1, 64))
	}
}

func isTrue(value *bool) bool {
	return value != nil && *value
}
