package elevenlabs

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// realtimeTranscriptionPath is the realtime speech-to-text WebSocket.
	realtimeTranscriptionPath = "/v1/speech-to-text/realtime"

	// defaultRealtimeSampleRate is the PCM rate when the input format has
	// none.
	defaultRealtimeSampleRate = 16000

	// mulawSampleRate is the only rate of audio/pcmu input.
	mulawSampleRate = 8000
)

// realtimePCMSampleRates are the PCM rates realtime transcription accepts.
var realtimePCMSampleRates = []int{
	8000,
	16000,
	22050,
	24000,
	44100,
	48000,
}

// realtimeAudioFormat is the audio_format query parameter and sample rate
// of the input audio.
type realtimeAudioFormat struct {
	name       string
	sampleRate int
}

var _ model.StreamingTranscriptionModel = (*TranscriptionModel)(nil)

// StreamTranscribe transcribes live audio with scribe_v2_realtime over the
// realtime WebSocket. Other models do not stream. The input audio must be
// audio/pcm at 8000, 16000, 22050, 24000, 44100 or 48000 Hz, or audio/pcmu
// at 8000 Hz. Once Audio reaches io.EOF, the audio is committed and the
// stream finishes with the committed transcripts.
func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, options model.TranscriptionStreamOptions) (*model.TranscriptionStreamResult, error) {
	if m.modelID != realtimeModelID {
		return nil, fmt.Errorf("%w: streaming transcription with %s", model.ErrUnsupported, m.modelID)
	}

	if options.Audio == nil {
		return nil, fmt.Errorf("%w: streaming transcription requires audio", model.ErrInvalidArgument)
	}

	timestamp := time.Now()

	transcriptionOptions, _, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	streaming := transcriptionOptions.Streaming
	if streaming == nil {
		streaming = &TranscriptionStreamingOptions{}
	}

	if err := streaming.validateRealtime(); err != nil {
		return nil, err
	}

	format, err := realtimeInputFormat(options.InputAudioFormat)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(realtimeTranscriptionPath, options.Headers)
	if err != nil {
		return nil, err
	}

	target, err := m.realtimeURL(request.URL, format, transcriptionOptions)
	if err != nil {
		return nil, err
	}

	requestBody, err := json.Marshal(target)
	if err != nil {
		return nil, fmt.Errorf("encode realtime transcription url: %w", err)
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

	protocol := newRealtimeProtocol(realtimeProtocolSetup{
		warnings:                 realtimeWarnings(transcriptionOptions),
		sampleRate:               format.sampleRate,
		previousText:             streaming.PreviousText,
		language:                 optionalString(transcriptionOptions.LanguageCode),
		includeTimestamps:        isTrue(streaming.IncludeTimestamps),
		includeLanguageDetection: isTrue(streaming.IncludeLanguageDetection),
	})

	result := livetranscription.NewResult(ctx, livetranscription.TranscriptionSetup{
		Conn:             conn,
		Protocol:         protocol,
		Audio:            options.Audio,
		IncludeRawChunks: options.IncludeRawChunks,
		Name:             "elevenlabs realtime transcription",
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

// validateRealtime rejects background audio filtering together with the
// timestamped commit events, which ElevenLabs documents as incompatible.
// Language detection arrives on the same events.
func (o TranscriptionStreamingOptions) validateRealtime() error {
	if !isTrue(o.FilterBackgroundAudio) {
		return nil
	}

	if isTrue(o.IncludeTimestamps) || isTrue(o.IncludeLanguageDetection) {
		return fmt.Errorf(
			"%w: providerOptions.elevenlabs.streaming.filterBackgroundAudio cannot be combined with includeTimestamps or includeLanguageDetection",
			model.ErrInvalidArgument,
		)
	}

	return nil
}

// realtimeInputFormat maps the input audio format onto an ElevenLabs
// audio_format.
func realtimeInputFormat(format model.AudioFormat) (realtimeAudioFormat, error) {
	mediaType := strings.ToLower(format.Type)

	if mediaType == "audio/pcmu" {
		if format.Rate != 0 && format.Rate != mulawSampleRate {
			return realtimeAudioFormat{}, fmt.Errorf("%w: ElevenLabs only supports audio/pcmu at 8000 Hz", model.ErrInvalidArgument)
		}

		return realtimeAudioFormat{
			name:       "ulaw_8000",
			sampleRate: mulawSampleRate,
		}, nil
	}

	rate := format.Rate
	if rate == 0 {
		rate = defaultRealtimeSampleRate
	}

	if mediaType != "audio/pcm" || !slices.Contains(realtimePCMSampleRates, rate) {
		return realtimeAudioFormat{}, fmt.Errorf(
			"%w: ElevenLabs realtime transcription supports audio/pcm at 8000, 16000, 22050, 24000, 44100, or 48000 Hz, and audio/pcmu at 8000 Hz",
			model.ErrInvalidArgument,
		)
	}

	return realtimeAudioFormat{
		name:       "pcm_" + strconv.Itoa(rate),
		sampleRate: rate,
	}, nil
}

// realtimeURL returns the WebSocket URL with the session settings as query
// parameters. Timestamps are requested whenever language detection is,
// because the language arrives on the timestamped commit events.
func (m *TranscriptionModel) realtimeURL(endpoint string, format realtimeAudioFormat, options TranscriptionOptions) (string, error) {
	target, err := livetranscription.WebSocketURL(endpoint)
	if err != nil {
		return "", err
	}

	streaming := options.Streaming
	if streaming == nil {
		streaming = &TranscriptionStreamingOptions{}
	}

	query := target.Query()
	query.Set("model_id", m.modelID)
	query.Set("audio_format", format.name)

	if streaming.CommitStrategy != "" {
		query.Set("commit_strategy", string(streaming.CommitStrategy))
	}

	setBool(query, "enable_logging", streaming.EnableLogging)
	setBool(query, "filter_background_audio", streaming.FilterBackgroundAudio)
	setBool(query, "include_language_detection", streaming.IncludeLanguageDetection)

	if isTrue(streaming.IncludeLanguageDetection) {
		query.Set("include_timestamps", "true")
	} else {
		setBool(query, "include_timestamps", streaming.IncludeTimestamps)
	}

	if options.LanguageCode != nil {
		query.Set("language_code", *options.LanguageCode)
	}

	setInt(query, "min_silence_duration_ms", streaming.MinSilenceDurationMS)
	setInt(query, "min_speech_duration_ms", streaming.MinSpeechDurationMS)
	setBool(query, "no_verbatim", streaming.NoVerbatim)
	setFloat(query, "vad_silence_threshold_secs", streaming.VADSilenceThresholdSecs)
	setFloat(query, "vad_threshold", streaming.VADThreshold)

	for _, keyterm := range streaming.Keyterms {
		query.Add("keyterms", keyterm)
	}

	for _, language := range streaming.SecondaryLanguages {
		query.Add("secondary_languages", language)
	}

	target.RawQuery = query.Encode()

	return target.String(), nil
}

// realtimeWarnings reports options that only batch transcription supports.
func realtimeWarnings(options TranscriptionOptions) []model.Warning {
	unsupported := []struct {
		name string
		set  bool
	}{
		{
			name: "diarize",
			set:  options.Diarize != nil,
		},
		{
			name: "fileFormat",
			set:  options.FileFormat != "",
		},
		{
			name: "numSpeakers",
			set:  options.NumSpeakers != nil,
		},
		{
			name: "tagAudioEvents",
			set:  options.TagAudioEvents != nil,
		},
		{
			name: "timestampsGranularity",
			set:  options.TimestampsGranularity != "",
		},
	}

	var warnings []model.Warning

	for _, option := range unsupported {
		if !option.set {
			continue
		}

		warnings = append(warnings, model.UnsupportedWarning(
			"providerOptions.elevenlabs."+option.name,
			"ElevenLabs realtime transcription does not support "+option.name+".",
		))
	}

	return warnings
}

func setBool(query url.Values, name string, value *bool) {
	if value == nil {
		return
	}

	query.Set(name, strconv.FormatBool(*value))
}

func setInt(query url.Values, name string, value *int) {
	if value == nil {
		return
	}

	query.Set(name, strconv.Itoa(*value))
}

func setFloat(query url.Values, name string, value *float64) {
	if value == nil {
		return
	}

	query.Set(name, strconv.FormatFloat(*value, 'f', -1, 64))
}

func isTrue(value *bool) bool {
	return value != nil && *value
}

func optionalString(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}
