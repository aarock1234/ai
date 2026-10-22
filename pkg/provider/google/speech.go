package google

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	defaultSpeechVoice = "Kore"

	// defaultSampleRate is the rate of raw PCM when the response omits it.
	defaultSampleRate = 24000

	audioWAV = "AUDIO_WAV"
	audioL16 = "AUDIO_L16"
)

var (
	sampleRatePattern = regexp.MustCompile(`rate=(\d+)`)
	pcmMimePattern    = regexp.MustCompile(`(?i)^audio/(?:l16|pcm)(?:;|$)`)
)

var _ model.SpeechModel = (*SpeechModel)(nil)

// SpeechOptions are the Google provider options of a speech call, read
// from ProviderOptions["google"].
type SpeechOptions struct {
	// SpeechMetadata directs the top-level text, for Gemini 3.8 TTS.
	SpeechMetadata *SpeechTurnMetadata `json:"speechMetadata,omitempty"`

	// Turns is a structured transcript for Gemini 3.8 TTS. It replaces the
	// top-level text, so pass empty text with turns. Each multi-speaker turn
	// names a configured speaker; per-turn styles override instructions.
	Turns []SpeechTurn `json:"turns,omitempty"`

	// MultiSpeakerVoiceConfig configures dialogue audio with up to two
	// speakers. It overrides the top-level voice.
	MultiSpeakerVoiceConfig *MultiSpeakerVoiceConfig `json:"multiSpeakerVoiceConfig,omitempty"`
}

// SpeechTurnMetadata directs one turn of speech.
type SpeechTurnMetadata struct {
	Speaker string `json:"speaker,omitempty"`
	Style   string `json:"style,omitempty"`
}

// SpeechTurn is one turn of a structured transcript.
type SpeechTurn struct {
	Text           string              `json:"text"`
	SpeechMetadata *SpeechTurnMetadata `json:"speechMetadata,omitempty"`
}

// MultiSpeakerVoiceConfig assigns voices to speakers.
type MultiSpeakerVoiceConfig struct {
	SpeakerVoiceConfigs []SpeakerVoiceConfig `json:"speakerVoiceConfigs"`
}

// SpeakerVoiceConfig assigns a voice to one speaker.
type SpeakerVoiceConfig struct {
	Speaker     string      `json:"speaker"`
	VoiceConfig VoiceConfig `json:"voiceConfig"`
}

// VoiceConfig selects a prebuilt voice.
type VoiceConfig struct {
	PrebuiltVoiceConfig PrebuiltVoiceConfig `json:"prebuiltVoiceConfig"`

	// Voice selects a custom voice, which is not supported. Setting it is an
	// error.
	Voice json.RawMessage `json:"voice,omitempty"`
}

// PrebuiltVoiceConfig names a prebuilt voice, such as "Kore".
type PrebuiltVoiceConfig struct {
	VoiceName string `json:"voiceName"`
}

// SpeechResultMetadata is the Google metadata of a speech response, stored
// under ProviderMetadata["google"].
type SpeechResultMetadata struct {
	SampleRate int     `json:"sampleRate"`
	MimeType   *string `json:"mimeType"`
}

// SpeechModel generates speech with Gemini text-to-speech models. It is safe
// for concurrent use.
type SpeechModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider name, "google.generative-ai.speech" by
// default.
func (m *SpeechModel) Provider() string {
	return m.provider.name + ".speech"
}

// ModelID returns the model ID.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

type speechRequest struct {
	Contents         []speechContent        `json:"contents"`
	GenerationConfig speechGenerationConfig `json:"generationConfig"`
}

type speechContent struct {
	Role  contentRole  `json:"role"`
	Parts []speechPart `json:"parts"`
}

type speechPart struct {
	Text           string              `json:"text"`
	SpeechMetadata *SpeechTurnMetadata `json:"speechMetadata,omitempty"`
}

type speechGenerationConfig struct {
	ResponseModalities []ResponseModality    `json:"responseModalities"`
	SpeechConfig       speechConfig          `json:"speechConfig"`
	ResponseFormat     *speechResponseFormat `json:"responseFormat,omitempty"`
}

type speechConfig struct {
	VoiceConfig             *VoiceConfig             `json:"voiceConfig,omitempty"`
	MultiSpeakerVoiceConfig *MultiSpeakerVoiceConfig `json:"multiSpeakerVoiceConfig,omitempty"`
}

type speechResponseFormat struct {
	Audio speechAudioFormat `json:"audio"`
}

type speechAudioFormat struct {
	MimeType string `json:"mimeType"`
}

type speechResponse struct {
	Candidates    []speechCandidate `json:"candidates"`
	UsageMetadata json.RawMessage   `json:"usageMetadata"`
}

type speechCandidate struct {
	Content *speechCandidateContent `json:"content"`
}

type speechCandidateContent struct {
	Parts []speechResponsePart `json:"parts"`
}

type speechResponsePart struct {
	InlineData *speechInlineData `json:"inlineData"`
}

type speechInlineData struct {
	MimeType *string `json:"mimeType"`
	Data     *string `json:"data"`
}

// audioData returns the base64 audio, or an empty string.
func (d *speechInlineData) audioData() string {
	if d == nil || d.Data == nil {
		return ""
	}

	return *d.Data
}

// preparedSpeech is a converted speech call.
type preparedSpeech struct {
	body                 speechRequest
	warnings             []model.Warning
	outputFormat         string
	usesStructuredSpeech bool
}

// Generate generates audio from the text. WAV output is the default; raw PCM
// from older models is wrapped in a WAV container.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	speech, err := m.prepareSpeech(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.request(ctx, "models/"+m.modelID+":generateContent", options.Headers)
	if err != nil {
		return nil, err
	}

	raw, err := apicall.PostJSON(ctx, request, speech.body)
	if err != nil {
		return nil, err
	}

	var response speechResponse
	if err := apicall.DecodeJSON(raw.Body, &response); err != nil {
		return nil, err
	}

	audio, mimeType, err := firstAudio(response)
	if err != nil {
		return nil, err
	}

	sampleRate := parseSampleRate(mimeType)

	// Older models return PCM, which needs a container for WAV output.
	// Gemini 3.8 returns WAV itself, which another header would corrupt.
	needsContainer := speech.outputFormat == audioWAV && isRawPCM(mimeType, speech.usesStructuredSpeech)
	if needsContainer && len(audio) > 0 {
		audio = addWAVHeader(audio, sampleRate)
	}

	warnings := speech.warnings
	if speech.outputFormat == audioL16 && len(audio) > 0 && !speech.usesStructuredSpeech {
		warnings = append(warnings, model.UnsupportedWarning("outputFormat", fmt.Sprintf(
			"Returning raw PCM audio (signed 16-bit little-endian, mono, %d Hz). "+
				"These bytes have no container header and are not directly playable; "+
				"see providerMetadata.google for the sample rate and mime type.",
			sampleRate,
		)))
	}

	var usage json.RawMessage
	if wirejson.HasValue(response.UsageMetadata) {
		usage = response.UsageMetadata
	}

	return &model.SpeechResult{
		Audio:    audio,
		Usage:    usage,
		Warnings: warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: SpeechResultMetadata{
				SampleRate: sampleRate,
				MimeType:   mimeType,
			},
		},
		Request: model.RequestInfo{
			Body: raw.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: time.Now(),
				ModelID:   m.modelID,
			},
			Headers: raw.Headers,
			Body:    raw.Body,
		},
	}, nil
}

// usesStructuredSpeech reports models with structured speech metadata and
// response formats. Gemini 2.5 and 3.1 models take directions in the prompt;
// newer and custom models use structured speech.
func (m *SpeechModel) usesStructuredSpeech() bool {
	return !strings.HasPrefix(m.modelID, "gemini-2.5-") && !strings.HasPrefix(m.modelID, "gemini-3.1-")
}

func (m *SpeechModel) prepareSpeech(options model.SpeechOptions) (preparedSpeech, error) {
	googleOptions, _, err := readProviderValue[SpeechOptions](options.ProviderOptions, m.provider.optionNamespaces())
	if err != nil {
		return preparedSpeech{}, err
	}

	if googleOptions.Turns != nil && len(googleOptions.Turns) == 0 {
		return preparedSpeech{}, fmt.Errorf("%w: providerOptions: turns must contain at least one turn", model.ErrInvalidArgument)
	}

	voice := options.Voice
	if voice == "" {
		voice = defaultSpeechVoice
	}

	if usesCustomVoice(voice, googleOptions.MultiSpeakerVoiceConfig) {
		return preparedSpeech{}, fmt.Errorf("%w: voice: custom voices are not supported; use a prebuilt voice instead", model.ErrInvalidArgument)
	}

	structured := m.usesStructuredSpeech()

	parts, warnings, err := speechParts(options, googleOptions, structured)
	if err != nil {
		return preparedSpeech{}, err
	}

	if transcript(options.Text, googleOptions.Turns) == "" {
		return preparedSpeech{}, fmt.Errorf("%w: text: speech input must contain a non-empty transcript", model.ErrInvalidArgument)
	}

	warnings = append(warnings, unsupportedSpeechWarnings(options)...)

	outputFormat, formatWarning := speechOutputFormat(options.OutputFormat, structured)
	if formatWarning != nil {
		warnings = append(warnings, *formatWarning)
	}

	generation := speechGenerationConfig{
		ResponseModalities: []ResponseModality{
			ResponseModalityAudio,
		},
		SpeechConfig: speechVoiceConfig(voice, googleOptions.MultiSpeakerVoiceConfig),
	}

	if structured && options.OutputFormat != "" {
		generation.ResponseFormat = &speechResponseFormat{
			Audio: speechAudioFormat{
				MimeType: outputFormat,
			},
		}
	}

	return preparedSpeech{
		body: speechRequest{
			Contents: []speechContent{
				{
					Role:  roleUser,
					Parts: parts,
				},
			},
			GenerationConfig: generation,
		},
		warnings:             warnings,
		outputFormat:         outputFormat,
		usesStructuredSpeech: structured,
	}, nil
}

// speechParts builds the transcript parts. Older models take instructions
// as a prompt prefix; structured models take them as the turn style.
func speechParts(options model.SpeechOptions, googleOptions SpeechOptions, structured bool) ([]speechPart, []model.Warning, error) {
	if !structured {
		return promptedSpeechParts(options, googleOptions)
	}

	if googleOptions.Turns != nil && googleOptions.SpeechMetadata != nil {
		return nil, nil, fmt.Errorf("%w: providerOptions: set speechMetadata on each turn when using turns", model.ErrInvalidArgument)
	}

	var warnings []model.Warning
	if googleOptions.Turns != nil && options.Text != "" {
		warnings = append(warnings, model.UnsupportedWarning("text", "Google TTS turns replace the top-level text."))
	}

	turns := googleOptions.Turns
	if turns == nil {
		turns = []SpeechTurn{
			{
				Text:           options.Text,
				SpeechMetadata: googleOptions.SpeechMetadata,
			},
		}
	}

	parts := make([]speechPart, 0, len(turns))
	for _, turn := range turns {
		speechPart, err := structuredSpeechPart(turn, options.Instructions, googleOptions.MultiSpeakerVoiceConfig)
		if err != nil {
			return nil, nil, err
		}

		parts = append(parts, speechPart)
	}

	return parts, warnings, nil
}

func structuredSpeechPart(turn SpeechTurn, instructions string, speakers *MultiSpeakerVoiceConfig) (speechPart, error) {
	var metadata SpeechTurnMetadata
	if turn.SpeechMetadata != nil {
		metadata = *turn.SpeechMetadata
	}

	if metadata.Style == "" {
		metadata.Style = instructions
	}

	if speakers != nil && !speakers.hasSpeaker(metadata.Speaker) {
		return speechPart{}, fmt.Errorf("%w: speechMetadata.speaker: every multi-speaker turn must specify a speechMetadata.speaker matching a configured speaker", model.ErrInvalidArgument)
	}

	converted := speechPart{
		Text: turn.Text,
	}

	if metadata != (SpeechTurnMetadata{}) {
		converted.SpeechMetadata = &metadata
	}

	return converted, nil
}

func promptedSpeechParts(options model.SpeechOptions, googleOptions SpeechOptions) ([]speechPart, []model.Warning, error) {
	if googleOptions.Turns != nil || googleOptions.SpeechMetadata != nil {
		return nil, nil, fmt.Errorf("%w: providerOptions: structured speech metadata and turns require Gemini 3.8 TTS", model.ErrInvalidArgument)
	}

	var warnings []model.Warning
	text := options.Text

	if options.Instructions != "" {
		if googleOptions.MultiSpeakerVoiceConfig != nil {
			warnings = append(warnings, model.UnsupportedWarning(
				"instructions",
				"Google Gemini TTS ignores `instructions` when `multiSpeakerVoiceConfig` is set, "+
					"because prepending them would break multi-speaker transcript parsing.",
			))
		} else {
			text = options.Instructions + ": " + options.Text
		}
	}

	return []speechPart{
		{
			Text: text,
		},
	}, warnings, nil
}

func (c *MultiSpeakerVoiceConfig) hasSpeaker(speaker string) bool {
	return slices.ContainsFunc(c.SpeakerVoiceConfigs, func(config SpeakerVoiceConfig) bool {
		return config.Speaker == speaker
	})
}

// transcript returns the text the model speaks: the joined turns, or the
// top-level text.
func transcript(text string, turns []SpeechTurn) string {
	if len(turns) == 0 {
		return text
	}

	var joined strings.Builder
	for _, turn := range turns {
		joined.WriteString(turn.Text)
	}

	return joined.String()
}

func usesCustomVoice(voice string, speakers *MultiSpeakerVoiceConfig) bool {
	if strings.HasPrefix(voice, "voice_") || strings.HasPrefix(voice, "voicekey_") {
		return true
	}

	if speakers == nil {
		return false
	}

	return slices.ContainsFunc(speakers.SpeakerVoiceConfigs, func(config SpeakerVoiceConfig) bool {
		return config.VoiceConfig.Voice != nil
	})
}

func speechVoiceConfig(voice string, speakers *MultiSpeakerVoiceConfig) speechConfig {
	if speakers != nil {
		return speechConfig{
			MultiSpeakerVoiceConfig: speakers,
		}
	}

	return speechConfig{
		VoiceConfig: &VoiceConfig{
			PrebuiltVoiceConfig: PrebuiltVoiceConfig{
				VoiceName: voice,
			},
		},
	}
}

func unsupportedSpeechWarnings(options model.SpeechOptions) []model.Warning {
	var warnings []model.Warning

	if options.Speed != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"speed",
			"Google Gemini TTS models do not support the `speed` option. It was ignored.",
		))
	}

	if options.Language != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"language",
			"Google Gemini TTS models do not support the `language` option. Language is detected automatically from the input text.",
		))
	}

	return warnings
}

// speechOutputFormat resolves the requested format to a Gemini audio
// encoding, falling back to WAV.
func speechOutputFormat(requested string, structured bool) (string, *model.Warning) {
	formats := map[string]string{
		"wav": audioWAV,
		"pcm": audioL16,
	}

	if structured {
		formats = map[string]string{
			"wav":         audioWAV,
			"audio/wav":   audioWAV,
			"pcm":         audioL16,
			"audio/l16":   audioL16,
			"mulaw":       "AUDIO_MULAW",
			"audio/mulaw": "AUDIO_MULAW",
			"alaw":        "AUDIO_ALAW",
			"audio/alaw":  "AUDIO_ALAW",
		}
	}

	if requested == "" {
		return audioWAV, nil
	}

	if encoding, ok := formats[requested]; ok {
		return encoding, nil
	}

	warning := model.UnsupportedWarning("outputFormat", fmt.Sprintf("Unsupported output format: %s. Using wav instead.", requested))

	return audioWAV, &warning
}

// firstAudio returns the first inline audio of the response. Gemini returns
// one inline audio part per request.
func firstAudio(response speechResponse) ([]byte, *string, error) {
	for _, candidate := range response.Candidates {
		if candidate.Content == nil {
			continue
		}

		for _, responsePart := range candidate.Content.Parts {
			encoded := responsePart.InlineData.audioData()
			if encoded == "" {
				continue
			}

			audio, err := base64.StdEncoding.DecodeString(encoded)
			if err != nil {
				return nil, nil, fmt.Errorf("%w: decode audio: %w", model.ErrInvalidResponse, err)
			}

			return audio, responsePart.InlineData.MimeType, nil
		}
	}

	return []byte{}, nil, nil
}

// parseSampleRate reads the rate of a PCM mime type such as
// "audio/L16;rate=24000".
func parseSampleRate(mimeType *string) int {
	if mimeType == nil {
		return defaultSampleRate
	}

	match := sampleRatePattern.FindStringSubmatch(*mimeType)
	if match == nil {
		return defaultSampleRate
	}

	rate, err := strconv.Atoi(match[1])
	if err != nil {
		return defaultSampleRate
	}

	return rate
}

// isRawPCM reports PCM audio. Without a mime type, only the older
// prompt-directed models return PCM.
func isRawPCM(mimeType *string, structured bool) bool {
	if mimeType == nil {
		return !structured
	}

	return pcmMimePattern.MatchString(*mimeType)
}

// addWAVHeader wraps signed 16-bit little-endian mono PCM in a minimal
// 44-byte WAV container.
func addWAVHeader(pcm []byte, sampleRate int) []byte {
	const (
		channels      = 1
		bitsPerSample = 16
		headerBytes   = 44
		fmtChunkBytes = 16
		pcmFormat     = 1
	)

	blockAlign := channels * bitsPerSample / 8
	byteRate := sampleRate * blockAlign

	wav := make([]byte, 0, headerBytes+len(pcm))
	wav = append(wav, "RIFF"...)
	wav = binary.LittleEndian.AppendUint32(wav, uint32(headerBytes-8+len(pcm)))
	wav = append(wav, "WAVE"...)
	wav = append(wav, "fmt "...)
	wav = binary.LittleEndian.AppendUint32(wav, fmtChunkBytes)
	wav = binary.LittleEndian.AppendUint16(wav, pcmFormat)
	wav = binary.LittleEndian.AppendUint16(wav, channels)
	wav = binary.LittleEndian.AppendUint32(wav, uint32(sampleRate))
	wav = binary.LittleEndian.AppendUint32(wav, uint32(byteRate))
	wav = binary.LittleEndian.AppendUint16(wav, uint16(blockAlign))
	wav = binary.LittleEndian.AppendUint16(wav, bitsPerSample)
	wav = append(wav, "data"...)
	wav = binary.LittleEndian.AppendUint32(wav, uint32(len(pcm)))

	return append(wav, pcm...)
}
