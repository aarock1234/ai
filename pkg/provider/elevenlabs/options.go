package elevenlabs

import (
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	// maxSpeechReferences limits pronunciation dictionaries and request IDs.
	maxSpeechReferences = 3

	// maxSeed is the largest speech seed.
	maxSeed = 4294967295
)

// TextNormalization controls text normalization, such as spelling out
// numbers.
type TextNormalization string

// Text normalization modes.
const (
	TextNormalizationAuto TextNormalization = "auto"
	TextNormalizationOn   TextNormalization = "on"
	TextNormalizationOff  TextNormalization = "off"
)

// SpeechOptions are the provider options of speech models.
type SpeechOptions struct {
	// LanguageCode is the ISO 639-1 language of the text. The call's language
	// takes precedence.
	LanguageCode *string `json:"languageCode,omitempty"`

	// VoiceSettings override the stored settings of the voice.
	VoiceSettings *VoiceSettings `json:"voiceSettings,omitempty"`

	// PronunciationDictionaryLocators apply up to three pronunciation
	// dictionaries, in order.
	PronunciationDictionaryLocators []PronunciationDictionaryLocator `json:"pronunciationDictionaryLocators,omitempty"`

	// Seed makes sampling deterministic on a best-effort basis, from 0 to
	// 4294967295.
	Seed *int64 `json:"seed,omitempty"`

	// PreviousText is the text before this request, for continuity.
	PreviousText *string `json:"previousText,omitempty"`

	// NextText is the text after this request, for continuity.
	NextText *string `json:"nextText,omitempty"`

	// PreviousRequestIDs are up to three earlier generations, for continuity.
	PreviousRequestIDs []string `json:"previousRequestIds,omitempty"`

	// NextRequestIDs are up to three later generations, for continuity.
	NextRequestIDs []string `json:"nextRequestIds,omitempty"`

	// ApplyTextNormalization controls text normalization.
	ApplyTextNormalization TextNormalization `json:"applyTextNormalization,omitempty"`

	// ApplyLanguageTextNormalization enables language-specific text
	// normalization.
	ApplyLanguageTextNormalization *bool `json:"applyLanguageTextNormalization,omitempty"`

	// EnableLogging controls request logging. Disabling it enables zero
	// retention mode for eligible accounts.
	EnableLogging *bool `json:"enableLogging,omitempty"`
}

// VoiceSettings tune a voice. Each number is between 0 and 1.
type VoiceSettings struct {
	Stability       *float64 `json:"stability,omitempty"`
	SimilarityBoost *float64 `json:"similarityBoost,omitempty"`
	Style           *float64 `json:"style,omitempty"`
	UseSpeakerBoost *bool    `json:"useSpeakerBoost,omitempty"`
}

// PronunciationDictionaryLocator identifies a pronunciation dictionary.
type PronunciationDictionaryLocator struct {
	PronunciationDictionaryID string `json:"pronunciationDictionaryId"`

	// VersionID defaults to the latest version.
	VersionID *string `json:"versionId,omitempty"`
}

func readSpeechOptions(options model.ProviderOptions) (SpeechOptions, error) {
	speechOptions, _, err := model.ProviderValue[SpeechOptions](options, Namespace)
	if err != nil {
		return SpeechOptions{}, err
	}

	if err := speechOptions.validate(); err != nil {
		return SpeechOptions{}, err
	}

	return speechOptions, nil
}

func (o SpeechOptions) validate() error {
	if o.VoiceSettings != nil {
		if err := o.VoiceSettings.validate(); err != nil {
			return err
		}
	}

	if len(o.PronunciationDictionaryLocators) > maxSpeechReferences {
		return fmt.Errorf("%w: elevenlabs accepts at most %d pronunciationDictionaryLocators", model.ErrInvalidArgument, maxSpeechReferences)
	}

	if o.Seed != nil && (*o.Seed < 0 || *o.Seed > maxSeed) {
		return fmt.Errorf("%w: elevenlabs seed must be between 0 and %d", model.ErrInvalidArgument, maxSeed)
	}

	if len(o.PreviousRequestIDs) > maxSpeechReferences || len(o.NextRequestIDs) > maxSpeechReferences {
		return fmt.Errorf("%w: elevenlabs accepts at most %d previous or next request IDs", model.ErrInvalidArgument, maxSpeechReferences)
	}

	switch o.ApplyTextNormalization {
	case "", TextNormalizationAuto, TextNormalizationOn, TextNormalizationOff:
		return nil
	default:
		return fmt.Errorf("%w: elevenlabs applyTextNormalization %q", model.ErrInvalidArgument, o.ApplyTextNormalization)
	}
}

func (s VoiceSettings) validate() error {
	if !floatWithin(s.Stability, 0, 1) {
		return fmt.Errorf("%w: elevenlabs voiceSettings.stability must be between 0 and 1", model.ErrInvalidArgument)
	}

	if !floatWithin(s.SimilarityBoost, 0, 1) {
		return fmt.Errorf("%w: elevenlabs voiceSettings.similarityBoost must be between 0 and 1", model.ErrInvalidArgument)
	}

	if !floatWithin(s.Style, 0, 1) {
		return fmt.Errorf("%w: elevenlabs voiceSettings.style must be between 0 and 1", model.ErrInvalidArgument)
	}

	return nil
}

// TimestampsGranularity selects the timestamps of a transcription.
type TimestampsGranularity string

// Timestamp granularities.
const (
	TimestampsNone      TimestampsGranularity = "none"
	TimestampsWord      TimestampsGranularity = "word"
	TimestampsCharacter TimestampsGranularity = "character"
)

// FileFormat is the format of the input audio.
type FileFormat string

// File formats.
const (
	// FilePCMS16LE16 is 16-bit little-endian mono PCM at 16 kHz, which has
	// lower latency than an encoded waveform.
	FilePCMS16LE16 FileFormat = "pcm_s16le_16"

	// FileOther is any encoded audio.
	FileOther FileFormat = "other"
)

// CommitStrategy selects how realtime transcripts are committed.
type CommitStrategy string

// Commit strategies.
const (
	CommitManual CommitStrategy = "manual"
	CommitVAD    CommitStrategy = "vad"
)

const (
	// maxSpeakers is the largest numSpeakers.
	maxSpeakers = 32

	// maxKeyterms limits TranscriptionStreamingOptions.Keyterms.
	maxKeyterms = 50

	// maxKeytermLength limits each keyterm.
	maxKeytermLength = 20
)

// TranscriptionOptions are the provider options of transcription models.
// When the namespace is present, unset tagAudioEvents, timestampsGranularity
// and fileFormat are sent with their upstream defaults.
type TranscriptionOptions struct {
	// LanguageCode is the ISO 639-1 or 639-3 language of the audio. It is
	// detected when unset.
	LanguageCode *string `json:"languageCode,omitempty"`

	// TagAudioEvents tags events such as (laughter). It defaults to true.
	TagAudioEvents *bool `json:"tagAudioEvents,omitempty"`

	// NumSpeakers is the maximum number of speakers, from 1 to 32.
	NumSpeakers *int `json:"numSpeakers,omitempty"`

	// TimestampsGranularity defaults to word timestamps.
	TimestampsGranularity TimestampsGranularity `json:"timestampsGranularity,omitempty"`

	// Diarize annotates the speaker of each word. It defaults to true.
	Diarize *bool `json:"diarize,omitempty"`

	// FileFormat defaults to FileOther.
	FileFormat FileFormat `json:"fileFormat,omitempty"`

	// Streaming configures realtime transcription. Batch transcription warns
	// that it is unsupported.
	Streaming *TranscriptionStreamingOptions `json:"streaming,omitempty"`
}

// TranscriptionStreamingOptions configure realtime transcription with
// scribe_v2_realtime.
type TranscriptionStreamingOptions struct {
	CommitStrategy           CommitStrategy `json:"commitStrategy,omitempty"`
	EnableLogging            *bool          `json:"enableLogging,omitempty"`
	FilterBackgroundAudio    *bool          `json:"filterBackgroundAudio,omitempty"`
	IncludeLanguageDetection *bool          `json:"includeLanguageDetection,omitempty"`
	IncludeTimestamps        *bool          `json:"includeTimestamps,omitempty"`

	// Keyterms are up to 50 terms of at most 20 characters that bias
	// recognition.
	Keyterms []string `json:"keyterms,omitempty"`

	// MinSilenceDurationMS is between 50 and 2000.
	MinSilenceDurationMS *int `json:"minSilenceDurationMs,omitempty"`

	// MinSpeechDurationMS is between 50 and 2000.
	MinSpeechDurationMS *int `json:"minSpeechDurationMs,omitempty"`

	NoVerbatim         *bool    `json:"noVerbatim,omitempty"`
	PreviousText       *string  `json:"previousText,omitempty"`
	SecondaryLanguages []string `json:"secondaryLanguages,omitempty"`

	// VADSilenceThresholdSecs is between 0.3 and 3.
	VADSilenceThresholdSecs *float64 `json:"vadSilenceThresholdSecs,omitempty"`

	// VADThreshold is between 0.1 and 0.9.
	VADThreshold *float64 `json:"vadThreshold,omitempty"`
}

func readTranscriptionOptions(options model.ProviderOptions) (TranscriptionOptions, bool, error) {
	transcriptionOptions, present, err := model.ProviderValue[TranscriptionOptions](options, Namespace)
	if err != nil {
		return TranscriptionOptions{}, false, err
	}

	if err := transcriptionOptions.validate(); err != nil {
		return TranscriptionOptions{}, false, err
	}

	return transcriptionOptions, present, nil
}

func (o TranscriptionOptions) validate() error {
	if o.NumSpeakers != nil && (*o.NumSpeakers < 1 || *o.NumSpeakers > maxSpeakers) {
		return fmt.Errorf("%w: elevenlabs numSpeakers must be between 1 and %d", model.ErrInvalidArgument, maxSpeakers)
	}

	switch o.TimestampsGranularity {
	case "", TimestampsNone, TimestampsWord, TimestampsCharacter:
	default:
		return fmt.Errorf("%w: elevenlabs timestampsGranularity %q", model.ErrInvalidArgument, o.TimestampsGranularity)
	}

	switch o.FileFormat {
	case "", FilePCMS16LE16, FileOther:
	default:
		return fmt.Errorf("%w: elevenlabs fileFormat %q", model.ErrInvalidArgument, o.FileFormat)
	}

	if o.Streaming == nil {
		return nil
	}

	return o.Streaming.validate()
}

func (o TranscriptionStreamingOptions) validate() error {
	switch o.CommitStrategy {
	case "", CommitManual, CommitVAD:
	default:
		return fmt.Errorf("%w: elevenlabs streaming commitStrategy %q", model.ErrInvalidArgument, o.CommitStrategy)
	}

	if len(o.Keyterms) > maxKeyterms {
		return fmt.Errorf("%w: elevenlabs streaming accepts at most %d keyterms", model.ErrInvalidArgument, maxKeyterms)
	}

	tooLong := slices.IndexFunc(o.Keyterms, func(keyterm string) bool {
		return len([]rune(keyterm)) > maxKeytermLength
	})
	if tooLong >= 0 {
		return fmt.Errorf("%w: elevenlabs streaming keyterms must have at most %d characters", model.ErrInvalidArgument, maxKeytermLength)
	}

	if !intWithin(o.MinSilenceDurationMS, 50, 2000) || !intWithin(o.MinSpeechDurationMS, 50, 2000) {
		return fmt.Errorf("%w: elevenlabs streaming minimum durations must be between 50 and 2000 ms", model.ErrInvalidArgument)
	}

	if !floatWithin(o.VADSilenceThresholdSecs, 0.3, 3) {
		return fmt.Errorf("%w: elevenlabs streaming vadSilenceThresholdSecs must be between 0.3 and 3", model.ErrInvalidArgument)
	}

	if !floatWithin(o.VADThreshold, 0.1, 0.9) {
		return fmt.Errorf("%w: elevenlabs streaming vadThreshold must be between 0.1 and 0.9", model.ErrInvalidArgument)
	}

	return nil
}

// intWithin reports whether an optional value is unset or within the
// inclusive bounds.
func intWithin(value *int, minimum int, maximum int) bool {
	return value == nil || (*value >= minimum && *value <= maximum)
}

// floatWithin reports whether an optional value is unset or within the
// inclusive bounds.
func floatWithin(value *float64, minimum float64, maximum float64) bool {
	return value == nil || (*value >= minimum && *value <= maximum)
}
