package assemblyai

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// AudioTagRemoval selects the inline annotations removed from rich
// transcripts.
type AudioTagRemoval string

// Audio tag removal modes.
const (
	// RemoveAllAudioTags removes every inline annotation.
	RemoveAllAudioTags AudioTagRemoval = "all"

	// RemoveSpeakerAudioTags removes only speaker cues.
	RemoveSpeakerAudioTags AudioTagRemoval = "speaker"
)

// AudioRedactionSilence replaces redacted PII with silence.
const AudioRedactionSilence = "silence"

// TranscriptionOptions are the provider options of transcription models.
// Times are in milliseconds.
type TranscriptionOptions struct {
	AudioEndAt     *int  `json:"audioEndAt,omitempty"`
	AudioStartFrom *int  `json:"audioStartFrom,omitempty"`
	AutoChapters   *bool `json:"autoChapters,omitempty"`
	AutoHighlights *bool `json:"autoHighlights,omitempty"`

	// BoostParam is "low", "default" or "high".
	//
	// Deprecated: it only applies to WordBoost. Use KeytermsPrompt.
	BoostParam *string `json:"boostParam,omitempty"`

	ContentSafety *bool `json:"contentSafety,omitempty"`

	// ContentSafetyConfidence is between 25 and 100.
	ContentSafetyConfidence *int `json:"contentSafetyConfidence,omitempty"`

	CustomSpelling []CustomSpelling `json:"customSpelling,omitempty"`
	Disfluencies   *bool            `json:"disfluencies,omitempty"`

	// Domain enables a domain-specific model, such as "medical-v1".
	Domain *string `json:"domain,omitempty"`

	EntityDetection *bool `json:"entityDetection,omitempty"`
	FilterProfanity *bool `json:"filterProfanity,omitempty"`
	FormatText      *bool `json:"formatText,omitempty"`
	IABCategories   *bool `json:"iabCategories,omitempty"`

	// KeytermsPrompt boosts domain-specific terms of up to six words.
	KeytermsPrompt []string `json:"keytermsPrompt,omitempty"`

	LanguageCode                *string                   `json:"languageCode,omitempty"`
	LanguageConfidenceThreshold *float64                  `json:"languageConfidenceThreshold,omitempty"`
	LanguageDetection           *bool                     `json:"languageDetection,omitempty"`
	LanguageDetectionOptions    *LanguageDetectionOptions `json:"languageDetectionOptions,omitempty"`
	Multichannel                *bool                     `json:"multichannel,omitempty"`

	// Prompt is natural-language context of up to 1,500 words.
	Prompt *string `json:"prompt,omitempty"`

	Punctuate                 *bool                  `json:"punctuate,omitempty"`
	RedactPII                 *bool                  `json:"redactPii,omitempty"`
	RedactPIIAudio            *bool                  `json:"redactPiiAudio,omitempty"`
	RedactPIIAudioOptions     *RedactPIIAudioOptions `json:"redactPiiAudioOptions,omitempty"`
	RedactPIIAudioQuality     *string                `json:"redactPiiAudioQuality,omitempty"`
	RedactPIIPolicies         []string               `json:"redactPiiPolicies,omitempty"`
	RedactPIIReturnUnredacted *bool                  `json:"redactPiiReturnUnredacted,omitempty"`
	RedactPIISub              *string                `json:"redactPiiSub,omitempty"`

	// RedactStaticEntities maps labels to exact terms to redact. It requires
	// RedactPII.
	RedactStaticEntities map[string][]string `json:"redactStaticEntities,omitempty"`

	RemoveAudioTags   AudioTagRemoval `json:"removeAudioTags,omitempty"`
	SentimentAnalysis *bool           `json:"sentimentAnalysis,omitempty"`
	SpeakerLabels     *bool           `json:"speakerLabels,omitempty"`
	SpeakerOptions    *SpeakerOptions `json:"speakerOptions,omitempty"`
	SpeakersExpected  *int            `json:"speakersExpected,omitempty"`

	// SpeechThreshold is between 0 and 1.
	SpeechThreshold *float64 `json:"speechThreshold,omitempty"`

	Summarization *bool   `json:"summarization,omitempty"`
	SummaryModel  *string `json:"summaryModel,omitempty"`
	SummaryType   *string `json:"summaryType,omitempty"`

	// Temperature is between 0 and 1.
	Temperature *float64 `json:"temperature,omitempty"`

	WebhookAuthHeaderName  *string `json:"webhookAuthHeaderName,omitempty"`
	WebhookAuthHeaderValue *string `json:"webhookAuthHeaderValue,omitempty"`
	WebhookURL             *string `json:"webhookUrl,omitempty"`

	// WordBoost lists words to boost.
	//
	// Deprecated: universal-3 and slam-1 models reject it. Use
	// KeytermsPrompt.
	WordBoost []string `json:"wordBoost,omitempty"`
}

// CustomSpelling replaces words or phrases with a spelling.
type CustomSpelling struct {
	From []string `json:"from"`
	To   string   `json:"to"`
}

// LanguageDetectionOptions configure automatic language detection.
type LanguageDetectionOptions struct {
	ExpectedLanguages []string `json:"expectedLanguages,omitempty"`
	FallbackLanguage  *string  `json:"fallbackLanguage,omitempty"`
	CodeSwitching     *bool    `json:"codeSwitching,omitempty"`

	// CodeSwitchingConfidenceThreshold is between 0 and 1.
	CodeSwitchingConfidenceThreshold *float64 `json:"codeSwitchingConfidenceThreshold,omitempty"`
}

// RedactPIIAudioOptions configure PII-redacted audio files.
type RedactPIIAudioOptions struct {
	ReturnRedactedNoSpeechAudio *bool `json:"returnRedactedNoSpeechAudio,omitempty"`

	// OverrideAudioRedactionMethod may be AudioRedactionSilence.
	OverrideAudioRedactionMethod *string `json:"overrideAudioRedactionMethod,omitempty"`
}

// SpeakerOptions bound the number of speakers.
type SpeakerOptions struct {
	MinSpeakersExpected *int `json:"minSpeakersExpected,omitempty"`
	MaxSpeakersExpected *int `json:"maxSpeakersExpected,omitempty"`
}

func readTranscriptionOptions(options model.ProviderOptions) (TranscriptionOptions, error) {
	transcriptionOptions, _, err := model.ProviderValue[TranscriptionOptions](options, Namespace)
	if err != nil {
		return TranscriptionOptions{}, err
	}

	if err := transcriptionOptions.validate(); err != nil {
		return TranscriptionOptions{}, err
	}

	return transcriptionOptions, nil
}

func (o TranscriptionOptions) validate() error {
	if o.ContentSafetyConfidence != nil && (*o.ContentSafetyConfidence < 25 || *o.ContentSafetyConfidence > 100) {
		return invalidOption("contentSafetyConfidence must be between 25 and 100")
	}

	if !unitInterval(o.SpeechThreshold) || !unitInterval(o.Temperature) {
		return invalidOption("speechThreshold and temperature must be between 0 and 1")
	}

	if o.LanguageDetectionOptions != nil && !unitInterval(o.LanguageDetectionOptions.CodeSwitchingConfidenceThreshold) {
		return invalidOption("codeSwitchingConfidenceThreshold must be between 0 and 1")
	}

	if o.RedactPIIAudioOptions != nil {
		method := o.RedactPIIAudioOptions.OverrideAudioRedactionMethod
		if method != nil && *method != AudioRedactionSilence {
			return invalidOption(fmt.Sprintf("overrideAudioRedactionMethod %q", *method))
		}
	}

	switch o.RemoveAudioTags {
	case "", RemoveAllAudioTags, RemoveSpeakerAudioTags:
		return nil
	default:
		return invalidOption(fmt.Sprintf("removeAudioTags %q", o.RemoveAudioTags))
	}
}

func invalidOption(details string) error {
	return fmt.Errorf("%w: assemblyai %s", model.ErrInvalidArgument, details)
}

// unitInterval reports whether an optional value is unset or between 0 and
// 1.
func unitInterval(value *float64) bool {
	return value == nil || (*value >= 0 && *value <= 1)
}
