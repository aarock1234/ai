package revai

import (
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// Tier is the quality tier of diarization, summarization and translation.
type Tier string

// Tiers.
const (
	TierStandard Tier = "standard"
	TierPremium  Tier = "premium"
)

// SummaryType is the format of a summary.
type SummaryType string

// Summary types.
const (
	SummaryParagraph SummaryType = "paragraph"
	SummaryBullets   SummaryType = "bullets"
)

// translationLanguages are the accepted translation target languages.
var translationLanguages = []string{
	"en",
	"en-us",
	"en-gb",
	"ar",
	"pt",
	"pt-br",
	"pt-pt",
	"fr",
	"fr-ca",
	"es",
	"es-es",
	"es-la",
	"it",
	"ja",
	"ko",
	"de",
	"ru",
}

// TranscriptionOptions are the provider options of transcription models.
// Their keys are snake_case, as upstream. When the namespace is present,
// unset settings with an upstream default are sent with that default:
// false for the flags, "standard" diarization and "en" language.
type TranscriptionOptions struct {
	Metadata               *string              `json:"metadata,omitempty"`
	NotificationConfig     *NotificationConfig  `json:"notification_config,omitempty"`
	DeleteAfterSeconds     *float64             `json:"delete_after_seconds,omitempty"`
	Verbatim               *bool                `json:"verbatim,omitempty"`
	Rush                   *bool                `json:"rush,omitempty"`
	TestMode               *bool                `json:"test_mode,omitempty"`
	SegmentsToTranscribe   []Segment            `json:"segments_to_transcribe,omitempty"`
	SpeakerNames           []SpeakerName        `json:"speaker_names,omitempty"`
	SkipDiarization        *bool                `json:"skip_diarization,omitempty"`
	SkipPostprocessing     *bool                `json:"skip_postprocessing,omitempty"`
	SkipPunctuation        *bool                `json:"skip_punctuation,omitempty"`
	RemoveDisfluencies     *bool                `json:"remove_disfluencies,omitempty"`
	RemoveAtmospherics     *bool                `json:"remove_atmospherics,omitempty"`
	FilterProfanity        *bool                `json:"filter_profanity,omitempty"`
	SpeakerChannelsCount   *float64             `json:"speaker_channels_count,omitempty"`
	SpeakersCount          *float64             `json:"speakers_count,omitempty"`
	DiarizationType        Tier                 `json:"diarization_type,omitempty"`
	CustomVocabularyID     *string              `json:"custom_vocabulary_id,omitempty"`
	CustomVocabularies     []CustomVocabulary   `json:"custom_vocabularies,omitempty"`
	StrictCustomVocabulary *bool                `json:"strict_custom_vocabulary,omitempty"`
	SummarizationConfig    *SummarizationConfig `json:"summarization_config,omitempty"`
	TranslationConfig      *TranslationConfig   `json:"translation_config,omitempty"`
	Language               *string              `json:"language,omitempty"`
	ForcedAlignment        *bool                `json:"forced_alignment,omitempty"`
}

// NotificationConfig receives a callback when the job finishes.
type NotificationConfig struct {
	URL         string       `json:"url"`
	AuthHeaders *AuthHeaders `json:"auth_headers,omitempty"`
}

// AuthHeaders authenticate the notification callback.
type AuthHeaders struct {
	Authorization string `json:"Authorization"`
}

// Segment is a span of the audio, in seconds.
type Segment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// SpeakerName names a speaker.
type SpeakerName struct {
	DisplayName string `json:"display_name"`
}

// CustomVocabulary lists phrases that guide recognition.
type CustomVocabulary struct {
	Phrases []string `json:"phrases"`
}

// SummarizationConfig requests a summary. Model defaults to TierStandard
// and Type to SummaryParagraph.
type SummarizationConfig struct {
	Model  Tier        `json:"model,omitempty"`
	Type   SummaryType `json:"type,omitempty"`
	Prompt *string     `json:"prompt,omitempty"`
}

// TranslationConfig requests translations. Model defaults to TierStandard.
type TranslationConfig struct {
	TargetLanguages []TargetLanguage `json:"target_languages"`
	Model           Tier             `json:"model,omitempty"`
}

// TargetLanguage is a translation target, such as "es".
type TargetLanguage struct {
	Language string `json:"language"`
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
	if !validTier(o.DiarizationType) {
		return invalidOption(fmt.Sprintf("diarization_type %q", o.DiarizationType))
	}

	if config := o.SummarizationConfig; config != nil {
		if !validTier(config.Model) {
			return invalidOption(fmt.Sprintf("summarization model %q", config.Model))
		}

		switch config.Type {
		case "", SummaryParagraph, SummaryBullets:
		default:
			return invalidOption(fmt.Sprintf("summarization type %q", config.Type))
		}
	}

	if config := o.TranslationConfig; config != nil {
		if !validTier(config.Model) {
			return invalidOption(fmt.Sprintf("translation model %q", config.Model))
		}

		for _, target := range config.TargetLanguages {
			if !slices.Contains(translationLanguages, target.Language) {
				return invalidOption(fmt.Sprintf("translation language %q", target.Language))
			}
		}
	}

	return nil
}

func validTier(tier Tier) bool {
	return tier == "" || tier == TierStandard || tier == TierPremium
}

func invalidOption(details string) error {
	return fmt.Errorf("%w: revai %s", model.ErrInvalidArgument, details)
}
