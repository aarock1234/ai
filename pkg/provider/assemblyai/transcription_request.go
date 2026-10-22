package assemblyai

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	// legacyModelID is selected with the deprecated speech_model parameter.
	legacyModelID = "best"

	// modelDocsURL documents model selection.
	modelDocsURL = "https://www.assemblyai.com/docs/pre-recorded-audio/select-the-speech-model"
)

// transcriptRequest is the body of a transcript submission.
type transcriptRequest struct {
	SpeechModel  string   `json:"speech_model,omitempty"`
	SpeechModels []string `json:"speech_models,omitempty"`

	AudioEndAt                  *int                             `json:"audio_end_at,omitempty"`
	AudioStartFrom              *int                             `json:"audio_start_from,omitempty"`
	AutoChapters                *bool                            `json:"auto_chapters,omitempty"`
	AutoHighlights              *bool                            `json:"auto_highlights,omitempty"`
	BoostParam                  *string                          `json:"boost_param,omitempty"`
	ContentSafety               *bool                            `json:"content_safety,omitempty"`
	ContentSafetyConfidence     *int                             `json:"content_safety_confidence,omitempty"`
	CustomSpelling              []CustomSpelling                 `json:"custom_spelling,omitempty"`
	Disfluencies                *bool                            `json:"disfluencies,omitempty"`
	EntityDetection             *bool                            `json:"entity_detection,omitempty"`
	FilterProfanity             *bool                            `json:"filter_profanity,omitempty"`
	FormatText                  *bool                            `json:"format_text,omitempty"`
	IABCategories               *bool                            `json:"iab_categories,omitempty"`
	LanguageCode                *string                          `json:"language_code,omitempty"`
	LanguageConfidenceThreshold *float64                         `json:"language_confidence_threshold,omitempty"`
	LanguageDetection           *bool                            `json:"language_detection,omitempty"`
	Multichannel                *bool                            `json:"multichannel,omitempty"`
	Punctuate                   *bool                            `json:"punctuate,omitempty"`
	RedactPII                   *bool                            `json:"redact_pii,omitempty"`
	RedactPIIAudio              *bool                            `json:"redact_pii_audio,omitempty"`
	RedactPIIAudioQuality       *string                          `json:"redact_pii_audio_quality,omitempty"`
	RedactPIIPolicies           []string                         `json:"redact_pii_policies,omitempty"`
	RedactPIISub                *string                          `json:"redact_pii_sub,omitempty"`
	SentimentAnalysis           *bool                            `json:"sentiment_analysis,omitempty"`
	SpeakerLabels               *bool                            `json:"speaker_labels,omitempty"`
	SpeakersExpected            *int                             `json:"speakers_expected,omitempty"`
	SpeechThreshold             *float64                         `json:"speech_threshold,omitempty"`
	Summarization               *bool                            `json:"summarization,omitempty"`
	SummaryModel                *string                          `json:"summary_model,omitempty"`
	SummaryType                 *string                          `json:"summary_type,omitempty"`
	WebhookAuthHeaderName       *string                          `json:"webhook_auth_header_name,omitempty"`
	WebhookAuthHeaderValue      *string                          `json:"webhook_auth_header_value,omitempty"`
	WebhookURL                  *string                          `json:"webhook_url,omitempty"`
	WordBoost                   []string                         `json:"word_boost,omitempty"`
	KeytermsPrompt              []string                         `json:"keyterms_prompt,omitempty"`
	Prompt                      *string                          `json:"prompt,omitempty"`
	Temperature                 *float64                         `json:"temperature,omitempty"`
	RemoveAudioTags             AudioTagRemoval                  `json:"remove_audio_tags,omitempty"`
	Domain                      *string                          `json:"domain,omitempty"`
	RedactPIIReturnUnredacted   *bool                            `json:"redact_pii_return_unredacted,omitempty"`
	RedactStaticEntities        map[string][]string              `json:"redact_static_entities,omitempty"`
	SpeakerOptions              *speakerOptionsRequest           `json:"speaker_options,omitempty"`
	LanguageDetectionOptions    *languageDetectionOptionsRequest `json:"language_detection_options,omitempty"`
	RedactPIIAudioOptions       *redactPIIAudioOptionsRequest    `json:"redact_pii_audio_options,omitempty"`

	AudioURL string `json:"audio_url"`
}

type speakerOptionsRequest struct {
	MinSpeakersExpected *int `json:"min_speakers_expected,omitempty"`
	MaxSpeakersExpected *int `json:"max_speakers_expected,omitempty"`
}

type languageDetectionOptionsRequest struct {
	ExpectedLanguages                []string `json:"expected_languages,omitempty"`
	FallbackLanguage                 *string  `json:"fallback_language,omitempty"`
	CodeSwitching                    *bool    `json:"code_switching,omitempty"`
	CodeSwitchingConfidenceThreshold *float64 `json:"code_switching_confidence_threshold,omitempty"`
}

type redactPIIAudioOptionsRequest struct {
	ReturnRedactedNoSpeechAudio  *bool   `json:"return_redacted_no_speech_audio,omitempty"`
	OverrideAudioRedactionMethod *string `json:"override_audio_redaction_method,omitempty"`
}

// transcriptRequest returns the submission body for the uploaded audio and
// the warnings about the model and option combinations.
func (m *TranscriptionModel) transcriptRequest(options TranscriptionOptions, audioURL string) (transcriptRequest, []model.Warning) {
	body := transcriptRequest{
		AudioEndAt:                  options.AudioEndAt,
		AudioStartFrom:              options.AudioStartFrom,
		AutoChapters:                options.AutoChapters,
		AutoHighlights:              options.AutoHighlights,
		BoostParam:                  options.BoostParam,
		ContentSafety:               options.ContentSafety,
		ContentSafetyConfidence:     options.ContentSafetyConfidence,
		CustomSpelling:              options.CustomSpelling,
		Disfluencies:                options.Disfluencies,
		EntityDetection:             options.EntityDetection,
		FilterProfanity:             options.FilterProfanity,
		FormatText:                  options.FormatText,
		IABCategories:               options.IABCategories,
		LanguageCode:                options.LanguageCode,
		LanguageConfidenceThreshold: options.LanguageConfidenceThreshold,
		LanguageDetection:           options.LanguageDetection,
		Multichannel:                options.Multichannel,
		Punctuate:                   options.Punctuate,
		RedactPII:                   options.RedactPII,
		RedactPIIAudio:              options.RedactPIIAudio,
		RedactPIIAudioQuality:       options.RedactPIIAudioQuality,
		RedactPIIPolicies:           options.RedactPIIPolicies,
		RedactPIISub:                options.RedactPIISub,
		SentimentAnalysis:           options.SentimentAnalysis,
		SpeakerLabels:               options.SpeakerLabels,
		SpeakersExpected:            options.SpeakersExpected,
		SpeechThreshold:             options.SpeechThreshold,
		Summarization:               options.Summarization,
		SummaryModel:                options.SummaryModel,
		SummaryType:                 options.SummaryType,
		WebhookAuthHeaderName:       options.WebhookAuthHeaderName,
		WebhookAuthHeaderValue:      options.WebhookAuthHeaderValue,
		WebhookURL:                  options.WebhookURL,
		WordBoost:                   options.WordBoost,
		KeytermsPrompt:              options.KeytermsPrompt,
		Prompt:                      options.Prompt,
		Temperature:                 options.Temperature,
		RemoveAudioTags:             options.RemoveAudioTags,
		Domain:                      options.Domain,
		RedactPIIReturnUnredacted:   options.RedactPIIReturnUnredacted,
		RedactStaticEntities:        options.RedactStaticEntities,
		SpeakerOptions:              speakerOptions(options.SpeakerOptions),
		LanguageDetectionOptions:    languageDetectionOptions(options.LanguageDetectionOptions),
		RedactPIIAudioOptions:       redactPIIAudioOptions(options.RedactPIIAudioOptions),
		AudioURL:                    audioURL,
	}

	var warnings []model.Warning

	if m.modelID == legacyModelID {
		body.SpeechModel = m.modelID
		warnings = append(warnings, model.Warning{
			Type:    model.WarningDeprecated,
			Setting: fmt.Sprintf("model '%s'", m.modelID),
			Message: "The 'best' model is a legacy AssemblyAI model. Use 'universal-3-5-pro' instead. See documentation: " + modelDocsURL,
		})
	} else {
		body.SpeechModels = []string{
			m.modelID,
		}
		warnings = append(warnings, modelNudge(m.modelID)...)
	}

	warnings = append(warnings, optionWarnings(options)...)

	return body, warnings
}

// modelNudge points older universal models at the latest flagship model.
// Both models still work, so this is informational.
func modelNudge(modelID string) []model.Warning {
	var message string

	switch modelID {
	case "universal-3-pro":
		message = "'universal-3-5-pro' is AssemblyAI's latest flagship model and is set to replace 'universal-3-pro'. See " + modelDocsURL
	case "universal-2":
		message = "'universal-3-5-pro' is AssemblyAI's latest flagship model. See " + modelDocsURL
	default:
		return nil
	}

	return []model.Warning{
		otherWarning(message),
	}
}

// optionWarnings reports deprecated options and options whose prerequisite
// is missing. The request is sent unchanged.
func optionWarnings(options TranscriptionOptions) []model.Warning {
	var warnings []model.Warning

	var deprecated []string
	if options.WordBoost != nil {
		deprecated = append(deprecated, "wordBoost")
	}

	if options.BoostParam != nil {
		deprecated = append(deprecated, "boostParam")
	}

	if len(deprecated) > 0 {
		warnings = append(warnings, model.Warning{
			Type:    model.WarningDeprecated,
			Setting: strings.Join(deprecated, ", "),
			Message: "'wordBoost' and 'boostParam' are deprecated and are rejected by 'universal-3-pro' / 'universal-3-5-pro' and 'slam-1'. Use 'keytermsPrompt' instead.",
		})
	}

	redactPII := options.RedactPII != nil && *options.RedactPII
	if (options.RedactPIIReturnUnredacted != nil || options.RedactStaticEntities != nil) && !redactPII {
		warnings = append(warnings, otherWarning(
			"'redactPiiReturnUnredacted' and 'redactStaticEntities' require 'redactPii' to be enabled; AssemblyAI rejects the request otherwise.",
		))
	}

	redactPIIAudio := options.RedactPIIAudio != nil && *options.RedactPIIAudio
	if options.RedactPIIAudioOptions != nil && !redactPIIAudio {
		warnings = append(warnings, otherWarning(
			"'redactPiiAudioOptions' only applies when 'redactPiiAudio' is enabled; it is otherwise ignored.",
		))
	}

	languageDetection := options.LanguageDetection != nil && *options.LanguageDetection
	if options.LanguageCode != nil && languageDetection {
		warnings = append(warnings, otherWarning(
			"'languageDetection' cannot be combined with an explicit 'languageCode'; AssemblyAI rejects requests that set both.",
		))
	}

	return warnings
}

func otherWarning(message string) model.Warning {
	return model.Warning{
		Type:    model.WarningOther,
		Message: message,
	}
}

func speakerOptions(options *SpeakerOptions) *speakerOptionsRequest {
	if options == nil {
		return nil
	}

	return &speakerOptionsRequest{
		MinSpeakersExpected: options.MinSpeakersExpected,
		MaxSpeakersExpected: options.MaxSpeakersExpected,
	}
}

func languageDetectionOptions(options *LanguageDetectionOptions) *languageDetectionOptionsRequest {
	if options == nil {
		return nil
	}

	return &languageDetectionOptionsRequest{
		ExpectedLanguages:                options.ExpectedLanguages,
		FallbackLanguage:                 options.FallbackLanguage,
		CodeSwitching:                    options.CodeSwitching,
		CodeSwitchingConfidenceThreshold: options.CodeSwitchingConfidenceThreshold,
	}
}

func redactPIIAudioOptions(options *RedactPIIAudioOptions) *redactPIIAudioOptionsRequest {
	if options == nil {
		return nil
	}

	return &redactPIIAudioOptionsRequest{
		ReturnRedactedNoSpeechAudio:  options.ReturnRedactedNoSpeechAudio,
		OverrideAudioRedactionMethod: options.OverrideAudioRedactionMethod,
	}
}
