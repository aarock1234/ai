package gladia

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// CallbackMethod is the HTTP method of a job callback.
type CallbackMethod string

// Callback methods.
const (
	CallbackPost CallbackMethod = "POST"
	CallbackPut  CallbackMethod = "PUT"
)

// SubtitleFormat is a subtitle file format.
type SubtitleFormat string

// Subtitle formats.
const (
	SubtitleSRT SubtitleFormat = "srt"
	SubtitleVTT SubtitleFormat = "vtt"
)

// SubtitleStyle is a subtitle layout style.
type SubtitleStyle string

// Subtitle styles.
const (
	SubtitleDefault    SubtitleStyle = "default"
	SubtitleCompliance SubtitleStyle = "compliance"
)

// TranslationModel is the translation quality.
type TranslationModel string

// Translation models.
const (
	TranslationBase     TranslationModel = "base"
	TranslationEnhanced TranslationModel = "enhanced"
)

// SummaryType is the style of a summary.
type SummaryType string

// Summary types.
const (
	SummaryGeneral      SummaryType = "general"
	SummaryBulletPoints SummaryType = "bullet_points"
	SummaryConcise      SummaryType = "concise"
)

// TranscriptionOptions are the provider options of the transcription model.
type TranscriptionOptions struct {
	ContextPrompt *string `json:"contextPrompt,omitempty"`

	// CustomVocabulary is a JSON boolean or array, sent unchanged.
	CustomVocabulary json.RawMessage `json:"customVocabulary,omitempty"`

	CustomVocabularyConfig *CustomVocabularyConfig `json:"customVocabularyConfig,omitempty"`
	DetectLanguage         *bool                   `json:"detectLanguage,omitempty"`
	EnableCodeSwitching    *bool                   `json:"enableCodeSwitching,omitempty"`
	CodeSwitchingConfig    *CodeSwitchingConfig    `json:"codeSwitchingConfig,omitempty"`
	Language               *string                 `json:"language,omitempty"`
	Callback               *bool                   `json:"callback,omitempty"`
	CallbackConfig         *CallbackConfig         `json:"callbackConfig,omitempty"`
	Subtitles              *bool                   `json:"subtitles,omitempty"`
	SubtitlesConfig        *SubtitlesConfig        `json:"subtitlesConfig,omitempty"`
	Diarization            *bool                   `json:"diarization,omitempty"`
	DiarizationConfig      *DiarizationConfig      `json:"diarizationConfig,omitempty"`
	Translation            *bool                   `json:"translation,omitempty"`
	TranslationConfig      *TranslationConfig      `json:"translationConfig,omitempty"`
	Summarization          *bool                   `json:"summarization,omitempty"`
	SummarizationConfig    *SummarizationConfig    `json:"summarizationConfig,omitempty"`
	Moderation             *bool                   `json:"moderation,omitempty"`
	NamedEntityRecognition *bool                   `json:"namedEntityRecognition,omitempty"`
	Chapterization         *bool                   `json:"chapterization,omitempty"`
	NameConsistency        *bool                   `json:"nameConsistency,omitempty"`
	CustomSpelling         *bool                   `json:"customSpelling,omitempty"`
	CustomSpellingConfig   *CustomSpellingConfig   `json:"customSpellingConfig,omitempty"`

	StructuredDataExtraction       *bool                           `json:"structuredDataExtraction,omitempty"`
	StructuredDataExtractionConfig *StructuredDataExtractionConfig `json:"structuredDataExtractionConfig,omitempty"`

	SentimentAnalysis *bool             `json:"sentimentAnalysis,omitempty"`
	AudioToLLM        *bool             `json:"audioToLlm,omitempty"`
	AudioToLLMConfig  *AudioToLLMConfig `json:"audioToLlmConfig,omitempty"`

	// CustomMetadata is a JSON object, sent unchanged.
	CustomMetadata json.RawMessage `json:"customMetadata,omitempty"`

	Sentences           *bool `json:"sentences,omitempty"`
	DisplayMode         *bool `json:"displayMode,omitempty"`
	PunctuationEnhanced *bool `json:"punctuationEnhanced,omitempty"`
}

// CustomVocabularyConfig lists vocabulary that guides recognition.
type CustomVocabularyConfig struct {
	Vocabulary       []VocabularyItem `json:"vocabulary"`
	DefaultIntensity *float64         `json:"defaultIntensity,omitempty"`
}

// VocabularyItem is a vocabulary term. A JSON string decodes as a term with
// only a Value.
type VocabularyItem struct {
	Value          string   `json:"value"`
	Intensity      *float64 `json:"intensity,omitempty"`
	Pronunciations []string `json:"pronunciations,omitempty"`
	Language       *string  `json:"language,omitempty"`
}

// vocabularyItemFields decodes the object form of a VocabularyItem without
// recursing into its UnmarshalJSON.
type vocabularyItemFields VocabularyItem

// UnmarshalJSON accepts a JSON string or an object.
func (v *VocabularyItem) UnmarshalJSON(data []byte) error {
	if bytes.HasPrefix(bytes.TrimSpace(data), []byte(`"`)) {
		var value string
		if err := json.Unmarshal(data, &value); err != nil {
			return err
		}

		*v = VocabularyItem{
			Value: value,
		}

		return nil
	}

	var fields vocabularyItemFields
	if err := json.Unmarshal(data, &fields); err != nil {
		return fmt.Errorf("vocabulary item must be a string or an object: %w", err)
	}

	*v = VocabularyItem(fields)

	return nil
}

// CodeSwitchingConfig restricts code switching to languages.
type CodeSwitchingConfig struct {
	Languages []string `json:"languages,omitempty"`
}

// CallbackConfig receives the result when the job finishes.
type CallbackConfig struct {
	URL    string         `json:"url"`
	Method CallbackMethod `json:"method,omitempty"`
}

// SubtitlesConfig configures generated subtitles.
type SubtitlesConfig struct {
	Formats                 []SubtitleFormat `json:"formats,omitempty"`
	MinimumDuration         *float64         `json:"minimumDuration,omitempty"`
	MaximumDuration         *float64         `json:"maximumDuration,omitempty"`
	MaximumCharactersPerRow *float64         `json:"maximumCharactersPerRow,omitempty"`
	MaximumRowsPerCaption   *float64         `json:"maximumRowsPerCaption,omitempty"`
	Style                   SubtitleStyle    `json:"style,omitempty"`
}

// DiarizationConfig configures speaker identification.
type DiarizationConfig struct {
	NumberOfSpeakers *float64 `json:"numberOfSpeakers,omitempty"`
	MinSpeakers      *float64 `json:"minSpeakers,omitempty"`
	MaxSpeakers      *float64 `json:"maxSpeakers,omitempty"`
	Enhanced         *bool    `json:"enhanced,omitempty"`
}

// TranslationConfig configures translation.
type TranslationConfig struct {
	TargetLanguages         []string         `json:"targetLanguages"`
	Model                   TranslationModel `json:"model,omitempty"`
	MatchOriginalUtterances *bool            `json:"matchOriginalUtterances,omitempty"`
}

// SummarizationConfig configures summarization.
type SummarizationConfig struct {
	Type SummaryType `json:"type,omitempty"`
}

// CustomSpellingConfig maps spellings to the variants they replace.
type CustomSpellingConfig struct {
	SpellingDictionary map[string][]string `json:"spellingDictionary"`
}

// StructuredDataExtractionConfig lists the classes of data to extract.
type StructuredDataExtractionConfig struct {
	Classes []string `json:"classes"`
}

// AudioToLLMConfig lists prompts to run against the audio.
type AudioToLLMConfig struct {
	Prompts []string `json:"prompts"`
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
	if !isAbsent(o.CustomVocabulary) && !isBooleanOrArray(o.CustomVocabulary) {
		return invalidOption("customVocabulary must be a boolean or an array")
	}

	if !isAbsent(o.CustomMetadata) && !jsonobject.IsObject(o.CustomMetadata) {
		return invalidOption("customMetadata must be an object")
	}

	if o.CallbackConfig != nil {
		switch o.CallbackConfig.Method {
		case "", CallbackPost, CallbackPut:
		default:
			return invalidOption(fmt.Sprintf("callback method %q", o.CallbackConfig.Method))
		}
	}

	if o.SubtitlesConfig != nil {
		if err := o.SubtitlesConfig.validate(); err != nil {
			return err
		}
	}

	if o.TranslationConfig != nil {
		switch o.TranslationConfig.Model {
		case "", TranslationBase, TranslationEnhanced:
		default:
			return invalidOption(fmt.Sprintf("translation model %q", o.TranslationConfig.Model))
		}
	}

	if o.SummarizationConfig != nil {
		switch o.SummarizationConfig.Type {
		case "", SummaryGeneral, SummaryBulletPoints, SummaryConcise:
		default:
			return invalidOption(fmt.Sprintf("summary type %q", o.SummarizationConfig.Type))
		}
	}

	return nil
}

func (c SubtitlesConfig) validate() error {
	for _, format := range c.Formats {
		if format != SubtitleSRT && format != SubtitleVTT {
			return invalidOption(fmt.Sprintf("subtitle format %q", format))
		}
	}

	switch c.Style {
	case "", SubtitleDefault, SubtitleCompliance:
		return nil
	default:
		return invalidOption(fmt.Sprintf("subtitle style %q", c.Style))
	}
}

func invalidOption(details string) error {
	return fmt.Errorf("%w: gladia %s", model.ErrInvalidArgument, details)
}

// isAbsent reports whether a raw value is missing or null.
func isAbsent(value json.RawMessage) bool {
	trimmed := bytes.TrimSpace(value)

	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}

func isBooleanOrArray(value json.RawMessage) bool {
	trimmed := bytes.TrimSpace(value)

	if bytes.Equal(trimmed, []byte("true")) || bytes.Equal(trimmed, []byte("false")) {
		return true
	}

	return bytes.HasPrefix(trimmed, []byte("[")) && json.Valid(trimmed)
}
