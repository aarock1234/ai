package gladia

import (
	"encoding/json"
)

// jobRequest is the body that starts a pre-recorded transcription job.
type jobRequest struct {
	ContextPrompt                  *string                                `json:"context_prompt,omitempty"`
	CustomVocabulary               json.RawMessage                        `json:"custom_vocabulary,omitempty"`
	DetectLanguage                 *bool                                  `json:"detect_language,omitempty"`
	EnableCodeSwitching            *bool                                  `json:"enable_code_switching,omitempty"`
	Language                       *string                                `json:"language,omitempty"`
	Callback                       *bool                                  `json:"callback,omitempty"`
	Subtitles                      *bool                                  `json:"subtitles,omitempty"`
	Diarization                    *bool                                  `json:"diarization,omitempty"`
	Translation                    *bool                                  `json:"translation,omitempty"`
	Summarization                  *bool                                  `json:"summarization,omitempty"`
	Moderation                     *bool                                  `json:"moderation,omitempty"`
	NamedEntityRecognition         *bool                                  `json:"named_entity_recognition,omitempty"`
	Chapterization                 *bool                                  `json:"chapterization,omitempty"`
	NameConsistency                *bool                                  `json:"name_consistency,omitempty"`
	CustomSpelling                 *bool                                  `json:"custom_spelling,omitempty"`
	StructuredDataExtraction       *bool                                  `json:"structured_data_extraction,omitempty"`
	StructuredDataExtractionConfig *structuredDataExtractionConfigRequest `json:"structured_data_extraction_config,omitempty"`
	SentimentAnalysis              *bool                                  `json:"sentiment_analysis,omitempty"`
	AudioToLLM                     *bool                                  `json:"audio_to_llm,omitempty"`
	AudioToLLMConfig               *audioToLLMConfigRequest               `json:"audio_to_llm_config,omitempty"`
	CustomMetadata                 json.RawMessage                        `json:"custom_metadata,omitempty"`
	Sentences                      *bool                                  `json:"sentences,omitempty"`
	DisplayMode                    *bool                                  `json:"display_mode,omitempty"`
	PunctuationEnhanced            *bool                                  `json:"punctuation_enhanced,omitempty"`
	CustomVocabularyConfig         *customVocabularyConfigRequest         `json:"custom_vocabulary_config,omitempty"`
	CodeSwitchingConfig            *codeSwitchingConfigRequest            `json:"code_switching_config,omitempty"`
	CallbackConfig                 *callbackConfigRequest                 `json:"callback_config,omitempty"`
	SubtitlesConfig                *subtitlesConfigRequest                `json:"subtitles_config,omitempty"`
	DiarizationConfig              *diarizationConfigRequest              `json:"diarization_config,omitempty"`
	TranslationConfig              *translationConfigRequest              `json:"translation_config,omitempty"`
	SummarizationConfig            *summarizationConfigRequest            `json:"summarization_config,omitempty"`
	CustomSpellingConfig           *customSpellingConfigRequest           `json:"custom_spelling_config,omitempty"`
	AudioURL                       string                                 `json:"audio_url"`
}

type customVocabularyConfigRequest struct {
	Vocabulary       []vocabularyItemRequest `json:"vocabulary"`
	DefaultIntensity *float64                `json:"default_intensity,omitempty"`
}

// vocabularyItemRequest is a vocabulary term. A term with only a value is
// sent as a plain string.
type vocabularyItemRequest struct {
	Value          string   `json:"value"`
	Intensity      *float64 `json:"intensity,omitempty"`
	Pronunciations []string `json:"pronunciations,omitempty"`
	Language       *string  `json:"language,omitempty"`
}

// vocabularyItemObject encodes the object form without recursing into
// MarshalJSON.
type vocabularyItemObject vocabularyItemRequest

// MarshalJSON sends a term with only a value as a string.
func (v vocabularyItemRequest) MarshalJSON() ([]byte, error) {
	if v.Intensity == nil && v.Pronunciations == nil && v.Language == nil {
		return json.Marshal(v.Value)
	}

	return json.Marshal(vocabularyItemObject(v))
}

type codeSwitchingConfigRequest struct {
	Languages []string `json:"languages,omitempty"`
}

type callbackConfigRequest struct {
	URL    string         `json:"url"`
	Method CallbackMethod `json:"method,omitempty"`
}

type subtitlesConfigRequest struct {
	Formats                 []SubtitleFormat `json:"formats,omitempty"`
	MinimumDuration         *float64         `json:"minimum_duration,omitempty"`
	MaximumDuration         *float64         `json:"maximum_duration,omitempty"`
	MaximumCharactersPerRow *float64         `json:"maximum_characters_per_row,omitempty"`
	MaximumRowsPerCaption   *float64         `json:"maximum_rows_per_caption,omitempty"`
	Style                   SubtitleStyle    `json:"style,omitempty"`
}

type diarizationConfigRequest struct {
	NumberOfSpeakers *float64 `json:"number_of_speakers,omitempty"`
	MinSpeakers      *float64 `json:"min_speakers,omitempty"`
	MaxSpeakers      *float64 `json:"max_speakers,omitempty"`
	Enhanced         *bool    `json:"enhanced,omitempty"`
}

type translationConfigRequest struct {
	TargetLanguages         []string         `json:"target_languages"`
	Model                   TranslationModel `json:"model,omitempty"`
	MatchOriginalUtterances *bool            `json:"match_original_utterances,omitempty"`
}

type summarizationConfigRequest struct {
	Type SummaryType `json:"type,omitempty"`
}

type customSpellingConfigRequest struct {
	SpellingDictionary map[string][]string `json:"spelling_dictionary"`
}

type structuredDataExtractionConfigRequest struct {
	Classes []string `json:"classes"`
}

type audioToLLMConfigRequest struct {
	Prompts []string `json:"prompts"`
}

// newJobRequest converts the provider options into the job body for the
// uploaded audio.
func newJobRequest(options TranscriptionOptions, audioURL string) jobRequest {
	return jobRequest{
		ContextPrompt:                  options.ContextPrompt,
		CustomVocabulary:               presentJSON(options.CustomVocabulary),
		DetectLanguage:                 options.DetectLanguage,
		EnableCodeSwitching:            options.EnableCodeSwitching,
		Language:                       options.Language,
		Callback:                       options.Callback,
		Subtitles:                      options.Subtitles,
		Diarization:                    options.Diarization,
		Translation:                    options.Translation,
		Summarization:                  options.Summarization,
		Moderation:                     options.Moderation,
		NamedEntityRecognition:         options.NamedEntityRecognition,
		Chapterization:                 options.Chapterization,
		NameConsistency:                options.NameConsistency,
		CustomSpelling:                 options.CustomSpelling,
		StructuredDataExtraction:       options.StructuredDataExtraction,
		StructuredDataExtractionConfig: structuredDataExtractionConfig(options.StructuredDataExtractionConfig),
		SentimentAnalysis:              options.SentimentAnalysis,
		AudioToLLM:                     options.AudioToLLM,
		AudioToLLMConfig:               audioToLLMConfig(options.AudioToLLMConfig),
		CustomMetadata:                 presentJSON(options.CustomMetadata),
		Sentences:                      options.Sentences,
		DisplayMode:                    options.DisplayMode,
		PunctuationEnhanced:            options.PunctuationEnhanced,
		CustomVocabularyConfig:         customVocabularyConfig(options.CustomVocabularyConfig),
		CodeSwitchingConfig:            codeSwitchingConfig(options.CodeSwitchingConfig),
		CallbackConfig:                 callbackConfig(options.CallbackConfig),
		SubtitlesConfig:                subtitlesConfig(options.SubtitlesConfig),
		DiarizationConfig:              diarizationConfig(options.DiarizationConfig),
		TranslationConfig:              translationConfig(options.TranslationConfig),
		SummarizationConfig:            summarizationConfig(options.SummarizationConfig),
		CustomSpellingConfig:           customSpellingConfig(options.CustomSpellingConfig),
		AudioURL:                       audioURL,
	}
}

// presentJSON returns value, or nil when it is absent or null so that the
// field is omitted.
func presentJSON(value json.RawMessage) json.RawMessage {
	if isAbsent(value) {
		return nil
	}

	return value
}

func customVocabularyConfig(config *CustomVocabularyConfig) *customVocabularyConfigRequest {
	if config == nil {
		return nil
	}

	vocabulary := make([]vocabularyItemRequest, 0, len(config.Vocabulary))
	for _, item := range config.Vocabulary {
		vocabulary = append(vocabulary, vocabularyItemRequest(item))
	}

	return &customVocabularyConfigRequest{
		Vocabulary:       vocabulary,
		DefaultIntensity: config.DefaultIntensity,
	}
}

func structuredDataExtractionConfig(config *StructuredDataExtractionConfig) *structuredDataExtractionConfigRequest {
	if config == nil {
		return nil
	}

	return &structuredDataExtractionConfigRequest{
		Classes: config.Classes,
	}
}

func audioToLLMConfig(config *AudioToLLMConfig) *audioToLLMConfigRequest {
	if config == nil {
		return nil
	}

	return &audioToLLMConfigRequest{
		Prompts: config.Prompts,
	}
}

func codeSwitchingConfig(config *CodeSwitchingConfig) *codeSwitchingConfigRequest {
	if config == nil {
		return nil
	}

	return &codeSwitchingConfigRequest{
		Languages: config.Languages,
	}
}

func callbackConfig(config *CallbackConfig) *callbackConfigRequest {
	if config == nil {
		return nil
	}

	return &callbackConfigRequest{
		URL:    config.URL,
		Method: config.Method,
	}
}

func subtitlesConfig(config *SubtitlesConfig) *subtitlesConfigRequest {
	if config == nil {
		return nil
	}

	return &subtitlesConfigRequest{
		Formats:                 config.Formats,
		MinimumDuration:         config.MinimumDuration,
		MaximumDuration:         config.MaximumDuration,
		MaximumCharactersPerRow: config.MaximumCharactersPerRow,
		MaximumRowsPerCaption:   config.MaximumRowsPerCaption,
		Style:                   config.Style,
	}
}

func diarizationConfig(config *DiarizationConfig) *diarizationConfigRequest {
	if config == nil {
		return nil
	}

	return &diarizationConfigRequest{
		NumberOfSpeakers: config.NumberOfSpeakers,
		MinSpeakers:      config.MinSpeakers,
		MaxSpeakers:      config.MaxSpeakers,
		Enhanced:         config.Enhanced,
	}
}

func translationConfig(config *TranslationConfig) *translationConfigRequest {
	if config == nil {
		return nil
	}

	return &translationConfigRequest{
		TargetLanguages:         config.TargetLanguages,
		Model:                   config.Model,
		MatchOriginalUtterances: config.MatchOriginalUtterances,
	}
}

func summarizationConfig(config *SummarizationConfig) *summarizationConfigRequest {
	if config == nil {
		return nil
	}

	return &summarizationConfigRequest{
		Type: config.Type,
	}
}

func customSpellingConfig(config *CustomSpellingConfig) *customSpellingConfigRequest {
	if config == nil {
		return nil
	}

	return &customSpellingConfigRequest{
		SpellingDictionary: config.SpellingDictionary,
	}
}
