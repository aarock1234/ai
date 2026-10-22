package revai

// defaultLanguage is the language sent when the options leave it unset.
const defaultLanguage = "en"

// jobConfig is the JSON config field of a job submission.
type jobConfig struct {
	Transcriber            string                      `json:"transcriber"`
	Metadata               *string                     `json:"metadata,omitempty"`
	NotificationConfig     *NotificationConfig         `json:"notification_config,omitempty"`
	DeleteAfterSeconds     *float64                    `json:"delete_after_seconds,omitempty"`
	Verbatim               *bool                       `json:"verbatim,omitempty"`
	Rush                   *bool                       `json:"rush,omitempty"`
	TestMode               *bool                       `json:"test_mode,omitempty"`
	SegmentsToTranscribe   []Segment                   `json:"segments_to_transcribe,omitempty"`
	SpeakerNames           []SpeakerName               `json:"speaker_names,omitempty"`
	SkipDiarization        *bool                       `json:"skip_diarization,omitempty"`
	SkipPostprocessing     *bool                       `json:"skip_postprocessing,omitempty"`
	SkipPunctuation        *bool                       `json:"skip_punctuation,omitempty"`
	RemoveDisfluencies     *bool                       `json:"remove_disfluencies,omitempty"`
	RemoveAtmospherics     *bool                       `json:"remove_atmospherics,omitempty"`
	FilterProfanity        *bool                       `json:"filter_profanity,omitempty"`
	SpeakerChannelsCount   *float64                    `json:"speaker_channels_count,omitempty"`
	SpeakersCount          *float64                    `json:"speakers_count,omitempty"`
	DiarizationType        Tier                        `json:"diarization_type,omitempty"`
	CustomVocabularyID     *string                     `json:"custom_vocabulary_id,omitempty"`
	CustomVocabularies     []CustomVocabulary          `json:"custom_vocabularies,omitempty"`
	StrictCustomVocabulary *bool                       `json:"strict_custom_vocabulary,omitempty"`
	SummarizationConfig    *summarizationConfigRequest `json:"summarization_config,omitempty"`
	TranslationConfig      *translationConfigRequest   `json:"translation_config,omitempty"`
	Language               *string                     `json:"language,omitempty"`
	ForcedAlignment        *bool                       `json:"forced_alignment,omitempty"`
}

type summarizationConfigRequest struct {
	Model  Tier        `json:"model"`
	Type   SummaryType `json:"type"`
	Prompt *string     `json:"prompt,omitempty"`
}

type translationConfigRequest struct {
	TargetLanguages []TargetLanguage `json:"target_languages"`
	Model           Tier             `json:"model"`
}

// newJobConfig returns the config for the transcriber. When the provider
// options are present, their unset settings take the upstream defaults.
func newJobConfig(transcriber string, options TranscriptionOptions, present bool) jobConfig {
	config := jobConfig{
		Transcriber: transcriber,
	}

	if !present {
		return config
	}

	config.Metadata = options.Metadata
	config.NotificationConfig = options.NotificationConfig
	config.DeleteAfterSeconds = options.DeleteAfterSeconds
	config.Verbatim = options.Verbatim
	config.Rush = orFalse(options.Rush)
	config.TestMode = orFalse(options.TestMode)
	config.SegmentsToTranscribe = options.SegmentsToTranscribe
	config.SpeakerNames = options.SpeakerNames
	config.SkipDiarization = orFalse(options.SkipDiarization)
	config.SkipPostprocessing = orFalse(options.SkipPostprocessing)
	config.SkipPunctuation = orFalse(options.SkipPunctuation)
	config.RemoveDisfluencies = orFalse(options.RemoveDisfluencies)
	config.RemoveAtmospherics = orFalse(options.RemoveAtmospherics)
	config.FilterProfanity = orFalse(options.FilterProfanity)
	config.SpeakerChannelsCount = options.SpeakerChannelsCount
	config.SpeakersCount = options.SpeakersCount
	config.DiarizationType = tierOrStandard(options.DiarizationType)
	config.CustomVocabularyID = options.CustomVocabularyID
	config.CustomVocabularies = options.CustomVocabularies
	config.StrictCustomVocabulary = options.StrictCustomVocabulary
	config.SummarizationConfig = summarizationConfig(options.SummarizationConfig)
	config.TranslationConfig = translationConfig(options.TranslationConfig)
	config.Language = options.Language
	config.ForcedAlignment = orFalse(options.ForcedAlignment)

	if config.Language == nil {
		config.Language = new(defaultLanguage)
	}

	return config
}

func summarizationConfig(options *SummarizationConfig) *summarizationConfigRequest {
	if options == nil {
		return nil
	}

	summaryType := options.Type
	if summaryType == "" {
		summaryType = SummaryParagraph
	}

	return &summarizationConfigRequest{
		Model:  tierOrStandard(options.Model),
		Type:   summaryType,
		Prompt: options.Prompt,
	}
}

func translationConfig(options *TranslationConfig) *translationConfigRequest {
	if options == nil {
		return nil
	}

	return &translationConfigRequest{
		TargetLanguages: options.TargetLanguages,
		Model:           tierOrStandard(options.Model),
	}
}

// orFalse returns value, or a pointer to false when it is unset.
func orFalse(value *bool) *bool {
	if value != nil {
		return value
	}

	return new(false)
}

func tierOrStandard(tier Tier) Tier {
	if tier == "" {
		return TierStandard
	}

	return tier
}
