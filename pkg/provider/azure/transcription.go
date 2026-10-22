package azure

import (
	"context"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

var _ model.StreamingTranscriptionModel = (*TranscriptionModel)(nil)

// maiStreamingTranscribeModel is the lowercase ID of the MAI model that only
// transcribes streams.
const maiStreamingTranscribeModel = "mai-transcribe-2-streaming"

// maiTranscribeModel is an Azure Speech MAI-Transcribe model.
type maiTranscribeModel struct {
	name string

	// supportsTimestamps is false for models that only accept Azure's default
	// timestamps ("none").
	supportsTimestamps bool
}

// maiTranscribeModels maps lowercase model IDs to Azure Speech models.
var maiTranscribeModels = map[string]maiTranscribeModel{
	"mai-transcribe-2": {
		name:               "MAI-Transcribe-2",
		supportsTimestamps: true,
	},
	"mai-transcribe-1.5": {
		name:               "MAI-Transcribe-1.5",
		supportsTimestamps: false,
	},
}

// TranscriptionAPI selects the API of a transcription call.
type TranscriptionAPI string

// Transcription APIs.
const (
	TranscriptionAPIOpenAI TranscriptionAPI = "openai"
	TranscriptionAPISpeech TranscriptionAPI = "speech"
	TranscriptionAPIMAI    TranscriptionAPI = "mai"
)

// TranscriptionTimestamps is the timing granularity of Azure Speech
// transcription.
type TranscriptionTimestamps string

// Timestamp granularities.
const (
	TranscriptionTimestampsWord    TranscriptionTimestamps = "word"
	TranscriptionTimestampsSegment TranscriptionTimestamps = "segment"
	TranscriptionTimestampsNone    TranscriptionTimestamps = "none"
)

// TranscribeStyle selects verbatim or cleaned-up transcripts.
type TranscribeStyle string

// Transcribe styles.
const (
	// TranscribeStyleVerbatim keeps fillers and false starts.
	TranscribeStyleVerbatim TranscribeStyle = "verbatim"

	// TranscribeStyleClean removes fillers and false starts.
	TranscribeStyleClean TranscribeStyle = "clean"
)

// TranscriptionOptions are the Azure options of a transcription call, read
// from ProviderOptions["azure"]. OpenAI transcription options stay in the
// "openai" namespace.
type TranscriptionOptions struct {
	// API selects the API. The default is Azure Speech for MAI-Transcribe
	// models, the MAI realtime API for MAI-Transcribe-2-Streaming and the
	// OpenAI transcription API otherwise.
	API TranscriptionAPI `json:"api,omitempty"`

	// Timestamps is the timing granularity of Azure Speech transcription.
	// The default is "segment", so that results include timed segments.
	Timestamps TranscriptionTimestamps `json:"timestamps,omitempty"`

	// TranscribeStyle defaults to "verbatim".
	TranscribeStyle TranscribeStyle `json:"transcribeStyle,omitempty"`

	// Locales forces a single language, such as ["en"]. Omit it for
	// automatic language detection and code switching.
	Locales []string `json:"locales,omitempty"`

	// Diarization enables speaker diarization. Speaker IDs are available in
	// the provider metadata.
	Diarization *TranscriptionDiarization `json:"diarization,omitempty"`

	// PhraseList biases recognition toward names and domain terminology.
	PhraseList *TranscriptionPhraseList `json:"phraseList,omitempty"`

	// Language is a language hint for MAI streaming transcription, such as
	// "en".
	Language string `json:"language,omitempty"`
}

// TranscriptionDiarization configures speaker diarization.
type TranscriptionDiarization struct {
	Enabled bool `json:"enabled"`
}

// TranscriptionPhraseList lists phrases for keyword biasing.
type TranscriptionPhraseList struct {
	Phrases []string `json:"phrases"`
}

func (o TranscriptionOptions) validate() error {
	switch o.API {
	case "", TranscriptionAPIOpenAI, TranscriptionAPISpeech, TranscriptionAPIMAI:
	default:
		return fmt.Errorf("%w: providerOptions.azure.api must be %q, %q or %q", model.ErrInvalidArgument, TranscriptionAPIOpenAI, TranscriptionAPISpeech, TranscriptionAPIMAI)
	}

	switch o.Timestamps {
	case "", TranscriptionTimestampsWord, TranscriptionTimestampsSegment, TranscriptionTimestampsNone:
	default:
		return fmt.Errorf("%w: providerOptions.azure.timestamps must be %q, %q or %q", model.ErrInvalidArgument, TranscriptionTimestampsWord, TranscriptionTimestampsSegment, TranscriptionTimestampsNone)
	}

	switch o.TranscribeStyle {
	case "", TranscribeStyleVerbatim, TranscribeStyleClean:
	default:
		return fmt.Errorf("%w: providerOptions.azure.transcribeStyle must be %q or %q", model.ErrInvalidArgument, TranscribeStyleVerbatim, TranscribeStyleClean)
	}

	if o.Locales != nil && len(o.Locales) != 1 {
		return fmt.Errorf("%w: providerOptions.azure.locales must contain exactly one locale", model.ErrInvalidArgument)
	}

	return nil
}

// speechOptionNames returns the names of the Azure Speech options that are
// set.
func (o TranscriptionOptions) speechOptionNames() []string {
	var names []string

	if o.Timestamps != "" {
		names = append(names, "timestamps")
	}

	if o.TranscribeStyle != "" {
		names = append(names, "transcribeStyle")
	}

	if o.Locales != nil {
		names = append(names, "locales")
	}

	if o.Diarization != nil {
		names = append(names, "diarization")
	}

	if o.PhraseList != nil {
		names = append(names, "phraseList")
	}

	return names
}

// maiOptionNames returns the names of the MAI streaming options that are
// set.
func (o TranscriptionOptions) maiOptionNames() []string {
	if o.Language == "" {
		return nil
	}

	return []string{
		"language",
	}
}

// TranscriptionModel is an Azure transcription deployment. It selects Azure
// Speech, the MAI realtime API or the OpenAI transcription API for each
// call. Azure Speech only transcribes files and the MAI realtime API only
// transcribes streams.
type TranscriptionModel struct {
	modelID  string
	provider string
	openai   *openai.TranscriptionModel
	speech   *speechTranscriptionModel
}

// Provider returns "azure.transcription".
func (m *TranscriptionModel) Provider() string {
	return m.provider
}

// ModelID returns the deployment ID.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

// Transcribe converts audio into text with the selected API. Options of
// another API are reported as unsupported.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	azureOptions, _, err := model.ProviderValue[TranscriptionOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := azureOptions.validate(); err != nil {
		return nil, err
	}

	switch m.selectedAPI(azureOptions.API) {
	case TranscriptionAPIMAI:
		return nil, fmt.Errorf("%w: file transcription with %s (use streaming transcription)", model.ErrUnsupported, m.modelID)

	case TranscriptionAPISpeech:
		result, err := m.speech.transcribe(ctx, options, azureOptions)
		if err != nil {
			return nil, err
		}

		result.Warnings = append(result.Warnings, unsupportedOptionWarnings(azureOptions.maiOptionNames(), maiStreamingDetails)...)

		return result, nil

	default:
		result, err := m.openai.Transcribe(ctx, options)
		if err != nil {
			return nil, err
		}

		result.Warnings = append(result.Warnings, unsupportedOptionWarnings(azureOptions.speechOptionNames(), speechOnlyDetails)...)
		result.Warnings = append(result.Warnings, unsupportedOptionWarnings(azureOptions.maiOptionNames(), maiStreamingDetails)...)

		return result, nil
	}
}

// StreamTranscribe transcribes live audio with the selected API: the MAI
// realtime API, or OpenAI realtime transcription models. Azure Speech does
// not stream, and Azure Speech options are reported as unsupported by MAI.
func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, options model.TranscriptionStreamOptions) (*model.TranscriptionStreamResult, error) {
	azureOptions, _, err := model.ProviderValue[TranscriptionOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := azureOptions.validate(); err != nil {
		return nil, err
	}

	switch m.selectedAPI(azureOptions.API) {
	case TranscriptionAPISpeech:
		return nil, fmt.Errorf("%w: streaming transcription with the Azure Speech API", model.ErrUnsupported)

	case TranscriptionAPIMAI:
		if options.Audio == nil {
			return nil, fmt.Errorf("%w: streaming transcription requires audio", model.ErrInvalidArgument)
		}

		warnings := unsupportedOptionWarnings(azureOptions.speechOptionNames(), speechOnlyDetails)

		return m.mai().streamTranscribe(ctx, options, azureOptions, warnings)

	default:
		return m.openai.StreamTranscribe(ctx, options)
	}
}

// mai returns the MAI realtime model of the deployment, which shares the
// provider settings of the Azure Speech model.
func (m *TranscriptionModel) mai() *maiTranscriptionModel {
	return &maiTranscriptionModel{
		modelID:  m.modelID,
		settings: m.speech.settings,
	}
}

// selectedAPI returns the requested API or the model's default.
func (m *TranscriptionModel) selectedAPI(requested TranscriptionAPI) TranscriptionAPI {
	if requested != "" {
		return requested
	}

	if strings.ToLower(m.modelID) == maiStreamingTranscribeModel {
		return TranscriptionAPIMAI
	}

	if _, ok := maiTranscribeModels[strings.ToLower(m.modelID)]; ok {
		return TranscriptionAPISpeech
	}

	return TranscriptionAPIOpenAI
}
