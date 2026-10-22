package azure

import (
	"context"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

var _ model.SpeechModel = (*SpeechModel)(nil)

// maiVoiceModels maps lowercase MAI-Voice model IDs to the suffix of Azure
// Speech voice names.
var maiVoiceModels = map[string]string{
	"mai-voice-2.1-flash": "MAI-Voice-2.1-Flash",
	"mai-voice-2.1":       "MAI-Voice-2.1",
	"mai-voice-2-flash":   "MAI-Voice-2-Flash",
	"mai-voice-2":         "MAI-Voice-2",
}

// SpeechAPI selects the API of a speech call.
type SpeechAPI string

// Speech APIs.
const (
	SpeechAPIOpenAI SpeechAPI = "openai"
	SpeechAPISpeech SpeechAPI = "speech"
)

// SpeechOptions are the Azure options of a speech call, read from
// ProviderOptions["azure"]. OpenAI speech options stay in the "openai"
// namespace.
type SpeechOptions struct {
	// API selects the API. The default is Azure Speech for MAI-Voice models
	// and the OpenAI speech API otherwise.
	API SpeechAPI `json:"api,omitempty"`

	// Style is the speaking style applied with mstts:express-as, such as
	// "excited" or "whispering". Supported styles vary by voice.
	Style string `json:"style,omitempty"`

	// StyleDegree is the intensity of Style, from 0.01 to 2. Azure defaults
	// to 1.
	StyleDegree *float64 `json:"styleDegree,omitempty"`
}

func (o SpeechOptions) validate() error {
	switch o.API {
	case "", SpeechAPIOpenAI, SpeechAPISpeech:
	default:
		return fmt.Errorf("%w: providerOptions.azure.api must be %q or %q", model.ErrInvalidArgument, SpeechAPIOpenAI, SpeechAPISpeech)
	}

	if o.StyleDegree != nil && (*o.StyleDegree < 0.01 || *o.StyleDegree > 2) {
		return fmt.Errorf("%w: providerOptions.azure.styleDegree must be between 0.01 and 2", model.ErrInvalidArgument)
	}

	return nil
}

// speechOptionNames returns the names of the Azure Speech options that are
// set.
func (o SpeechOptions) speechOptionNames() []string {
	var names []string

	if o.Style != "" {
		names = append(names, "style")
	}

	if o.StyleDegree != nil {
		names = append(names, "styleDegree")
	}

	return names
}

// SpeechModel is an Azure speech deployment. It selects Azure Speech or the
// OpenAI speech API for each call.
type SpeechModel struct {
	modelID  string
	provider string
	openai   *openai.SpeechModel
	speech   *speechSynthesisModel
}

// Provider returns "azure.speech".
func (m *SpeechModel) Provider() string {
	return m.provider
}

// ModelID returns the deployment ID.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

// Generate converts text into audio with the selected API. Azure Speech
// options sent to the OpenAI speech API are reported as unsupported.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	azureOptions, _, err := model.ProviderValue[SpeechOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := azureOptions.validate(); err != nil {
		return nil, err
	}

	if m.selectedAPI(azureOptions.API) == SpeechAPISpeech {
		return m.speech.generate(ctx, options, azureOptions)
	}

	result, err := m.openai.Generate(ctx, options)
	if err != nil {
		return nil, err
	}

	result.Warnings = append(result.Warnings, unsupportedOptionWarnings(azureOptions.speechOptionNames(), speechOnlyDetails)...)

	return result, nil
}

// selectedAPI returns the requested API or the model's default.
func (m *SpeechModel) selectedAPI(requested SpeechAPI) SpeechAPI {
	if requested != "" {
		return requested
	}

	if _, ok := maiVoiceModel(m.modelID); ok {
		return SpeechAPISpeech
	}

	return SpeechAPIOpenAI
}

// maiVoiceModel returns the voice name suffix of a MAI-Voice model.
func maiVoiceModel(modelID string) (string, bool) {
	suffix, ok := maiVoiceModels[strings.ToLower(modelID)]

	return suffix, ok
}
