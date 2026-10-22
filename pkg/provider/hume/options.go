package hume

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// VoiceProvider selects the library a voice comes from.
type VoiceProvider string

// Voice providers.
const (
	VoiceHumeAI      VoiceProvider = "HUME_AI"
	VoiceCustomVoice VoiceProvider = "CUSTOM_VOICE"
)

// SpeechOptions are the provider options of the speech model.
type SpeechOptions struct {
	// Context continues a previous generation or supplies earlier utterances
	// so that the new speech sounds consistent with them.
	Context *SpeechContext `json:"context,omitempty"`
}

// SpeechContext is either the ID of a previous generation or a list of
// utterances. GenerationID takes precedence when both are set.
type SpeechContext struct {
	// GenerationID is the ID of a previous generation.
	GenerationID *string `json:"generationId,omitempty"`

	// Utterances are earlier utterances that provide context.
	Utterances []Utterance `json:"utterances,omitempty"`
}

// Utterance is one piece of text to speak.
type Utterance struct {
	// Text is the text to speak.
	Text string `json:"text"`

	// Description guides how the text is spoken.
	Description *string `json:"description,omitempty"`

	// Speed is the speech rate multiplier.
	Speed *float64 `json:"speed,omitempty"`

	// TrailingSilence is the silence after the utterance, in seconds.
	TrailingSilence *float64 `json:"trailingSilence,omitempty"`

	// Voice selects the voice by ID or by name.
	Voice *Voice `json:"voice,omitempty"`
}

// Voice selects a voice. Exactly one of ID and Name is set.
type Voice struct {
	ID       string        `json:"id,omitempty"`
	Name     string        `json:"name,omitempty"`
	Provider VoiceProvider `json:"provider,omitempty"`
}

func readSpeechOptions(options model.ProviderOptions) (SpeechOptions, error) {
	speechOptions, _, err := model.ProviderValue[SpeechOptions](options, Namespace)
	if err != nil {
		return SpeechOptions{}, err
	}

	if speechOptions.Context == nil {
		return speechOptions, nil
	}

	if err := speechOptions.Context.validate(); err != nil {
		return SpeechOptions{}, err
	}

	return speechOptions, nil
}

func (c SpeechContext) validate() error {
	if c.GenerationID != nil {
		return nil
	}

	if c.Utterances == nil {
		return fmt.Errorf("%w: hume context requires generationId or utterances", model.ErrInvalidArgument)
	}

	for _, utterance := range c.Utterances {
		if utterance.Voice == nil {
			continue
		}

		if err := utterance.Voice.validate(); err != nil {
			return err
		}
	}

	return nil
}

func (v Voice) validate() error {
	if (v.ID == "") == (v.Name == "") {
		return fmt.Errorf("%w: hume voice requires exactly one of id and name", model.ErrInvalidArgument)
	}

	switch v.Provider {
	case "", VoiceHumeAI, VoiceCustomVoice:
		return nil
	default:
		return fmt.Errorf("%w: hume voice provider %q", model.ErrInvalidArgument, v.Provider)
	}
}
