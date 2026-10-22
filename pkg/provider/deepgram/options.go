package deepgram

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// CallbackMethod is the HTTP method of a speech callback.
type CallbackMethod string

// Callback methods.
const (
	CallbackPost CallbackMethod = "POST"
	CallbackPut  CallbackMethod = "PUT"
)

// SpeechOptions are the provider options of speech models. They refine the
// encoding, container, sample rate and bit rate derived from the output
// format; incompatible combinations are ignored with a warning.
type SpeechOptions struct {
	// BitRate is the bit rate in bits per second, as a number or a
	// predefined value.
	BitRate *BitRate `json:"bitRate,omitempty"`

	// Container is the output container, such as "wav", "ogg" or "none".
	Container *string `json:"container,omitempty"`

	// Encoding is the output encoding, such as "linear16", "mulaw" or "mp3".
	Encoding *string `json:"encoding,omitempty"`

	// SampleRate is the output sample rate in Hz.
	SampleRate *float64 `json:"sampleRate,omitempty"`

	// Callback is the URL that receives the audio asynchronously.
	Callback *string `json:"callback,omitempty"`

	// CallbackMethod is the method of the callback request.
	CallbackMethod CallbackMethod `json:"callbackMethod,omitempty"`

	// MIPOptOut opts out of the Deepgram Model Improvement Program.
	MIPOptOut *bool `json:"mipOptOut,omitempty"`

	// Tag labels the request for usage reporting.
	Tag StringList `json:"tag,omitempty"`
}

// BitRate is a bit rate given as a number or a string. It holds the text
// that is sent to Deepgram.
type BitRate string

// UnmarshalJSON accepts a JSON number or string.
func (b *BitRate) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*b = BitRate(text)

		return nil
	}

	var number json.Number
	if err := json.Unmarshal(data, &number); err != nil {
		return fmt.Errorf("bit rate must be a number or a string: %w", err)
	}

	*b = BitRate(number.String())

	return nil
}

// StringList is one string or a list of strings. A single JSON string
// decodes as a one-item list.
type StringList []string

// UnmarshalJSON accepts a JSON string or an array of strings.
func (l *StringList) UnmarshalJSON(data []byte) error {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		*l = nil

		return nil
	}

	var single string
	if err := json.Unmarshal(data, &single); err == nil {
		*l = StringList{
			single,
		}

		return nil
	}

	var list []string
	if err := json.Unmarshal(data, &list); err != nil {
		return fmt.Errorf("value must be a string or an array of strings: %w", err)
	}

	*l = list

	return nil
}

// joined returns the items separated by commas, as JavaScript formats an
// array.
func (l StringList) joined() string {
	return strings.Join(l, ",")
}

func readSpeechOptions(options model.ProviderOptions) (SpeechOptions, error) {
	speechOptions, _, err := model.ProviderValue[SpeechOptions](options, Namespace)
	if err != nil {
		return SpeechOptions{}, err
	}

	if speechOptions.Callback != nil && !isAbsoluteURL(*speechOptions.Callback) {
		return SpeechOptions{}, fmt.Errorf("%w: deepgram callback must be an absolute URL", model.ErrInvalidArgument)
	}

	switch speechOptions.CallbackMethod {
	case "", CallbackPost, CallbackPut:
		return speechOptions, nil
	default:
		return SpeechOptions{}, fmt.Errorf("%w: deepgram callbackMethod %q", model.ErrInvalidArgument, speechOptions.CallbackMethod)
	}
}

func isAbsoluteURL(value string) bool {
	parsed, err := url.Parse(value)

	return err == nil && parsed.Scheme != ""
}

// Summarize selects transcript summarization.
type Summarize string

// Summarize values.
const (
	// SummarizeV2 uses the latest summarization.
	SummarizeV2 Summarize = "v2"

	// SummarizeDisabled disables summarization. It encodes as JSON false.
	SummarizeDisabled Summarize = "false"
)

// UnmarshalJSON accepts "v2" or false.
func (s *Summarize) UnmarshalJSON(data []byte) error {
	var disabled bool
	if err := json.Unmarshal(data, &disabled); err == nil {
		if disabled {
			return fmt.Errorf("summarize must be %q or false", SummarizeV2)
		}

		*s = SummarizeDisabled

		return nil
	}

	var version string
	if err := json.Unmarshal(data, &version); err != nil || Summarize(version) != SummarizeV2 {
		return fmt.Errorf("summarize must be %q or false", SummarizeV2)
	}

	*s = SummarizeV2

	return nil
}

// MarshalJSON encodes SummarizeDisabled as false.
func (s Summarize) MarshalJSON() ([]byte, error) {
	if s == SummarizeDisabled {
		return []byte("false"), nil
	}

	return json.Marshal(string(s))
}

// TranscriptionOptions are the provider options of transcription models.
type TranscriptionOptions struct {
	// Language of the audio. Deepgram defaults to English.
	Language *string `json:"language,omitempty"`

	// DetectLanguage detects the language of the audio.
	DetectLanguage *bool `json:"detectLanguage,omitempty"`

	// SmartFormat formats numbers, dates and times.
	SmartFormat *bool `json:"smartFormat,omitempty"`

	// Punctuate adds punctuation.
	Punctuate *bool `json:"punctuate,omitempty"`

	// Paragraphs splits the transcript into paragraphs.
	Paragraphs *bool `json:"paragraphs,omitempty"`

	// Summarize generates a summary.
	Summarize Summarize `json:"summarize,omitempty"`

	// Topics identifies topics.
	Topics *bool `json:"topics,omitempty"`

	// Intents identifies intents.
	Intents *bool `json:"intents,omitempty"`

	// Sentiment analyzes sentiment.
	Sentiment *bool `json:"sentiment,omitempty"`

	// DetectEntities tags named entities.
	DetectEntities *bool `json:"detectEntities,omitempty"`

	// Redact lists terms or patterns to redact.
	Redact StringList `json:"redact,omitempty"`

	// Replace replaces redacted content.
	Replace *string `json:"replace,omitempty"`

	// Search is a term to search for.
	Search *string `json:"search,omitempty"`

	// Keyterm is a key term to identify.
	Keyterm *string `json:"keyterm,omitempty"`

	// Diarize identifies speakers.
	Diarize *bool `json:"diarize,omitempty"`

	// Utterances segments the transcript into utterances.
	Utterances *bool `json:"utterances,omitempty"`

	// UttSplit is the silence, in seconds, that starts a new utterance.
	UttSplit *float64 `json:"uttSplit,omitempty"`

	// FillerWords keeps filler words such as "um".
	FillerWords *bool `json:"fillerWords,omitempty"`
}

func readTranscriptionOptions(options model.ProviderOptions) (TranscriptionOptions, error) {
	transcriptionOptions, _, err := model.ProviderValue[TranscriptionOptions](options, Namespace)
	if err != nil {
		return TranscriptionOptions{}, err
	}

	switch transcriptionOptions.Summarize {
	case "", SummarizeV2, SummarizeDisabled:
		return transcriptionOptions, nil
	default:
		return TranscriptionOptions{}, fmt.Errorf("%w: deepgram summarize %q", model.ErrInvalidArgument, transcriptionOptions.Summarize)
	}
}
