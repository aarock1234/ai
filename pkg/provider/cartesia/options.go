package cartesia

import (
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// Container is the container of generated audio.
type Container string

// Containers.
const (
	ContainerRaw Container = "raw"
	ContainerWAV Container = "wav"
	ContainerMP3 Container = "mp3"
)

// Encoding is the sample encoding of raw and WAV audio.
type Encoding string

// Encodings.
const (
	EncodingPCMF32LE Encoding = "pcm_f32le"
	EncodingPCMS16LE Encoding = "pcm_s16le"
	EncodingPCMMulaw Encoding = "pcm_mulaw"
	EncodingPCMAlaw  Encoding = "pcm_alaw"
)

const (
	// minSpeed and maxSpeed bound the generation speed.
	minSpeed = 0.6
	maxSpeed = 1.5
)

var (
	// sampleRates are the sample rates Cartesia accepts, in Hz.
	sampleRates = []int{
		8000,
		16000,
		22050,
		24000,
		44100,
		48000,
	}

	// bitRates are the mp3 bit rates Cartesia accepts, in bits per second.
	bitRates = []int{
		32000,
		64000,
		96000,
		128000,
		192000,
	}
)

// SpeechOptions are the provider options of speech models. They override the
// container, encoding, sample rate and bit rate derived from the output
// format, and the call's language and speed.
type SpeechOptions struct {
	Container  Container `json:"container,omitempty"`
	Encoding   Encoding  `json:"encoding,omitempty"`
	SampleRate *int      `json:"sampleRate,omitempty"`

	// BitRate applies to mp3 output only.
	BitRate *int `json:"bitRate,omitempty"`

	// Speed is between 0.6 and 1.5.
	Speed *float64 `json:"speed,omitempty"`

	// Language is the ISO 639-1 language of the speech.
	Language *string `json:"language,omitempty"`
}

func readSpeechOptions(options model.ProviderOptions) (SpeechOptions, error) {
	speechOptions, _, err := model.ProviderValue[SpeechOptions](options, Namespace)
	if err != nil {
		return SpeechOptions{}, err
	}

	if err := speechOptions.validate(); err != nil {
		return SpeechOptions{}, err
	}

	return speechOptions, nil
}

func (o SpeechOptions) validate() error {
	switch o.Container {
	case "", ContainerRaw, ContainerWAV, ContainerMP3:
	default:
		return fmt.Errorf("%w: cartesia container %q", model.ErrInvalidArgument, o.Container)
	}

	switch o.Encoding {
	case "", EncodingPCMF32LE, EncodingPCMS16LE, EncodingPCMMulaw, EncodingPCMAlaw:
	default:
		return fmt.Errorf("%w: cartesia encoding %q", model.ErrInvalidArgument, o.Encoding)
	}

	if o.SampleRate != nil && !slices.Contains(sampleRates, *o.SampleRate) {
		return fmt.Errorf("%w: cartesia sampleRate %d", model.ErrInvalidArgument, *o.SampleRate)
	}

	if o.BitRate != nil && !slices.Contains(bitRates, *o.BitRate) {
		return fmt.Errorf("%w: cartesia bitRate %d", model.ErrInvalidArgument, *o.BitRate)
	}

	if o.Speed != nil && (*o.Speed < minSpeed || *o.Speed > maxSpeed) {
		return fmt.Errorf("%w: cartesia speed must be between %g and %g", model.ErrInvalidArgument, minSpeed, maxSpeed)
	}

	return nil
}

// TimestampGranularity selects the timestamps of a transcription.
type TimestampGranularity string

// TimestampWord requests word timestamps, the only supported granularity.
const TimestampWord TimestampGranularity = "word"

// StreamingEncoding is a raw audio encoding of streaming transcription.
type StreamingEncoding string

// Streaming encodings.
const (
	StreamingPCMAlaw  StreamingEncoding = "pcm_alaw"
	StreamingPCMF16LE StreamingEncoding = "pcm_f16le"
	StreamingPCMF32LE StreamingEncoding = "pcm_f32le"
	StreamingPCMMulaw StreamingEncoding = "pcm_mulaw"
	StreamingPCMS16LE StreamingEncoding = "pcm_s16le"
	StreamingPCMS32LE StreamingEncoding = "pcm_s32le"
)

// TranscriptionOptions are the provider options of transcription models.
type TranscriptionOptions struct {
	// Language is the ISO 639-1 language of the audio. Cartesia defaults to
	// English.
	Language *string `json:"language,omitempty"`

	// TimestampGranularities selects the timestamps to populate.
	TimestampGranularities []TimestampGranularity `json:"timestampGranularities,omitempty"`

	// Streaming configures realtime Ink 2 transcription. Batch transcription
	// warns that it is unsupported.
	Streaming *TranscriptionStreamingOptions `json:"streaming,omitempty"`
}

// TranscriptionStreamingOptions configure realtime Ink 2 transcription.
type TranscriptionStreamingOptions struct {
	// Encoding defaults to the encoding of the input audio format.
	Encoding StreamingEncoding `json:"encoding,omitempty"`

	// TurnDetection uses Cartesia's turn detection. It defaults to true.
	TurnDetection *bool `json:"turnDetection,omitempty"`
}

func readTranscriptionOptions(options model.ProviderOptions) (TranscriptionOptions, error) {
	transcriptionOptions, _, err := model.ProviderValue[TranscriptionOptions](options, Namespace)
	if err != nil {
		return TranscriptionOptions{}, err
	}

	for _, granularity := range transcriptionOptions.TimestampGranularities {
		if granularity != TimestampWord {
			return TranscriptionOptions{}, fmt.Errorf("%w: cartesia timestamp granularity %q", model.ErrInvalidArgument, granularity)
		}
	}

	if transcriptionOptions.Streaming == nil {
		return transcriptionOptions, nil
	}

	switch transcriptionOptions.Streaming.Encoding {
	case "", StreamingPCMAlaw, StreamingPCMF16LE, StreamingPCMF32LE, StreamingPCMMulaw, StreamingPCMS16LE, StreamingPCMS32LE:
		return transcriptionOptions, nil
	default:
		return TranscriptionOptions{}, fmt.Errorf("%w: cartesia streaming encoding %q", model.ErrInvalidArgument, transcriptionOptions.Streaming.Encoding)
	}
}
