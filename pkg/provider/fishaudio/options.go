package fishaudio

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// Latency is the latency and quality tradeoff of speech generation.
type Latency string

// Latencies.
const (
	// LatencyNormal gives the best quality.
	LatencyNormal Latency = "normal"

	// LatencyBalanced reduces latency.
	LatencyBalanced Latency = "balanced"

	// LatencyLow is the fastest.
	LatencyLow Latency = "low"
)

var (
	// mp3Bitrates are the accepted mp3 bitrates, in kbps.
	mp3Bitrates = []int{
		64,
		128,
		192,
	}

	// opusBitrates are the accepted opus bitrates, in bps. -1000 selects
	// automatic.
	opusBitrates = []int{
		-1000,
		24000,
		32000,
		48000,
		64000,
	}
)

// SpeechOptions are the provider options of speech models.
type SpeechOptions struct {
	// ReferenceID selects the voice, or several voices for multi-speaker
	// dialogue. It takes precedence over the call's voice.
	ReferenceID *ReferenceID `json:"referenceId,omitempty"`

	// SampleRate is the output sample rate in Hz.
	SampleRate *int `json:"sampleRate,omitempty"`

	// MP3Bitrate is 64, 128 or 192 kbps and only applies to mp3 output.
	MP3Bitrate *int `json:"mp3Bitrate,omitempty"`

	// OpusBitrate is -1000 (automatic), 24000, 32000, 48000 or 64000 bps and
	// only applies to opus output.
	OpusBitrate *int `json:"opusBitrate,omitempty"`

	// Latency is the latency and quality tradeoff.
	Latency Latency `json:"latency,omitempty"`

	// Volume is a volume offset in dB.
	Volume *float64 `json:"volume,omitempty"`

	// NormalizeLoudness enables loudness normalization on the S2 family.
	NormalizeLoudness *bool `json:"normalizeLoudness,omitempty"`

	// Temperature governs expressiveness, from 0 to 1.
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP is the nucleus sampling threshold, from 0 to 1.
	TopP *float64 `json:"topP,omitempty"`

	// ChunkLength is the text segment size, from 100 to 300.
	ChunkLength *int `json:"chunkLength,omitempty"`

	// MinChunkLength is the minimum characters before a new chunk, from 0 to
	// 100.
	MinChunkLength *int `json:"minChunkLength,omitempty"`

	// Normalize normalizes English and Chinese text.
	Normalize *bool `json:"normalize,omitempty"`

	// MaxNewTokens is the maximum audio tokens per text chunk.
	MaxNewTokens *int `json:"maxNewTokens,omitempty"`

	// RepetitionPenalty above 1.0 discourages repeated audio patterns.
	RepetitionPenalty *float64 `json:"repetitionPenalty,omitempty"`

	// ConditionOnPreviousChunks reuses prior audio for voice consistency.
	ConditionOnPreviousChunks *bool `json:"conditionOnPreviousChunks,omitempty"`

	// EarlyStopThreshold is the batch early-stop threshold, from 0 to 1.
	EarlyStopThreshold *float64 `json:"earlyStopThreshold,omitempty"`

	// Features are request-scoped backend flags, such as "quality-guard".
	Features []string `json:"features,omitempty"`
}

// ReferenceID is one voice model ID, or a list of IDs for multi-speaker
// dialogue whose speaker tokens index the list. Set either ID or IDs.
type ReferenceID struct {
	ID  string
	IDs []string
}

// UnmarshalJSON accepts a JSON string or an array of strings.
func (r *ReferenceID) UnmarshalJSON(data []byte) error {
	var id string
	if err := json.Unmarshal(data, &id); err == nil {
		*r = ReferenceID{
			ID: id,
		}

		return nil
	}

	var ids []string
	if err := json.Unmarshal(data, &ids); err != nil {
		return fmt.Errorf("reference ID must be a string or an array of strings: %w", err)
	}

	*r = ReferenceID{
		IDs: ids,
	}

	return nil
}

// MarshalJSON encodes a list as an array and a single ID as a string.
func (r ReferenceID) MarshalJSON() ([]byte, error) {
	if r.IDs != nil {
		return json.Marshal(r.IDs)
	}

	return json.Marshal(r.ID)
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
	if o.SampleRate != nil && *o.SampleRate <= 0 {
		return invalidOption("sampleRate must be positive")
	}

	if o.MP3Bitrate != nil && !slices.Contains(mp3Bitrates, *o.MP3Bitrate) {
		return invalidOption("mp3Bitrate must be 64, 128 or 192")
	}

	if o.OpusBitrate != nil && !slices.Contains(opusBitrates, *o.OpusBitrate) {
		return invalidOption("opusBitrate must be -1000, 24000, 32000, 48000 or 64000")
	}

	switch o.Latency {
	case "", LatencyNormal, LatencyBalanced, LatencyLow:
	default:
		return invalidOption(fmt.Sprintf("latency %q", o.Latency))
	}

	if !floatWithin(o.Temperature, 0, 1) || !floatWithin(o.TopP, 0, 1) || !floatWithin(o.EarlyStopThreshold, 0, 1) {
		return invalidOption("temperature, topP and earlyStopThreshold must be between 0 and 1")
	}

	if !intWithin(o.ChunkLength, 100, 300) {
		return invalidOption("chunkLength must be between 100 and 300")
	}

	if !intWithin(o.MinChunkLength, 0, 100) {
		return invalidOption("minChunkLength must be between 0 and 100")
	}

	if o.MaxNewTokens != nil && *o.MaxNewTokens <= 0 {
		return invalidOption("maxNewTokens must be positive")
	}

	return nil
}

// TranscriptionOptions are the provider options of the transcription model.
type TranscriptionOptions struct {
	// Language is a hint only: detection is authoritative.
	Language *string `json:"language,omitempty"`

	// IgnoreTimestamps skips precise timestamps. It defaults to false here,
	// unlike the API, so that segments are populated.
	IgnoreTimestamps *bool `json:"ignoreTimestamps,omitempty"`
}

func invalidOption(details string) error {
	return fmt.Errorf("%w: fish audio %s", model.ErrInvalidArgument, details)
}

// intWithin reports whether an optional value is unset or within the
// inclusive bounds.
func intWithin(value *int, minimum int, maximum int) bool {
	return value == nil || (*value >= minimum && *value <= maximum)
}

// floatWithin reports whether an optional value is unset or within the
// inclusive bounds.
func floatWithin(value *float64, minimum float64, maximum float64) bool {
	return value == nil || (*value >= minimum && *value <= maximum)
}
