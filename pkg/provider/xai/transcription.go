package xai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// transcriptionPath is the speech-to-text endpoint.
const transcriptionPath = "/stt"

// Channel limits of multichannel transcription.
const (
	minChannels = 2
	maxChannels = 8
)

// audioFormats are the raw audio encodings xAI accepts.
var audioFormats = []string{
	"pcm",
	"mulaw",
	"alaw",
	"opus",
}

// TranscriptionOptions are the provider options of transcription models,
// read from the "xai" namespace.
type TranscriptionOptions struct {
	// AudioFormat is the encoding of raw audio: "pcm", "mulaw", "alaw" or
	// "opus".
	AudioFormat *string `json:"audioFormat,omitempty"`

	// SampleRate is the sample rate of raw audio in Hz: 8000, 16000, 22050,
	// 24000, 44100 or 48000.
	SampleRate *int `json:"sampleRate,omitempty"`

	// Language is the language of the audio.
	Language *string `json:"language,omitempty"`

	// Format applies text formatting to the transcript.
	Format *bool `json:"format,omitempty"`

	// Multichannel transcribes each channel separately.
	Multichannel *bool `json:"multichannel,omitempty"`

	// Channels is the number of channels, from 2 to 8.
	Channels *int `json:"channels,omitempty"`

	// Diarize labels words with speakers, reported in
	// TranscriptionMetadata.
	Diarize *bool `json:"diarize,omitempty"`

	// Keyterm boosts the recognition of terms.
	Keyterm Keyterms `json:"keyterm,omitempty"`

	// FillerWords keeps filler words in the transcript.
	FillerWords *bool `json:"fillerWords,omitempty"`

	// VadThreshold is the voice activity detection threshold between 0 and
	// 1. It defaults to 0.5 for Transcribe and 0.08 for StreamTranscribe; 0
	// disables the filter.
	VadThreshold *float64 `json:"vadThreshold,omitempty"`

	// Streaming configures StreamTranscribe.
	Streaming *TranscriptionStreamingOptions `json:"streaming,omitempty"`
}

// Keyterms are one or more terms. A single term may be given as a JSON
// string.
type Keyterms []string

// UnmarshalJSON accepts a string or an array of strings.
func (k *Keyterms) UnmarshalJSON(data []byte) error {
	if bytes.HasPrefix(bytes.TrimSpace(data), []byte(`"`)) {
		var term string
		if err := json.Unmarshal(data, &term); err != nil {
			return err
		}

		*k = Keyterms{
			term,
		}

		return nil
	}

	var terms []string
	if err := json.Unmarshal(data, &terms); err != nil {
		return err
	}

	*k = terms

	return nil
}

// TranscriptionMetadata is the result provider metadata of transcription
// models, stored under "xai" when words carry speakers.
type TranscriptionMetadata struct {
	Words []TranscribedWord `json:"words"`
}

// TranscribedWord is one transcribed word.
type TranscribedWord struct {
	Text    string   `json:"text"`
	Start   float64  `json:"start"`
	End     float64  `json:"end"`
	Speaker *float64 `json:"speaker,omitempty"`
}

// TranscriptionModel is an xAI transcription model. It transcribes files
// with Transcribe and live audio over a WebSocket with StreamTranscribe.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "xai.transcription" by default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns the model identifier, which is empty for xAI's default
// model.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

type transcriptionResponse struct {
	Text     string            `json:"text"`
	Language *string           `json:"language"`
	Duration *float64          `json:"duration"`
	Words    []TranscribedWord `json:"words"`
}

// Transcribe converts the audio into text.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	transcriptionOptions, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(transcriptionPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostForm(ctx, request, m.formFields(options, transcriptionOptions))
	if err != nil {
		return nil, err
	}

	var decoded transcriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	result := &model.TranscriptionResult{
		Text:     decoded.Text,
		Segments: transcriptSegments(decoded.Words),
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.Language != nil {
		result.Language = *decoded.Language
	}

	if decoded.Duration != nil {
		result.Duration = seconds(*decoded.Duration)
	}

	if hasSpeakers(decoded.Words) {
		result.ProviderMetadata = model.ProviderMetadata{
			Namespace: TranscriptionMetadata{
				Words: decoded.Words,
			},
		}
	}

	return result, nil
}

// formFields lists the parameters in order, with the file last as xAI
// requires.
func (m *TranscriptionModel) formFields(options model.TranscriptionOptions, transcriptionOptions TranscriptionOptions) []apicall.FormField {
	var fields []apicall.FormField

	add := func(name string, value string) {
		fields = append(fields, apicall.FormField{
			Name:  name,
			Value: value,
		})
	}

	if m.modelID != "" {
		add("model", m.modelID)
	}

	if transcriptionOptions.SampleRate != nil {
		add("sample_rate", strconv.Itoa(*transcriptionOptions.SampleRate))
	}

	if transcriptionOptions.Language != nil {
		add("language", *transcriptionOptions.Language)
	}

	addBool(add, "multichannel", transcriptionOptions.Multichannel)

	if transcriptionOptions.Channels != nil {
		add("channels", strconv.Itoa(*transcriptionOptions.Channels))
	}

	addBool(add, "diarize", transcriptionOptions.Diarize)
	addBool(add, "filler_words", transcriptionOptions.FillerWords)

	if transcriptionOptions.VadThreshold != nil {
		add("vad_threshold", strconv.FormatFloat(*transcriptionOptions.VadThreshold, 'f', -1, 64))
	}

	for _, term := range transcriptionOptions.Keyterm {
		add("keyterm", term)
	}

	if transcriptionOptions.AudioFormat != nil {
		add("audio_format", *transcriptionOptions.AudioFormat)
	}

	addBool(add, "format", transcriptionOptions.Format)

	return append(fields, apicall.FormField{
		Name:      "file",
		Filename:  "audio." + audioExtension(options.MediaType),
		MediaType: options.MediaType,
		Data:      options.Audio,
	})
}

func addBool(add func(name string, value string), name string, value *bool) {
	if value != nil {
		add(name, strconv.FormatBool(*value))
	}
}

// audioExtension returns the file extension of an audio media type.
func audioExtension(mediaType string) string {
	_, subtype, _ := strings.Cut(strings.ToLower(mediaType), "/")

	switch subtype {
	case "mpeg":
		return "mp3"
	case "x-wav":
		return "wav"
	case "opus":
		return "ogg"
	case "mp4", "x-m4a":
		return "m4a"
	default:
		return subtype
	}
}

func transcriptSegments(words []TranscribedWord) []model.TranscriptionSegment {
	segments := make([]model.TranscriptionSegment, 0, len(words))
	for _, word := range words {
		segments = append(segments, model.TranscriptionSegment{
			Text:  word.Text,
			Start: seconds(word.Start),
			End:   seconds(word.End),
		})
	}

	return segments
}

func hasSpeakers(words []TranscribedWord) bool {
	return slices.ContainsFunc(words, func(word TranscribedWord) bool {
		return word.Speaker != nil
	})
}

func seconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Second))
}

// readTranscriptionOptions decodes and validates the transcription options.
func readTranscriptionOptions(options model.ProviderOptions) (TranscriptionOptions, error) {
	transcriptionOptions, _, err := model.ProviderValue[TranscriptionOptions](options, Namespace)
	if err != nil {
		return TranscriptionOptions{}, err
	}

	if err := transcriptionOptions.validate(); err != nil {
		return TranscriptionOptions{}, fmt.Errorf("%w: xai provider options: %w", model.ErrInvalidArgument, err)
	}

	return transcriptionOptions, nil
}

func (o TranscriptionOptions) validate() error {
	if o.AudioFormat != nil && !slices.Contains(audioFormats, *o.AudioFormat) {
		return fmt.Errorf("unsupported audioFormat %q", *o.AudioFormat)
	}

	if o.SampleRate != nil && !slices.Contains(speechSampleRates, *o.SampleRate) {
		return fmt.Errorf("unsupported sampleRate %d", *o.SampleRate)
	}

	if o.Channels != nil && (*o.Channels < minChannels || *o.Channels > maxChannels) {
		return fmt.Errorf("channels must be between %d and %d", minChannels, maxChannels)
	}

	if o.VadThreshold != nil && (*o.VadThreshold < 0 || *o.VadThreshold > 1) {
		return errors.New("vadThreshold must be between 0 and 1")
	}

	if o.Streaming != nil {
		return o.Streaming.validate()
	}

	return nil
}

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)
