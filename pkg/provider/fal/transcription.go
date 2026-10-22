package fal

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// transcriptionTimeout bounds the wait for a queued transcription.
	transcriptionTimeout = 60 * time.Second

	// transcriptionPollInterval is the delay between status checks.
	transcriptionPollInterval = time.Second

	// inProgressDetail is the detail of a failed status response while the
	// request is still running.
	inProgressDetail = "Request is still in progress"
)

// ErrTranscriptionTimeout reports a transcription that did not finish within
// 60 seconds.
var ErrTranscriptionTimeout = errors.New("fal transcription timed out after 60 seconds")

// ChunkLevel is the granularity of transcript chunks.
type ChunkLevel string

// Chunk levels.
const (
	ChunkSegment ChunkLevel = "segment"
	ChunkWord    ChunkLevel = "word"
)

// TranscriptionOptions are the provider options of transcription models,
// read from the "fal" namespace. When the namespace is present, unset
// options take the documented defaults; without it, the request uses
// word-level chunks and fal.ai's language detection.
type TranscriptionOptions struct {
	// Language is the audio language. The default is "en"; an explicit JSON
	// null detects the language.
	Language *string `json:"language,omitempty"`

	// Diarize labels speakers. The default is true.
	Diarize *bool `json:"diarize,omitempty"`

	// ChunkLevel is the chunk granularity. The default is "segment".
	ChunkLevel *ChunkLevel `json:"chunkLevel,omitempty"`

	// Version is the Whisper version. The default is "3".
	Version *string `json:"version,omitempty"`

	// BatchSize is the inference batch size. The default is 64.
	BatchSize *float64 `json:"batchSize,omitempty"`

	// NumSpeakers is the number of speakers. Nil detects it.
	NumSpeakers *float64 `json:"numSpeakers,omitempty"`
}

// TranscriptionModel is a fal.ai Whisper transcription model. Requests are
// queued and polled for up to 60 seconds.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "fal.transcription" by default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

type transcriptionRequest struct {
	Task        string          `json:"task"`
	Diarize     bool            `json:"diarize"`
	ChunkLevel  ChunkLevel      `json:"chunk_level"`
	Language    json.RawMessage `json:"language,omitempty"`
	Version     *string         `json:"version,omitempty"`
	BatchSize   *float64        `json:"batch_size,omitempty"`
	NumSpeakers *float64        `json:"num_speakers,omitempty"`
	AudioURL    string          `json:"audio_url"`
}

type queueResponse struct {
	RequestID *string `json:"request_id"`
}

type transcriptionResponse struct {
	Text              string               `json:"text"`
	Chunks            []transcriptionChunk `json:"chunks"`
	InferredLanguages []string             `json:"inferred_languages"`
}

type transcriptionChunk struct {
	Text      string    `json:"text"`
	Timestamp []float64 `json:"timestamp"`
}

// statusError is the body of a failed status check.
type statusError struct {
	Detail string `json:"detail"`
}

// Transcribe queues the audio and polls until the transcript is ready.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	body, err := transcriptionBody(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(queueBaseURL+"/"+m.modelID, options.Headers, parseError)
	if err != nil {
		return nil, err
	}

	queued, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var queue queueResponse
	if err := apicall.DecodeJSON(queued.Body, &queue); err != nil {
		return nil, err
	}

	if queue.RequestID == nil {
		return nil, fmt.Errorf("%w: fal queue response has no request_id", model.ErrInvalidResponse)
	}

	statusRequest, err := m.provider.apiRequest(
		queueBaseURL+"/"+m.modelID+"/requests/"+url.PathEscape(*queue.RequestID),
		options.Headers,
		parseError,
	)
	if err != nil {
		return nil, err
	}

	response, err := pollTranscription(ctx, statusRequest)
	if err != nil {
		return nil, err
	}

	var decoded transcriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	return transcriptionResult(decoded, response, timestamp, m.modelID), nil
}

// transcriptionBody builds the queued request. Provider options, when
// present, apply their defaults.
func transcriptionBody(options model.TranscriptionOptions) (transcriptionRequest, error) {
	body := transcriptionRequest{
		Task:       "transcribe",
		Diarize:    true,
		ChunkLevel: ChunkWord,
		AudioURL:   dataurl.Encode(options.MediaType, options.Audio),
	}

	transcriptionOptions, present, err := model.ProviderValue[TranscriptionOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return transcriptionRequest{}, err
	}

	if !present {
		return body, nil
	}

	if transcriptionOptions.ChunkLevel != nil &&
		*transcriptionOptions.ChunkLevel != ChunkSegment &&
		*transcriptionOptions.ChunkLevel != ChunkWord {
		return transcriptionRequest{}, fmt.Errorf("%w: fal chunkLevel %q", model.ErrInvalidArgument, *transcriptionOptions.ChunkLevel)
	}

	if transcriptionOptions.Version != nil && *transcriptionOptions.Version != "3" {
		return transcriptionRequest{}, fmt.Errorf("%w: fal version must be \"3\"", model.ErrInvalidArgument)
	}

	language, err := transcriptionLanguage(options.ProviderOptions, transcriptionOptions.Language)
	if err != nil {
		return transcriptionRequest{}, err
	}

	body.Language = language
	body.ChunkLevel = ChunkSegment
	body.Version = new("3")
	body.BatchSize = new(64.0)
	body.NumSpeakers = transcriptionOptions.NumSpeakers

	if transcriptionOptions.ChunkLevel != nil {
		body.ChunkLevel = *transcriptionOptions.ChunkLevel
	}

	if transcriptionOptions.Version != nil {
		body.Version = transcriptionOptions.Version
	}

	if transcriptionOptions.BatchSize != nil {
		body.BatchSize = transcriptionOptions.BatchSize
	}

	if transcriptionOptions.Diarize != nil {
		body.Diarize = *transcriptionOptions.Diarize
	}

	return body, nil
}

// transcriptionLanguage returns the language member: the option, "en" when
// it is absent, or null when it is explicitly null.
func transcriptionLanguage(options model.ProviderOptions, language *string) (json.RawMessage, error) {
	if language != nil {
		return json.Marshal(*language)
	}

	raw, _, err := model.ProviderValue[map[string]json.RawMessage](options, Namespace)
	if err != nil {
		return nil, err
	}

	if value, ok := raw["language"]; ok && bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
		return json.RawMessage("null"), nil
	}

	return json.RawMessage(`"en"`), nil
}

// pollTranscription checks the queued request until it succeeds. A failed
// check whose detail reports the request as in progress is retried until
// the timeout.
func pollTranscription(ctx context.Context, request apicall.Request) (*apicall.Response, error) {
	start := time.Now()

	for {
		response, err := apicall.Get(ctx, request)
		if err == nil {
			return response, nil
		}

		if !isInProgress(err) {
			return nil, err
		}

		if time.Since(start) > transcriptionTimeout {
			return nil, ErrTranscriptionTimeout
		}

		if err := poll.Wait(ctx, transcriptionPollInterval); err != nil {
			return nil, err
		}
	}
}

// isInProgress reports whether a failed status check means the request is
// still running.
func isInProgress(err error) bool {
	var apiErr *model.APICallError
	if !errors.As(err, &apiErr) {
		return false
	}

	var body statusError
	if json.Unmarshal([]byte(apiErr.ResponseBody), &body) != nil {
		return false
	}

	return body.Detail == inProgressDetail
}

func transcriptionResult(
	decoded transcriptionResponse,
	response *apicall.Response,
	timestamp time.Time,
	modelID string,
) *model.TranscriptionResult {
	segments := make([]model.TranscriptionSegment, 0, len(decoded.Chunks))
	for _, chunk := range decoded.Chunks {
		segments = append(segments, model.TranscriptionSegment{
			Text:  chunk.Text,
			Start: chunkTime(chunk, 0),
			End:   chunkTime(chunk, 1),
		})
	}

	var language string
	if len(decoded.InferredLanguages) > 0 {
		language = decoded.InferredLanguages[0]
	}

	var duration time.Duration
	if len(decoded.Chunks) > 0 {
		duration = chunkTime(decoded.Chunks[len(decoded.Chunks)-1], 1)
	}

	return &model.TranscriptionResult{
		Text:     decoded.Text,
		Segments: segments,
		Language: language,
		Duration: duration,
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}
}

// chunkTime returns the chunk timestamp at index, in seconds, or zero.
func chunkTime(chunk transcriptionChunk, index int) time.Duration {
	if index >= len(chunk.Timestamp) {
		return 0
	}

	return time.Duration(chunk.Timestamp[index] * float64(time.Second))
}

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)
