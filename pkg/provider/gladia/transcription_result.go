package gladia

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// JobStatus is the processing state of a transcription job.
type JobStatus string

// Job statuses.
const (
	JobQueued     JobStatus = "queued"
	JobProcessing JobStatus = "processing"
	JobDone       JobStatus = "done"
	JobError      JobStatus = "error"
)

func (s JobStatus) validate() error {
	switch s {
	case JobQueued, JobProcessing, JobDone, JobError:
		return nil
	default:
		return fmt.Errorf("%w: gladia job status %q", model.ErrInvalidResponse, s)
	}
}

// TranscriptionMetadata is the provider metadata of a transcription: the
// finished job as Gladia reported it, with times in seconds.
type TranscriptionMetadata struct {
	Status JobStatus  `json:"status"`
	Result *JobResult `json:"result,omitempty"`
}

// JobResult is the result of a finished job.
type JobResult struct {
	Metadata      ResultMetadata      `json:"metadata"`
	Transcription ResultTranscription `json:"transcription"`
}

// ResultMetadata describes the transcribed audio.
type ResultMetadata struct {
	AudioDuration float64 `json:"audio_duration"`
}

// ResultTranscription is the transcript of a finished job.
type ResultTranscription struct {
	FullTranscript string      `json:"full_transcript"`
	Languages      []string    `json:"languages"`
	Utterances     []Utterance `json:"utterances"`
}

// Utterance is a timed span of speech.
type Utterance struct {
	Start      float64  `json:"start"`
	End        float64  `json:"end"`
	Text       string   `json:"text"`
	Speaker    *Speaker `json:"speaker,omitempty"`
	Confidence *float64 `json:"confidence,omitempty"`
	Language   *string  `json:"language,omitempty"`
	Words      []Word   `json:"words,omitempty"`
}

// Word is a timed word of an utterance.
type Word struct {
	Word       string   `json:"word"`
	Start      float64  `json:"start"`
	End        float64  `json:"end"`
	Confidence *float64 `json:"confidence,omitempty"`
}

// Speaker identifies the speaker of an utterance. Gladia reports it as a
// string or a number; a number keeps its JSON text.
type Speaker string

// UnmarshalJSON accepts a JSON string or number.
func (s *Speaker) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)

	if bytes.HasPrefix(trimmed, []byte(`"`)) {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return err
		}

		*s = Speaker(text)

		return nil
	}

	var number json.Number
	if err := json.Unmarshal(trimmed, &number); err != nil {
		return fmt.Errorf("speaker must be a string or a number: %w", err)
	}

	*s = Speaker(number.String())

	return nil
}

// transcriptionResult converts a finished job.
func transcriptionResult(job TranscriptionMetadata) (*model.TranscriptionResult, error) {
	if job.Result == nil {
		return nil, fmt.Errorf("%w: gladia transcription result is empty", model.ErrNoContentGenerated)
	}

	transcription := job.Result.Transcription

	segments := make([]model.TranscriptionSegment, 0, len(transcription.Utterances))
	for _, utterance := range transcription.Utterances {
		segments = append(segments, model.TranscriptionSegment{
			Text:  utterance.Text,
			Start: seconds(utterance.Start),
			End:   seconds(utterance.End),
		})
	}

	result := &model.TranscriptionResult{
		Text:     transcription.FullTranscript,
		Segments: segments,
		Duration: seconds(job.Result.Metadata.AudioDuration),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: job,
		},
	}

	if len(transcription.Languages) > 0 {
		result.Language = transcription.Languages[0]
	}

	return result, nil
}
