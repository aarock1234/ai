package ai

import (
	"io"
	"sync/atomic"

	"github.com/aarock1234/ai/pkg/model"
)

// streamTranscribeOperationID identifies StreamTranscribe in telemetry.
const streamTranscribeOperationID = "ai.streamTranscribe"

// startEvent describes the stream before the model is called. The audio
// size is unknown until the stream ends.
func (r *StreamTranscribeResult) startEvent() TranscriptionStartEvent {
	inputAudioFormat := r.options.InputAudioFormat

	return TranscriptionStartEvent{
		CallID:           r.callID,
		OperationID:      streamTranscribeOperationID,
		Provider:         r.model.Provider(),
		ModelID:          r.model.ModelID(),
		AudioMediaType:   inputAudioFormat.Type,
		InputAudioFormat: &inputAudioFormat,
		Headers:          r.options.Headers,
		ProviderOptions:  r.options.ProviderOptions,
	}
}

// endEvent describes a stream that finished with a transcript, after
// audioByteLength bytes of audio were read.
func (r *StreamTranscribeResult) endEvent(
	result *TranscribeResult,
	finish *model.TranscriptFinish,
	audioByteLength int,
) TranscriptionEndEvent {
	return TranscriptionEndEvent{
		CallID:           r.callID,
		OperationID:      streamTranscribeOperationID,
		Provider:         r.model.Provider(),
		ModelID:          r.model.ModelID(),
		AudioByteLength:  audioByteLength,
		AudioMediaType:   r.options.InputAudioFormat.Type,
		Text:             result.Text,
		Segments:         result.Segments,
		Language:         result.Language,
		Duration:         result.Duration,
		Usage:            finish.Usage,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Response:         result.Responses[0],
	}
}

// countingReader counts the audio bytes the model reads. The model may
// read on its own goroutine, so the count is atomic.
type countingReader struct {
	reader io.Reader
	read   atomic.Int64
}

// Read reads from the audio and counts the bytes read.
func (c *countingReader) Read(buffer []byte) (int, error) {
	n, err := c.reader.Read(buffer)
	c.read.Add(int64(n))

	return n, err
}

// count returns the number of bytes read so far.
func (c *countingReader) count() int {
	return int(c.read.Load())
}
