package model

import (
	"context"
	"encoding/json"
	"io"
	"iter"
	"net/http"
	"time"
)

// TranscriptionModel converts audio into text.
type TranscriptionModel interface {
	Provider() string
	ModelID() string

	Transcribe(ctx context.Context, options TranscriptionOptions) (*TranscriptionResult, error)
}

// StreamingTranscriptionModel is a TranscriptionModel that also transcribes
// live audio as it arrives. The streaming contract is experimental and may
// change.
type StreamingTranscriptionModel interface {
	TranscriptionModel

	// StreamTranscribe opens the stream. Connection and option errors are
	// returned directly; failures after the stream has opened are reported
	// as StreamError parts.
	StreamTranscribe(ctx context.Context, options TranscriptionStreamOptions) (*TranscriptionStreamResult, error)
}

// TranscriptionOptions are the inputs of one transcription call.
type TranscriptionOptions struct {
	Audio     []byte
	MediaType string

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// TranscriptionResult is the outcome of one transcription call.
type TranscriptionResult struct {
	Text     string
	Segments []TranscriptionSegment

	// Language is the detected language code, if reported.
	Language string

	// Duration is zero when not reported.
	Duration time.Duration

	// Usage is the provider's native usage object, if any.
	Usage json.RawMessage

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
	Request          RequestInfo
	Response         ResponseInfo
}

// TranscriptionSegment is a timed span of the transcript.
type TranscriptionSegment struct {
	Text  string
	Start time.Duration
	End   time.Duration
}

// AudioFormat describes raw audio, such as {Type: "audio/pcm", Rate: 24000}.
type AudioFormat struct {
	// Type is the audio media type, such as "audio/pcm", "audio/pcmu" or
	// "audio/pcma".
	Type string

	// Rate is the sample rate in Hz, or zero for formats without one.
	Rate int
}

// TranscriptionStreamOptions are the inputs of one streaming transcription
// call.
type TranscriptionStreamOptions struct {
	// Audio supplies raw audio bytes in InputAudioFormat. The stream reads it
	// until io.EOF, on its own goroutine, while transcripts arrive; any read
	// error ends the stream. Chunk boundaries carry no meaning. The caller
	// keeps ownership: the stream never closes Audio and stops reading it
	// when the stream ends, but cannot interrupt a Read already in progress.
	Audio io.Reader

	InputAudioFormat AudioFormat

	Headers         http.Header
	ProviderOptions ProviderOptions

	// IncludeRawChunks asks the provider to emit Raw parts.
	IncludeRawChunks bool
}

// TranscriptionStreamPart is one event of a streaming transcription. A
// stream begins with StreamStart and ends with TranscriptFinish when it
// completes. ResponseMetadata, Raw and StreamError parts may appear between.
type TranscriptionStreamPart interface {
	isTranscriptionStreamPart()
}

// TranscriptDelta appends text to the transcript.
type TranscriptDelta struct {
	// ID identifies the provider segment or utterance, if any.
	ID string

	Delta            string
	ProviderMetadata ProviderMetadata
}

// TranscriptPartial is non-final transcript text that later parts may
// revise. Nil fields were not reported.
type TranscriptPartial struct {
	ID   string
	Text string

	Start        *time.Duration
	Duration     *time.Duration
	ChannelIndex *int

	ProviderMetadata ProviderMetadata
}

// TranscriptFinal is the final text of a provider-defined segment or
// utterance. Nil fields were not reported.
type TranscriptFinal struct {
	ID   string
	Text string

	Start        *time.Duration
	End          *time.Duration
	ChannelIndex *int

	ProviderMetadata ProviderMetadata
}

// TranscriptFinish ends the stream with the complete transcript.
type TranscriptFinish struct {
	Text     string
	Segments []TranscriptionSegment

	// Language is the detected language code, if reported.
	Language string

	// Duration is zero when not reported.
	Duration time.Duration

	// Usage is the provider's native usage object, if any.
	Usage json.RawMessage

	ProviderMetadata ProviderMetadata
}

func (StreamStart) isTranscriptionStreamPart()       {}
func (TranscriptDelta) isTranscriptionStreamPart()   {}
func (TranscriptPartial) isTranscriptionStreamPart() {}
func (TranscriptFinal) isTranscriptionStreamPart()   {}
func (ResponseMetadata) isTranscriptionStreamPart()  {}
func (TranscriptFinish) isTranscriptionStreamPart()  {}
func (Raw) isTranscriptionStreamPart()               {}
func (StreamError) isTranscriptionStreamPart()       {}

// TranscriptionStreamResult is the outcome of a streaming transcription
// call. It owns the underlying connection: ranging over Parts releases it
// when iteration ends, including an early break, and Close releases it
// without iterating.
type TranscriptionStreamResult struct {
	// Request records the setup payload sent to the provider, if any.
	Request RequestInfo

	// Response describes the connection response.
	Response ResponseInfo

	parts  iter.Seq[TranscriptionStreamPart]
	stream singleUseStream
}

// NewTranscriptionStreamResult returns a stream over parts. Release, which
// may be nil, is called exactly once when the stream ends or is closed.
func NewTranscriptionStreamResult(parts iter.Seq[TranscriptionStreamPart], release func() error) *TranscriptionStreamResult {
	return &TranscriptionStreamResult{
		parts: parts,
		stream: singleUseStream{
			release: release,
		},
	}
}

// Parts returns the stream's parts. The sequence can be ranged over once; a
// second iteration yields a single StreamError wrapping ErrStreamConsumed.
func (r *TranscriptionStreamResult) Parts() iter.Seq[TranscriptionStreamPart] {
	return func(yield func(TranscriptionStreamPart) bool) {
		if !r.stream.claim() {
			yield(StreamError{
				Err: ErrStreamConsumed,
			})

			return
		}

		defer r.closeAfterIteration()

		for part := range r.parts {
			if !yield(part) {
				return
			}
		}
	}
}

// closeAfterIteration releases the stream when iteration ends. The release
// error is kept and returned by Close.
func (r *TranscriptionStreamResult) closeAfterIteration() {
	_ = r.Close()
}

// Close releases the stream. It is safe to call more than once and returns
// the release error, if any.
func (r *TranscriptionStreamResult) Close() error {
	return r.stream.close()
}
