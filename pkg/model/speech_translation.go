package model

import (
	"context"
	"io"
	"iter"
	"net/http"
	"time"
)

// SpeechTranslationModelKind names speech translation models in errors and
// registries.
const SpeechTranslationModelKind Kind = "speech translation model"

// SpeechTranslationModel translates live source audio into audio and text
// in a target language. Speech translation only streams. The contract is
// experimental and may change.
type SpeechTranslationModel interface {
	Provider() string
	ModelID() string

	// StreamTranslation opens the stream. Connection and option errors are
	// returned directly; failures after the stream has opened are reported
	// as StreamError parts.
	StreamTranslation(ctx context.Context, options SpeechTranslationStreamOptions) (*SpeechTranslationStreamResult, error)
}

// SpeechTranslationProvider is implemented by providers that offer speech
// translation models. It is separate from Provider because most providers
// have none.
type SpeechTranslationProvider interface {
	SpeechTranslationModel(modelID string) (SpeechTranslationModel, error)
}

// SpeechTranslationStreamOptions are the inputs of one speech translation
// stream.
type SpeechTranslationStreamOptions struct {
	// Audio supplies raw audio bytes in InputAudioFormat. The stream reads it
	// until io.EOF, on its own goroutine, while translations arrive; any read
	// error ends the stream. The caller keeps ownership: the stream never
	// closes Audio and stops reading it when the stream ends, but cannot
	// interrupt a Read already in progress.
	Audio io.Reader

	InputAudioFormat AudioFormat

	// TargetLanguage is the BCP-47 style language of the output, such as
	// "en", "es" or "fr-CA". Supported values are provider-specific.
	TargetLanguage string

	// SourceLanguage is the language of the audio. Empty asks the provider
	// to detect it.
	SourceLanguage string

	// OutputAudioFormat is nil for the provider default.
	OutputAudioFormat *AudioFormat

	Headers         http.Header
	ProviderOptions ProviderOptions

	// IncludeRawChunks asks the provider to emit Raw parts.
	IncludeRawChunks bool
}

// SpeechTranslationStreamPart is one event of a speech translation stream.
// A stream begins with StreamStart and ends with TranslationFinish when it
// completes. TranslationAudio, OutputTextDelta, OutputTextFinal,
// SourceTranscriptDelta, SourceTranscriptPartial, SourceTranscriptFinal,
// ResponseMetadata, Raw and StreamError parts may appear between.
type SpeechTranslationStreamPart interface {
	isSpeechTranslationStreamPart()
}

// TranslationAudio is a chunk of translated audio. Providers decode base64
// audio into raw bytes.
type TranslationAudio struct {
	// ID identifies the provider segment or utterance, if any.
	ID string

	Audio            []byte
	ProviderMetadata ProviderMetadata
}

// OutputTextDelta appends to the translated text. Output text is
// append-only and finalized per utterance with OutputTextFinal.
type OutputTextDelta struct {
	ID               string
	Delta            string
	ProviderMetadata ProviderMetadata
}

// OutputTextFinal is the final translated text of a provider-defined
// segment or utterance.
type OutputTextFinal struct {
	ID               string
	Text             string
	ProviderMetadata ProviderMetadata
}

// SourceTranscriptDelta appends to the transcript of the source audio.
type SourceTranscriptDelta struct {
	ID               string
	Delta            string
	ProviderMetadata ProviderMetadata
}

// SourceTranscriptPartial is non-final source transcript text that later
// parts may revise. Nil fields were not reported.
type SourceTranscriptPartial struct {
	ID   string
	Text string

	Start        *time.Duration
	End          *time.Duration
	ChannelIndex *int

	ProviderMetadata ProviderMetadata
}

// SourceTranscriptFinal is the final source transcript text of a
// provider-defined segment or utterance. Nil fields were not reported.
type SourceTranscriptFinal struct {
	ID   string
	Text string

	Start        *time.Duration
	End          *time.Duration
	ChannelIndex *int

	ProviderMetadata ProviderMetadata
}

// TranslationFinish ends the stream with the complete texts.
type TranslationFinish struct {
	// SourceText is the final transcript of the source audio.
	SourceText string

	// OutputText is the final translated text. It may be empty for
	// providers that produce only audio.
	OutputText string

	// Duration is the length of the source audio, or zero when unknown.
	Duration time.Duration

	// Usage is nil when the provider did not report it.
	Usage *SpeechTranslationUsage

	ProviderMetadata ProviderMetadata
}

// SpeechTranslationUsage reports the resources a translation used. Nil
// counts were not reported; providers report usage with different
// granularity.
type SpeechTranslationUsage struct {
	InputAudioDuration *time.Duration
	InputAudioTokens   *int
	OutputAudioTokens  *int
	InputTextTokens    *int
	OutputTextTokens   *int
}

func (StreamStart) isSpeechTranslationStreamPart()             {}
func (TranslationAudio) isSpeechTranslationStreamPart()        {}
func (OutputTextDelta) isSpeechTranslationStreamPart()         {}
func (OutputTextFinal) isSpeechTranslationStreamPart()         {}
func (SourceTranscriptDelta) isSpeechTranslationStreamPart()   {}
func (SourceTranscriptPartial) isSpeechTranslationStreamPart() {}
func (SourceTranscriptFinal) isSpeechTranslationStreamPart()   {}
func (ResponseMetadata) isSpeechTranslationStreamPart()        {}
func (TranslationFinish) isSpeechTranslationStreamPart()       {}
func (Raw) isSpeechTranslationStreamPart()                     {}
func (StreamError) isSpeechTranslationStreamPart()             {}

// SpeechTranslationStreamResult is the outcome of a speech translation
// stream call. It owns the underlying connection: ranging over Parts
// releases it when iteration ends, including an early break, and Close
// releases it without iterating.
type SpeechTranslationStreamResult struct {
	// Request records the setup payload sent to the provider, if any.
	Request RequestInfo

	// Response describes the connection response.
	Response ResponseInfo

	parts  iter.Seq[SpeechTranslationStreamPart]
	stream singleUseStream
}

// NewSpeechTranslationStreamResult returns a stream over parts. Release,
// which may be nil, is called exactly once when the stream ends or is
// closed.
func NewSpeechTranslationStreamResult(parts iter.Seq[SpeechTranslationStreamPart], release func() error) *SpeechTranslationStreamResult {
	return &SpeechTranslationStreamResult{
		parts: parts,
		stream: singleUseStream{
			release: release,
		},
	}
}

// Parts returns the stream's parts. The sequence can be ranged over once; a
// second iteration yields a single StreamError wrapping ErrStreamConsumed.
func (r *SpeechTranslationStreamResult) Parts() iter.Seq[SpeechTranslationStreamPart] {
	return func(yield func(SpeechTranslationStreamPart) bool) {
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
func (r *SpeechTranslationStreamResult) closeAfterIteration() {
	_ = r.Close()
}

// Close releases the stream. It is safe to call more than once and returns
// the release error, if any.
func (r *SpeechTranslationStreamResult) Close() error {
	return r.stream.close()
}
