package ai

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"regexp"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

const defaultSmoothStreamDelay = 10 * time.Millisecond

var (
	wordChunkPattern = regexp.MustCompile(`\S+\s+`)
	lineChunkPattern = regexp.MustCompile(`\n+`)
)

// ChunkDetector returns the first chunk of buffer, which must be a non-empty
// prefix of it, or the empty string when the buffer holds no complete chunk
// yet.
type ChunkDetector func(buffer string) (string, error)

// SmoothStreamOptions configures SmoothStream.
type SmoothStreamOptions struct {
	// Delay is the pause after each chunk. Nil selects 10ms; zero disables
	// the pause.
	Delay *time.Duration

	// Chunking splits the text into chunks. Nil selects WordChunking.
	Chunking ChunkDetector
}

// WordChunking emits one word, with its trailing whitespace, per chunk.
func WordChunking() ChunkDetector {
	return RegexpChunking(wordChunkPattern)
}

// LineChunking emits one line per chunk.
func LineChunking() ChunkDetector {
	return RegexpChunking(lineChunkPattern)
}

// RegexpChunking emits the text up to and including each match of pattern.
// The pattern must not match the empty string.
func RegexpChunking(pattern *regexp.Regexp) ChunkDetector {
	return func(buffer string) (string, error) {
		match := pattern.FindStringIndex(buffer)
		if match == nil {
			return "", nil
		}

		if match[0] == match[1] {
			return "", errors.New("chunking pattern must not match an empty string")
		}

		return buffer[:match[1]], nil
	}
}

// SmoothStream returns a transform that re-chunks text and reasoning deltas,
// by default into words, with a short pause between chunks. Other parts
// flush the buffered text and pass through unchanged.
func SmoothStream(options SmoothStreamOptions) StreamTransform {
	delay := defaultSmoothStreamDelay
	if options.Delay != nil {
		delay = *options.Delay
	}

	detect := options.Chunking
	if detect == nil {
		detect = WordChunking()
	}

	return func(ctx context.Context, input StreamTransformInput) iter.Seq[StreamPart] {
		return func(yield func(StreamPart) bool) {
			smoother := &streamSmoother{
				delay:  delay,
				detect: detect,
			}

			for part := range input.Parts {
				if !smoother.process(ctx, part, yield) {
					return
				}
			}

			smoother.flush(yield)
		}
	}
}

// streamSmoother buffers the text of consecutive deltas of one block.
type streamSmoother struct {
	delay  time.Duration
	detect ChunkDetector

	buffer    string
	id        string
	reasoning bool
	hasBlock  bool
	metadata  model.ProviderMetadata
}

// smoothableDelta is a text or reasoning delta.
type smoothableDelta struct {
	id        string
	text      string
	reasoning bool
	metadata  model.ProviderMetadata
}

func asSmoothableDelta(part StreamPart) (smoothableDelta, bool) {
	switch typed := part.(type) {
	case TextDeltaPart:
		return smoothableDelta{
			id:       typed.ID,
			text:     typed.Text,
			metadata: typed.ProviderMetadata,
		}, true

	case ReasoningDeltaPart:
		return smoothableDelta{
			id:        typed.ID,
			text:      typed.Text,
			reasoning: true,
			metadata:  typed.ProviderMetadata,
		}, true

	default:
		return smoothableDelta{}, false
	}
}

func (s *streamSmoother) process(ctx context.Context, part StreamPart, yield func(StreamPart) bool) bool {
	delta, ok := asSmoothableDelta(part)

	isMetadataOnly := ok && delta.text == "" && delta.metadata != nil
	if !ok || isMetadataOnly {
		return s.flush(yield) && yield(part)
	}

	// One emitted delta cannot carry the metadata of several input deltas,
	// so metadata and block changes end the buffered chunk.
	startsNewChunk := delta.reasoning != s.reasoning || delta.id != s.id || s.metadata != nil || delta.metadata != nil
	if s.buffer != "" && startsNewChunk && !s.flush(yield) {
		return false
	}

	s.buffer += delta.text
	s.id = delta.id
	s.reasoning = delta.reasoning
	s.hasBlock = true
	s.metadata = delta.metadata

	for {
		chunk, err := s.detect(s.buffer)
		if err != nil {
			yield(ErrorPart{
				Err: err,
			})

			return false
		}

		if chunk == "" {
			break
		}

		if !strings.HasPrefix(s.buffer, chunk) {
			yield(ErrorPart{
				Err: fmt.Errorf("chunk detector returned %q, which is not a prefix of the buffer", chunk),
			})

			return false
		}

		if !yield(s.delta(chunk)) {
			return false
		}

		s.buffer = s.buffer[len(chunk):]

		if err := sleep(ctx, s.delay); err != nil {
			// Keep forwarding without pauses so that the abort part
			// still arrives.
			s.delay = 0
		}
	}

	if s.buffer == "" {
		s.metadata = nil
	}

	return true
}

// flush emits the buffered text, or buffered metadata without text.
func (s *streamSmoother) flush(yield func(StreamPart) bool) bool {
	if !s.hasBlock || (s.buffer == "" && s.metadata == nil) {
		return true
	}

	part := s.delta(s.buffer)
	s.buffer = ""
	s.metadata = nil

	return yield(part)
}

func (s *streamSmoother) delta(text string) StreamPart {
	if s.reasoning {
		return ReasoningDeltaPart{
			ID:               s.id,
			Text:             text,
			ProviderMetadata: s.metadata,
		}
	}

	return TextDeltaPart{
		ID:               s.id,
		Text:             text,
		ProviderMetadata: s.metadata,
	}
}
