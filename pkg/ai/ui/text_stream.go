package ui

import (
	"io"
	"iter"
	"net/http"
)

// textStreamPartID is the ID of the single text part of a text stream
// converted to a UI message stream.
const textStreamPartID = "text-1"

// TextStreamResponseOptions configure PipeTextStreamToResponse and
// CreateTextStreamResponse.
type TextStreamResponseOptions struct {
	// Stream yields the text deltas, such as StreamTextResult.TextStream.
	Stream iter.Seq[string]

	ResponseInit
}

// textStreamHeaders returns the response headers of plain text streams.
func textStreamHeaders() http.Header {
	return http.Header{
		"Content-Type": {"text/plain; charset=utf-8"},
	}
}

// PipeTextStreamToResponse writes each text delta to w as UTF-8 and flushes
// it. It returns once the stream has ended or a write has failed.
func PipeTextStreamToResponse(w http.ResponseWriter, options TextStreamResponseOptions) error {
	writeResponseHead(w, options.ResponseInit, textStreamHeaders())

	destination := &frameDestination{
		writer: w,
		flush:  responseFlusher(w),
	}

	return pipeFrames(destination, textFrames(options.Stream), 0)
}

// CreateTextStreamResponse returns an HTTP response whose body streams the
// text deltas. The stream starts on the first read of the body; closing the
// body stops it and waits for it to end.
func CreateTextStreamResponse(options TextStreamResponseOptions) *http.Response {
	body := newStreamBody(func(w io.Writer) error {
		destination := &frameDestination{
			writer: w,
		}

		return pipeFrames(destination, textFrames(options.Stream), 0)
	})

	return newStreamResponse(options.ResponseInit, textStreamHeaders(), body)
}

// textFrames returns the deltas as frames.
func textFrames(stream iter.Seq[string]) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		for text := range stream {
			if !yield([]byte(text), nil) {
				return
			}
		}
	}
}

// TextToMessageStream converts a stream of text deltas into a UI message
// stream of a single step with one text part.
func TextToMessageStream(stream iter.Seq[string]) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		opening := []Chunk{
			StartChunk{},
			StartStepChunk{},
			TextStartChunk{
				ID: textStreamPartID,
			},
		}

		for _, chunk := range opening {
			if !yield(chunk, nil) {
				return
			}
		}

		for text := range stream {
			delta := TextDeltaChunk{
				ID:    textStreamPartID,
				Delta: text,
			}

			if !yield(delta, nil) {
				return
			}
		}

		closing := []Chunk{
			TextEndChunk{
				ID: textStreamPartID,
			},
			FinishStepChunk{},
			FinishChunk{},
		}

		for _, chunk := range closing {
			if !yield(chunk, nil) {
				return
			}
		}
	}
}
