package model

import (
	"iter"
	"net/http"
)

// StreamResult is the outcome of a streaming language model call. It owns the
// underlying connection: ranging over Parts releases it when iteration ends,
// including an early break, and Close releases it without iterating.
type StreamResult struct {
	Request  RequestInfo
	Response StreamResponseInfo

	parts  iter.Seq[StreamPart]
	stream singleUseStream
}

// StreamResponseInfo describes the HTTP response of a streaming call.
type StreamResponseInfo struct {
	Headers http.Header
}

// NewStreamResult returns a stream over parts. Release, which may be nil, is
// called exactly once when the stream ends or is closed.
func NewStreamResult(parts iter.Seq[StreamPart], release func() error) *StreamResult {
	return &StreamResult{
		parts: parts,
		stream: singleUseStream{
			release: release,
		},
	}
}

// Parts returns the stream's parts. The sequence can be ranged over once; a
// second iteration yields a single StreamError wrapping ErrStreamConsumed.
func (r *StreamResult) Parts() iter.Seq[StreamPart] {
	return func(yield func(StreamPart) bool) {
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
func (r *StreamResult) closeAfterIteration() {
	_ = r.Close()
}

// Close releases the stream. It is safe to call more than once and returns
// the release error, if any.
func (r *StreamResult) Close() error {
	return r.stream.close()
}
