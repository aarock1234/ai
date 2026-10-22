package ui

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/aarock1234/ai/pkg/model"
)

// ChatTrigger is why a chat request is sent.
type ChatTrigger string

// Chat triggers.
const (
	// ChatTriggerSubmitMessage sends a new user message or continues the
	// last assistant message, for example after tool approvals.
	ChatTriggerSubmitMessage ChatTrigger = "submit-message"

	// ChatTriggerRegenerateMessage regenerates the assistant message.
	ChatTriggerRegenerateMessage ChatTrigger = "regenerate-message"
)

// ChatRequestOptions are per-request additions to a chat request, the
// upstream ChatRequestOptions.
type ChatRequestOptions struct {
	// Headers are added to the transport's headers, replacing headers of
	// the same name.
	Headers http.Header

	// Body members are added to the request body, replacing the transport's
	// body members of the same name.
	Body map[string]json.RawMessage

	// Metadata is passed to the transport's request preparation hooks. It
	// is not sent by default.
	Metadata json.RawMessage
}

// SendMessagesOptions configure ChatTransport.SendMessages.
type SendMessagesOptions struct {
	Trigger ChatTrigger
	ChatID  string

	// MessageID is the ID of the message to regenerate, or empty.
	MessageID string

	// Messages is the whole conversation.
	Messages []Message

	ChatRequestOptions
}

// ReconnectToStreamOptions configure ChatTransport.ReconnectToStream.
type ReconnectToStreamOptions struct {
	ChatID string

	ChatRequestOptions
}

// ChatTransport exchanges UI messages with a chat backend, the client side
// of the UI message stream protocol. SendMessages and ReconnectToStream
// return a stream that the caller ranges over once and closes.
//
// Go adaptation: upstream's abort signal is the context. Canceling it ends
// the request and its stream.
type ChatTransport interface {
	// SendMessages sends the conversation and returns the stream of the
	// assistant's response.
	SendMessages(ctx context.Context, options SendMessagesOptions) (*ChatStream, error)

	// ReconnectToStream resumes the active response stream of a chat. It
	// returns nil without an error when the chat has no active stream.
	ReconnectToStream(ctx context.Context, options ReconnectToStreamOptions) (*ChatStream, error)
}

// ChatStream is the UI message stream of one chat response. It owns the
// response body, which Chunks releases when the range ends, including on an
// early break, and which Close releases otherwise.
type ChatStream struct {
	chunks    iter.Seq2[Chunk, error]
	closeBody func() error

	started   atomic.Bool
	closeOnce sync.Once
	closeErr  error
}

// newChatStream returns a stream of chunks read from a body that closeBody
// releases.
func newChatStream(chunks iter.Seq2[Chunk, error], closeBody func() error) *ChatStream {
	return &ChatStream{
		chunks:    chunks,
		closeBody: closeBody,
	}
}

// Chunks returns the chunks of the response. It can be ranged over once; a
// second iteration yields a single error wrapping model.ErrStreamConsumed.
// An error ends it.
func (s *ChatStream) Chunks() iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		if !s.started.CompareAndSwap(false, true) {
			yield(nil, model.ErrStreamConsumed)

			return
		}

		defer s.closeAfterIteration()

		for chunk, err := range s.chunks {
			if !yield(chunk, err) || err != nil {
				return
			}
		}
	}
}

// closeAfterIteration releases the response when iteration ends. The
// release error is kept and returned by Close.
func (s *ChatStream) closeAfterIteration() {
	_ = s.Close()
}

// Close releases the response. It is safe to call more than once and from
// any goroutine, and returns the release error, if any.
func (s *ChatStream) Close() error {
	s.closeOnce.Do(func() {
		if s.closeBody != nil {
			s.closeErr = s.closeBody()
		}
	})

	return s.closeErr
}

// ErrInvalidChatID reports a chat ID that the default reconnect URL cannot
// contain.
var ErrInvalidChatID = errors.New(`chat ids must not be "." or ".." when using the default reconnect url`)
