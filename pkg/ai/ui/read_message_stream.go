package ui

import (
	"errors"
	"iter"
)

// ReadMessageStreamOptions configure ReadMessageStream.
type ReadMessageStreamOptions struct {
	// Message is the assistant message to continue when a conversation is
	// resumed. Nil starts a new message.
	Message *Message

	Stream iter.Seq2[Chunk, error]

	// OnError is called with stream errors, error chunks and chunks that do
	// not fit the message.
	OnError func(err error)

	// TerminateOnError ends the stream at the first error and yields it,
	// including errors reported by error chunks. Otherwise error chunks are
	// only reported to OnError, and other errors end the stream without
	// being yielded.
	TerminateOnError bool
}

// ReadMessageStream rebuilds the assistant message from a UI message stream.
// It yields a snapshot of the message whenever it changes; each snapshot is
// a separate copy. A start chunk with a different message ID starts a new
// message.
func ReadMessageStream(options ReadMessageStreamOptions) iter.Seq2[Message, error] {
	return func(yield func(Message, error) bool) {
		messageID := ""
		if options.Message != nil {
			messageID = options.Message.ID
		}

		state := newMessageState(options.Message, messageID)
		state.resetOnMessageIDChange = true

		reportError := func(err error) {
			if options.OnError != nil {
				options.OnError(err)
			}

			if options.TerminateOnError {
				yield(Message{}, err)
			}
		}

		for chunk, err := range options.Stream {
			if err != nil {
				reportError(err)

				return
			}

			changed, err := state.apply(chunk)
			if err != nil {
				reportError(err)

				return
			}

			if errorChunk, isError := chunk.(ErrorChunk); isError {
				reportError(errors.New(errorChunk.ErrorText))

				if options.TerminateOnError {
					return
				}
			}

			if changed && !yield(state.message.clone(), nil) {
				return
			}
		}
	}
}
