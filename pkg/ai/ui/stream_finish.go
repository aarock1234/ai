package ui

import (
	"context"
	"errors"
	"iter"

	"github.com/aarock1234/ai/pkg/model"
)

// OutcomeStatus is the operation-level status of a UI message stream.
type OutcomeStatus string

// Outcome statuses.
const (
	OutcomeUnknown   OutcomeStatus = "unknown"
	OutcomeCompleted OutcomeStatus = "completed"
	OutcomeFailed    OutcomeStatus = "failed"
	OutcomeAborted   OutcomeStatus = "aborted"
)

// Outcome is the operation-level outcome of a UI message stream. It is
// separate from model finish reasons and individual chunks. A stream whose
// consumer stopped before an outcome was declared keeps the unknown status.
type Outcome struct {
	Status OutcomeStatus

	// Error is the failure of a failed stream, if known.
	Error error
}

// FinishEvent describes a UI message stream that has ended.
type FinishEvent struct {
	// Messages are the original messages followed by the response message,
	// which replaces the last original message when it continues it.
	Messages []Message

	// IsContinuation reports whether the response message continues the
	// last original message.
	IsContinuation bool

	// IsAborted reports whether the stream was aborted.
	IsAborted bool

	// IsCancelled reports that the consumer stopped reading before an outcome
	// was declared, for example because the client disconnected.
	IsCancelled bool

	// Outcome is the outcome of the stream. Fatal stream processing failures
	// override the outcome declared by the stream owner.
	Outcome Outcome

	// ResponseMessage is the message sent to the client, including the
	// original message when it was extended.
	ResponseMessage Message

	// FinishReason is the reason of the finish chunk, if any.
	FinishReason model.UnifiedFinishReason
}

// StepFinishEvent describes the response message after a step, for
// persisting intermediate messages of multi-step runs.
type StepFinishEvent struct {
	Messages        []Message
	IsContinuation  bool
	ResponseMessage Message
}

// streamFinishOptions configure handleMessageStreamFinish.
type streamFinishOptions struct {
	// messageID is the ID of a new response message. It is injected into
	// start chunks without one.
	messageID string

	originalMessages []Message

	onStepFinish func(ctx context.Context, event StepFinishEvent)
	onFinish     func(ctx context.Context, event FinishEvent)
	onError      func(err error) string

	// outcome returns the outcome declared by the stream owner.
	outcome func() Outcome
}

// handleMessageStreamFinish injects the response message ID into the start
// chunk and, when callbacks are set, builds the response message to report
// it after each step and once the stream ends, however it ends.
func handleMessageStreamFinish(ctx context.Context, stream iter.Seq2[Chunk, error], options streamFinishOptions) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		lastMessage := continuedMessage(options.originalMessages)

		messageID := options.messageID
		if lastMessage != nil {
			messageID = lastMessage.ID
		}

		tracker := &finishTracker{
			options:     options,
			lastMessage: lastMessage,
		}

		if options.onStepFinish != nil || options.onFinish != nil {
			tracker.state = newMessageState(lastMessage, messageID)
		}

		defer tracker.finish(ctx)

		for chunk, err := range stream {
			if err != nil {
				tracker.processingErr = err
				yield(nil, err)

				return
			}

			chunk = withMessageID(chunk, messageID)

			if err := tracker.observe(ctx, chunk); err != nil {
				tracker.processingErr = err
				yield(nil, err)

				return
			}

			if !yield(chunk, nil) {
				tracker.consumerStopped = true

				return
			}
		}
	}
}

// finishTracker follows a stream for handleMessageStreamFinish.
type finishTracker struct {
	options     streamFinishOptions
	lastMessage *Message

	// state is nil when no callback needs the response message.
	state *messageState

	aborted         bool
	consumerStopped bool
	processingErr   error
}

// observe updates the response message with a chunk and reports finished
// steps.
func (t *finishTracker) observe(ctx context.Context, chunk Chunk) error {
	if _, isAbort := chunk.(AbortChunk); isAbort {
		t.aborted = true
	}

	if t.state == nil {
		return nil
	}

	if _, err := t.state.apply(chunk); err != nil {
		return err
	}

	switch typed := chunk.(type) {
	case ErrorChunk:
		if t.options.onError != nil {
			t.options.onError(errors.New(typed.ErrorText))
		}

	case FinishStepChunk:
		if t.options.onStepFinish != nil {
			t.options.onStepFinish(ctx, t.stepFinishEvent())
		}
	}

	return nil
}

func (t *finishTracker) stepFinishEvent() StepFinishEvent {
	responseMessage := t.state.message.clone()
	isContinuation := t.isContinuation()

	return StepFinishEvent{
		Messages:        t.messagesWith(responseMessage, isContinuation),
		IsContinuation:  isContinuation,
		ResponseMessage: responseMessage,
	}
}

// finish calls the finish callback once the stream has ended.
func (t *finishTracker) finish(ctx context.Context) {
	if t.options.onFinish == nil {
		return
	}

	outcome := t.outcome()
	responseMessage := t.state.message.clone()
	isContinuation := t.isContinuation()

	t.options.onFinish(ctx, FinishEvent{
		Messages:        t.messagesWith(responseMessage, isContinuation),
		IsContinuation:  isContinuation,
		IsAborted:       t.aborted || outcome.Status == OutcomeAborted,
		IsCancelled:     t.consumerStopped && outcome.Status == OutcomeUnknown,
		Outcome:         outcome,
		ResponseMessage: responseMessage,
		FinishReason:    t.state.finishReason,
	})
}

// outcome combines the declared outcome with processing failures and abort
// chunks.
func (t *finishTracker) outcome() Outcome {
	if t.processingErr != nil {
		return Outcome{
			Status: OutcomeFailed,
			Error:  t.processingErr,
		}
	}

	declared := Outcome{
		Status: OutcomeUnknown,
	}
	if t.options.outcome != nil {
		declared = t.options.outcome()
	}

	if declared.Status == OutcomeUnknown && t.aborted {
		return Outcome{
			Status: OutcomeAborted,
		}
	}

	return declared
}

func (t *finishTracker) isContinuation() bool {
	return t.lastMessage != nil && t.state.message.ID == t.lastMessage.ID
}

// messagesWith returns the original messages followed by the response
// message, replacing the last original message for continuations.
func (t *finishTracker) messagesWith(responseMessage Message, isContinuation bool) []Message {
	original := t.options.originalMessages
	if isContinuation {
		original = original[:len(original)-1]
	}

	messages := make([]Message, 0, len(original)+1)
	messages = append(messages, original...)

	return append(messages, responseMessage)
}

// continuedMessage returns the last original message when it is an
// assistant message that the response continues.
func continuedMessage(originalMessages []Message) *Message {
	if len(originalMessages) == 0 {
		return nil
	}

	last := originalMessages[len(originalMessages)-1]
	if last.Role != RoleAssistant {
		return nil
	}

	return &last
}

// withMessageID sets the message ID of a start chunk that has none.
func withMessageID(chunk Chunk, messageID string) Chunk {
	start, isStart := chunk.(StartChunk)
	if !isStart || start.MessageID != "" || messageID == "" {
		return chunk
	}

	start.MessageID = messageID

	return start
}

// responseMessageID returns the ID of the response message in persistence
// mode, when original messages are provided: the ID of a continued
// assistant message, or a generated one. It returns the empty string
// otherwise, leaving ID generation to the client.
func responseMessageID(originalMessages []Message, generateID func() string) string {
	if originalMessages == nil {
		return ""
	}

	if last := continuedMessage(originalMessages); last != nil {
		return last.ID
	}

	return generateID()
}
