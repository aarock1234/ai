package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/aarock1234/ai/internal/optional"

	"github.com/aarock1234/ai/pkg/ai"
)

// defaultErrorText replaces error details sent to the client unless an
// OnError function is set, so that server errors do not leak.
const defaultErrorText = "An error occurred."

// MessageStreamOptions configure ToMessageStream, the upstream
// UIMessageStreamOptions.
type MessageStreamOptions struct {
	// Tools are the tools of the generation. They decide which tool chunks
	// are marked as dynamic.
	Tools []ai.Tool

	// SendReasoning sends reasoning parts. Nil sends them.
	SendReasoning *bool

	// SendSources sends source parts.
	SendSources bool

	// SendStart sends the start chunk. Nil sends it. Disable it when the
	// stream continues a message that another stream started.
	SendStart *bool

	// SendFinish sends the finish chunk. Nil sends it. Disable it when
	// another stream continues the message.
	SendFinish *bool

	// OnError formats stream errors, invalid tool calls and the errors of
	// application-executed tools for the client. Nil sends a generic
	// message. Errors of provider-executed tools are sent unchanged.
	OnError func(err error) string

	// MessageMetadata returns the message metadata to send for a stream
	// part. Nil sends none. Start and finish chunks carry the metadata;
	// other parts are followed by a message-metadata chunk.
	MessageMetadata func(ctx context.Context, event MessageMetadataEvent) any

	// OriginalMessages are the messages of the conversation. When set,
	// persistence mode is assumed and the response message receives an ID:
	// the ID of a continued assistant message or one from
	// GenerateMessageID.
	OriginalMessages []Message

	// GenerateMessageID generates the ID of a new response message. Nil
	// leaves ID generation to the client.
	GenerateMessageID func() string

	// OnStepFinish is called with the response message after each step.
	OnStepFinish func(ctx context.Context, event StepFinishEvent)

	// OnFinish is called once the stream has ended.
	OnFinish func(ctx context.Context, event FinishEvent)
}

// MessageMetadataEvent carries the stream part for which metadata is
// requested.
type MessageMetadataEvent struct {
	Part ai.StreamPart
}

// ToMessageStream converts the parts of a StreamText stream, such as
// StreamTextResult.FullStream, into a UI message stream. Callbacks receive
// ctx.
func ToMessageStream(ctx context.Context, stream iter.Seq[ai.StreamPart], options MessageStreamOptions) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		var messageID string
		if options.GenerateMessageID != nil {
			messageID = responseMessageID(options.OriginalMessages, options.GenerateMessageID)
		}

		conversion := &partConversion{
			converter: chunkConverter{
				tools:             options.Tools,
				sendReasoning:     isEnabled(options.SendReasoning),
				sendSources:       options.SendSources,
				sendStart:         isEnabled(options.SendStart),
				sendFinish:        isEnabled(options.SendFinish),
				onError:           errorFormatter(options.OnError),
				responseMessageID: messageID,
			},
			messageMetadata: options.MessageMetadata,
			outcome: Outcome{
				Status: OutcomeUnknown,
			},
		}

		finishMessageID := messageID
		if finishMessageID == "" && options.GenerateMessageID != nil {
			finishMessageID = options.GenerateMessageID()
		}

		finished := handleMessageStreamFinish(ctx, conversion.chunks(ctx, stream), streamFinishOptions{
			messageID:        finishMessageID,
			originalMessages: options.OriginalMessages,
			onStepFinish:     options.OnStepFinish,
			onFinish:         options.OnFinish,
			onError:          conversion.converter.onError,
			outcome:          conversion.currentOutcome,
		})

		for chunk, err := range finished {
			if !yield(chunk, err) {
				return
			}
		}
	}
}

// partConversion converts stream parts and tracks the outcome they declare.
type partConversion struct {
	converter       chunkConverter
	messageMetadata func(ctx context.Context, event MessageMetadataEvent) any

	outcome      Outcome
	fatalFailure bool
}

// chunks converts the parts of stream.
func (c *partConversion) chunks(ctx context.Context, stream iter.Seq[ai.StreamPart]) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		for part := range stream {
			converted, err := c.convert(ctx, part)
			if err != nil {
				c.fail(err)
				yield(nil, err)

				return
			}

			c.declareOutcome(part)

			for _, chunk := range converted {
				if !yield(chunk, nil) {
					return
				}
			}
		}
	}
}

// convert returns the chunks of one part: its own chunk, if any, and a
// message-metadata chunk when metadata was produced for a part other than
// the start and finish parts, which carry it themselves. A finish-step chunk
// follows the metadata, so that the step's message includes it.
func (c *partConversion) convert(ctx context.Context, part ai.StreamPart) ([]Chunk, error) {
	metadata, err := c.metadata(ctx, part)
	if err != nil {
		return nil, err
	}

	chunk, ok, err := c.converter.convert(part, metadata)
	if err != nil {
		return nil, err
	}

	_, isFinishStep := part.(ai.FinishStepPart)

	var chunks []Chunk
	if ok && !isFinishStep {
		chunks = append(chunks, chunk)
	}

	if metadata != nil && !carriesMetadata(part) {
		chunks = append(chunks, MessageMetadataChunk{
			MessageMetadata: metadata,
		})
	}

	if ok && isFinishStep {
		chunks = append(chunks, chunk)
	}

	return chunks, nil
}

func carriesMetadata(part ai.StreamPart) bool {
	switch part.(type) {
	case ai.StartPart, ai.FinishPart:
		return true
	default:
		return false
	}
}

// metadata returns the encoded message metadata for a part, or nil.
func (c *partConversion) metadata(ctx context.Context, part ai.StreamPart) (json.RawMessage, error) {
	if c.messageMetadata == nil {
		return nil, nil
	}

	value := c.messageMetadata(ctx, MessageMetadataEvent{
		Part: part,
	})
	if value == nil {
		return nil, nil
	}

	encoded, err := encodeValue(value)
	if err != nil {
		return nil, fmt.Errorf("encode message metadata: %w", err)
	}

	if isJSONNull(encoded) {
		return nil, nil
	}

	return encoded, nil
}

// declareOutcome records the outcome a part declares. A completed or
// aborted outcome is final, and a failure does not replace an earlier
// outcome; fatal failures replace everything.
func (c *partConversion) declareOutcome(part ai.StreamPart) {
	var declared Outcome

	switch typed := part.(type) {
	case ai.FinishPart:
		declared = Outcome{
			Status: OutcomeCompleted,
		}
	case ai.AbortPart:
		declared = Outcome{
			Status: OutcomeAborted,
		}
	case ai.ErrorPart:
		declared = Outcome{
			Status: OutcomeFailed,
			Error:  typed.Err,
		}
	default:
		return
	}

	current := c.outcome.Status
	isFinal := current == OutcomeCompleted || current == OutcomeAborted
	replacesFailure := current == OutcomeUnknown || declared.Status != OutcomeFailed

	if c.fatalFailure || isFinal || !replacesFailure {
		return
	}

	c.outcome = declared
}

func (c *partConversion) fail(err error) {
	c.fatalFailure = true
	c.outcome = Outcome{
		Status: OutcomeFailed,
		Error:  err,
	}
}

func (c *partConversion) currentOutcome() Outcome {
	return c.outcome
}

// isEnabled returns the value of an option that defaults to true.
func isEnabled(option *bool) bool {
	return optional.Or(option, true)
}

// errorFormatter returns onError, or a formatter that hides error details.
func errorFormatter(onError func(err error) string) func(err error) string {
	if onError != nil {
		return onError
	}

	return func(error) string {
		return defaultErrorText
	}
}
