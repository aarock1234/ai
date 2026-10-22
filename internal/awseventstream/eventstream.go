// Package awseventstream decodes the binary AWS event stream framing,
// application/vnd.amazon.eventstream, that Amazon Bedrock uses for streaming
// responses.
//
// Each message is a prelude with the total and header lengths and their
// CRC32, the headers, the payload and a CRC32 of everything before it. The
// reader verifies both checksums.
package awseventstream

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

const (
	// preludeBytes holds the total length and the headers length.
	preludeBytes = 8

	// checksumBytes is the size of each CRC32.
	checksumBytes = 4

	// minMessageBytes is a message without headers or payload.
	minMessageBytes = preludeBytes + 2*checksumBytes

	// maxMessageBytes bounds the memory one message may claim.
	maxMessageBytes = 24 << 20
)

// Standard header names.
const (
	headerMessageType   = ":message-type"
	headerEventType     = ":event-type"
	headerExceptionType = ":exception-type"
)

var (
	// ErrIncompleteMessage reports a stream that ended inside a message.
	ErrIncompleteMessage = errors.New("incomplete event stream message")

	// ErrInvalidMessage reports a malformed message or a checksum mismatch.
	ErrInvalidMessage = errors.New("invalid event stream message")
)

// Message is one decoded event stream message.
type Message struct {
	// Headers holds the string-valued headers by name. Headers of other
	// types are validated and skipped.
	Headers map[string]string

	// Payload is the message body.
	Payload []byte
}

// MessageType returns the :message-type header, such as "event" or
// "exception".
func (m Message) MessageType() string {
	return m.Headers[headerMessageType]
}

// EventType returns the :event-type header of an event message.
func (m Message) EventType() string {
	return m.Headers[headerEventType]
}

// ExceptionType returns the :exception-type header of an exception message.
func (m Message) ExceptionType() string {
	return m.Headers[headerExceptionType]
}

// Reader reads messages from an event stream. It does not close the
// underlying reader.
type Reader struct {
	source io.Reader
}

// NewReader returns a reader of the messages in source.
func NewReader(source io.Reader) *Reader {
	return &Reader{
		source: source,
	}
}

// Next returns the next message. It returns io.EOF when the stream ends
// between messages, and an error wrapping ErrIncompleteMessage when it ends
// inside one.
func (r *Reader) Next() (Message, error) {
	var lengthBytes [4]byte

	read, err := io.ReadFull(r.source, lengthBytes[:])
	if errors.Is(err, io.EOF) {
		return Message{}, io.EOF
	}

	if errors.Is(err, io.ErrUnexpectedEOF) {
		return Message{}, incompleteMessage(read)
	}

	if err != nil {
		return Message{}, err
	}

	totalLength := binary.BigEndian.Uint32(lengthBytes[:])
	if totalLength < minMessageBytes || totalLength > maxMessageBytes {
		return Message{}, fmt.Errorf("%w: message length %d", ErrInvalidMessage, totalLength)
	}

	frame := make([]byte, totalLength)
	copy(frame, lengthBytes[:])

	read, err = io.ReadFull(r.source, frame[len(lengthBytes):])
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return Message{}, incompleteMessage(len(lengthBytes) + read)
	}

	if err != nil {
		return Message{}, err
	}

	return decodeMessage(frame)
}

func incompleteMessage(bufferedBytes int) error {
	return fmt.Errorf("%w: %d buffered bytes remain at end of stream", ErrIncompleteMessage, bufferedBytes)
}

// decodeMessage verifies the checksums of one complete frame and decodes its
// headers and payload.
func decodeMessage(frame []byte) (Message, error) {
	headersLength := binary.BigEndian.Uint32(frame[4:8])
	preludeChecksum := binary.BigEndian.Uint32(frame[8:12])

	if crc32.ChecksumIEEE(frame[:preludeBytes]) != preludeChecksum {
		return Message{}, fmt.Errorf("%w: prelude checksum mismatch", ErrInvalidMessage)
	}

	messageEnd := len(frame) - checksumBytes
	messageChecksum := binary.BigEndian.Uint32(frame[messageEnd:])

	if crc32.ChecksumIEEE(frame[:messageEnd]) != messageChecksum {
		return Message{}, fmt.Errorf("%w: message checksum mismatch", ErrInvalidMessage)
	}

	headersStart := preludeBytes + checksumBytes
	if uint64(headersLength) > uint64(messageEnd-headersStart) {
		return Message{}, fmt.Errorf("%w: headers length %d exceeds message", ErrInvalidMessage, headersLength)
	}

	headersEnd := headersStart + int(headersLength)

	headers, err := decodeHeaders(frame[headersStart:headersEnd])
	if err != nil {
		return Message{}, err
	}

	return Message{
		Headers: headers,
		Payload: frame[headersEnd:messageEnd],
	}, nil
}
