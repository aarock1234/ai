package websocket

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"unicode/utf8"
)

// opcode identifies the kind of a frame.
type opcode byte

// Frame opcodes.
const (
	opContinuation opcode = 0x0
	opText         opcode = 0x1
	opBinary       opcode = 0x2
	opClose        opcode = 0x8
	opPing         opcode = 0x9
	opPong         opcode = 0xA
)

// Frame header bits and lengths.
const (
	finalBit         = 0x80
	reservedBits     = 0x70
	opcodeBits       = 0x0F
	maskBit          = 0x80
	lengthBits       = 0x7F
	length16         = 126
	length64         = 127
	maxControlLength = 125
	maxHeaderLength  = 14
)

// frame is one received frame.
type frame struct {
	final   bool
	opcode  opcode
	payload []byte
}

// isControl reports whether op is a close, ping or pong opcode.
func (op opcode) isControl() bool {
	return op >= opClose
}

// ReadMessage returns the next data message, assembling fragments and
// answering pings on the way. It returns a *CloseError when the peer closes
// the connection, after replying to its close frame. It must not be called
// concurrently with itself.
func (c *Conn) ReadMessage() (MessageType, []byte, error) {
	var (
		messageType MessageType
		message     []byte
		inMessage   bool
	)

	for {
		received, err := c.readFrame()
		if err != nil {
			return 0, nil, c.readFailure(err)
		}

		switch received.opcode {
		case opPing:
			if err := c.writeFrame(opPong, received.payload); err != nil {
				return 0, nil, err
			}

			continue

		case opPong:
			continue

		case opClose:
			return 0, nil, c.peerClosed(received.payload)

		case opText, opBinary:
			if inMessage {
				return 0, nil, c.fail(StatusProtocolError, "new message before the previous one ended")
			}

			inMessage = true
			messageType = MessageType(received.opcode)
			message = received.payload

		case opContinuation:
			if !inMessage {
				return 0, nil, c.fail(StatusProtocolError, "continuation frame without a message")
			}

			if int64(len(message)+len(received.payload)) > c.readLimit {
				return 0, nil, c.tooBig()
			}

			message = append(message, received.payload...)

		default:
			return 0, nil, c.fail(StatusProtocolError, fmt.Sprintf("reserved opcode %d", received.opcode))
		}

		if !received.final {
			continue
		}

		if messageType == TextMessage && !utf8.Valid(message) {
			return 0, nil, c.fail(StatusInvalidPayload, "text message is not valid utf-8")
		}

		return messageType, message, nil
	}
}

// readFrame reads one frame. Servers must not mask frames, and control
// frames must be final and short.
func (c *Conn) readFrame() (frame, error) {
	var head [2]byte
	if _, err := io.ReadFull(c.reader, head[:]); err != nil {
		return frame{}, err
	}

	if head[0]&reservedBits != 0 {
		return frame{}, c.fail(StatusProtocolError, "reserved bits set without an extension")
	}

	if head[1]&maskBit != 0 {
		return frame{}, c.fail(StatusProtocolError, "server frame is masked")
	}

	received := frame{
		final:  head[0]&finalBit != 0,
		opcode: opcode(head[0] & opcodeBits),
	}

	length, err := c.readLength(head[1] & lengthBits)
	if err != nil {
		return frame{}, err
	}

	if received.opcode.isControl() && (!received.final || length > maxControlLength) {
		return frame{}, c.fail(StatusProtocolError, "fragmented or oversized control frame")
	}

	if length > uint64(c.readLimit) {
		return frame{}, c.tooBig()
	}

	received.payload = make([]byte, length)
	if _, err := io.ReadFull(c.reader, received.payload); err != nil {
		return frame{}, err
	}

	return received, nil
}

// readLength reads the extended payload length that the 7-bit length
// selects.
func (c *Conn) readLength(short byte) (uint64, error) {
	switch short {
	case length16:
		var extended [2]byte
		if _, err := io.ReadFull(c.reader, extended[:]); err != nil {
			return 0, err
		}

		return uint64(binary.BigEndian.Uint16(extended[:])), nil

	case length64:
		var extended [8]byte
		if _, err := io.ReadFull(c.reader, extended[:]); err != nil {
			return 0, err
		}

		length := binary.BigEndian.Uint64(extended[:])
		if length>>63 != 0 {
			return 0, c.fail(StatusProtocolError, "payload length has the most significant bit set")
		}

		return length, nil

	default:
		return uint64(short), nil
	}
}

// readFailure reports why a read failed: the protocol failure, the local
// close or context cause when the connection was closed here, or the
// transport error.
func (c *Conn) readFailure(err error) error {
	if errors.Is(err, ErrProtocol) || errors.Is(err, ErrMessageTooBig) {
		return err
	}

	if cause := c.closedCause(); cause != nil {
		return cause
	}

	if errors.Is(err, io.EOF) {
		err = io.ErrUnexpectedEOF
	}

	failure := fmt.Errorf("read websocket frame: %w", err)

	_ = c.closeTransport(failure)

	return failure
}

// peerClosed answers the peer's close frame and closes the connection.
func (c *Conn) peerClosed(payload []byte) error {
	closeErr := &CloseError{
		Code: StatusNoStatus,
	}

	switch {
	case len(payload) == 1:
		return c.fail(StatusProtocolError, "close frame payload of one byte")

	case len(payload) >= 2:
		closeErr.Code = StatusCode(binary.BigEndian.Uint16(payload))
		closeErr.Reason = string(payload[2:])

		if !isReceivableCloseCode(closeErr.Code) {
			return c.fail(StatusProtocolError, fmt.Sprintf("invalid close status %d", closeErr.Code))
		}

		if !utf8.ValidString(closeErr.Reason) {
			return c.fail(StatusInvalidPayload, "close reason is not valid utf-8")
		}
	}

	_ = c.closeWith(closeErr.Code, "", closeErr)

	return closeErr
}

// isReceivableCloseCode reports whether a peer may send code in a close
// frame. RFC 6455 section 7.4 reserves 1004, 1005 and 1006 and every code
// below 1000 or between 1016 and 2999; 1015 is reserved for TLS failures,
// and 5000 and above are outside the defined ranges. Codes 1012 to 1014
// are registered with IANA after the RFC.
func isReceivableCloseCode(code StatusCode) bool {
	switch {
	case code >= 1000 && code <= 1003:
		return true
	case code >= 1007 && code <= 1014:
		return true
	default:
		return code >= 3000 && code <= 4999
	}
}

// fail closes the connection with code after a protocol violation.
func (c *Conn) fail(code StatusCode, message string) error {
	err := fmt.Errorf("%w: %s", ErrProtocol, message)

	_ = c.closeWith(code, "", err)

	return err
}

// tooBig closes the connection after a message over the read limit.
func (c *Conn) tooBig() error {
	err := fmt.Errorf("%w: over %d bytes", ErrMessageTooBig, c.readLimit)

	_ = c.closeWith(StatusMessageTooBig, "", err)

	return err
}

// closeWith sends a close frame when no other write is in progress, then
// closes the connection; later reads and writes fail with cause. The frame
// is best effort: the connection is torn down either way, and waiting for
// a write blocked on an unresponsive peer would never end.
func (c *Conn) closeWith(code StatusCode, reason string, cause error) error {
	if c.writeMu.TryLock() {
		_ = c.writeFrameLocked(opClose, closePayload(code, reason))
		c.writeMu.Unlock()
	}

	return c.closeTransport(cause)
}

// closePayload encodes a close status and reason. StatusNoStatus encodes
// an empty payload. A long reason is cut at a character boundary so that
// it stays valid UTF-8.
func closePayload(code StatusCode, reason string) []byte {
	if code == StatusNoStatus {
		return nil
	}

	payload := binary.BigEndian.AppendUint16(nil, uint16(code))

	return append(payload, truncateReason(reason)...)
}

// truncateReason shortens reason to the longest prefix of whole characters
// that fits a control frame after the status code.
func truncateReason(reason string) string {
	const maxReasonLength = maxControlLength - 2

	if len(reason) <= maxReasonLength {
		return reason
	}

	end := maxReasonLength
	for end > 0 && !utf8.RuneStart(reason[end]) {
		end--
	}

	return reason[:end]
}

// writeFrame sends one final, masked frame.
func (c *Conn) writeFrame(op opcode, payload []byte) error {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()

	return c.writeFrameLocked(op, payload)
}

// writeFrameLocked sends one frame while writeMu is held. Nothing is sent
// after a close frame.
func (c *Conn) writeFrameLocked(op opcode, payload []byte) error {
	if cause := c.closedCause(); cause != nil {
		return cause
	}

	if c.closeSent {
		return ErrClosed
	}

	encoded, err := encodeFrame(op, payload)
	if err != nil {
		return err
	}

	if op == opClose {
		c.closeSent = true
	}

	if _, err := c.stream.Write(encoded); err != nil {
		if cause := c.closedCause(); cause != nil {
			return cause
		}

		return fmt.Errorf("write websocket frame: %w", err)
	}

	return nil
}

// encodeFrame returns the header and masked payload of a final frame.
// Clients must mask every frame with a fresh random key.
func encodeFrame(op opcode, payload []byte) ([]byte, error) {
	encoded := make([]byte, 0, maxHeaderLength+len(payload))
	encoded = append(encoded, finalBit|byte(op))

	length := len(payload)

	switch {
	case length <= maxControlLength:
		encoded = append(encoded, maskBit|byte(length))
	case length <= 0xFFFF:
		encoded = append(encoded, maskBit|length16)
		encoded = binary.BigEndian.AppendUint16(encoded, uint16(length))
	default:
		encoded = append(encoded, maskBit|length64)
		encoded = binary.BigEndian.AppendUint64(encoded, uint64(length))
	}

	var key [4]byte
	if _, err := rand.Read(key[:]); err != nil {
		return nil, fmt.Errorf("generate websocket mask: %w", err)
	}

	encoded = append(encoded, key[:]...)

	for index, value := range payload {
		encoded = append(encoded, value^key[index%4])
	}

	return encoded, nil
}
