// Package websocket is a minimal RFC 6455 WebSocket client built on net/http.
// It owns the transport: the opening handshake, framing and masking, ping
// replies, the closing handshake and context cancellation. Callers own the
// URL, authentication and the message protocol.
//
// Messages are read from one goroutine at a time; writes and Close are safe
// from any goroutine. Extensions such as compression are not negotiated.
package websocket

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
)

// defaultReadLimit bounds one received message when DialOptions.ReadLimit
// is zero.
const defaultReadLimit = 32 << 20

// Sentinel errors. Errors returned by this package wrap them, so test with
// errors.Is.
var (
	// ErrBadHandshake reports a failed opening handshake.
	ErrBadHandshake = errors.New("websocket handshake failed")

	// ErrProtocol reports a peer that violated the WebSocket protocol.
	ErrProtocol = errors.New("websocket protocol error")

	// ErrMessageTooBig reports a received message over the read limit.
	ErrMessageTooBig = errors.New("websocket message too big")

	// ErrClosed reports use of a connection after Close.
	ErrClosed = errors.New("websocket connection closed")
)

// MessageType is the type of a data message.
type MessageType int

// Message types, numbered as their frame opcodes.
const (
	TextMessage   MessageType = 1
	BinaryMessage MessageType = 2
)

// StatusCode is a close status code.
type StatusCode int

// Close status codes defined by RFC 6455.
const (
	StatusNormalClosure   StatusCode = 1000
	StatusGoingAway       StatusCode = 1001
	StatusProtocolError   StatusCode = 1002
	StatusUnsupportedData StatusCode = 1003

	// StatusNoStatus is never sent on the wire: it reports a close frame
	// without a status code, and Close sends such a frame when given it.
	StatusNoStatus StatusCode = 1005

	StatusInvalidPayload  StatusCode = 1007
	StatusPolicyViolation StatusCode = 1008
	StatusMessageTooBig   StatusCode = 1009
	StatusInternalError   StatusCode = 1011
)

// CloseError reports that the peer closed the connection.
type CloseError struct {
	Code   StatusCode
	Reason string
}

// Error returns the close code and reason.
func (e *CloseError) Error() string {
	if e.Reason == "" {
		return fmt.Sprintf("websocket closed with status %d", e.Code)
	}

	return fmt.Sprintf("websocket closed with status %d: %s", e.Code, e.Reason)
}

// Conn is an open WebSocket connection.
type Conn struct {
	reader *bufio.Reader
	stream io.ReadWriteCloser

	protocol       string
	responseHeader http.Header
	readLimit      int64

	// writeMu serializes frames and guards closeSent.
	writeMu   sync.Mutex
	closeSent bool

	// stateMu guards closed and closedErr.
	stateMu   sync.Mutex
	closed    bool
	closedErr error

	stopWatchingContext func() bool
}

// Subprotocol returns the subprotocol the server selected, if any.
func (c *Conn) Subprotocol() string {
	return c.protocol
}

// ResponseHeader returns the headers of the handshake response.
func (c *Conn) ResponseHeader() http.Header {
	return c.responseHeader
}

// WriteMessage sends data as one unfragmented message.
func (c *Conn) WriteMessage(messageType MessageType, data []byte) error {
	if messageType != TextMessage && messageType != BinaryMessage {
		return fmt.Errorf("write websocket message: unknown message type %d", messageType)
	}

	return c.writeFrame(opcode(messageType), data)
}

// Close sends a close frame with code and reason and closes the connection.
// StatusNoStatus sends a close frame without a status. The frame is skipped
// when one was already sent or another write is in progress, and Close does
// not wait for the peer's close frame. Close is safe to call more than once
// and from any goroutine.
func (c *Conn) Close(code StatusCode, reason string) error {
	return c.closeWith(code, reason, ErrClosed)
}

// closeTransport closes the underlying connection once. Later reads and
// writes fail with cause.
func (c *Conn) closeTransport(cause error) error {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()

	if c.closed {
		return nil
	}

	c.closed = true
	c.closedErr = cause
	c.stopWatchingContext()

	if err := c.stream.Close(); err != nil {
		return fmt.Errorf("close websocket connection: %w", err)
	}

	return nil
}

// closedCause returns the cause recorded when the connection was closed
// locally, or nil while it is open.
func (c *Conn) closedCause() error {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()

	return c.closedErr
}

// watchContext closes the connection when ctx ends. The lock keeps an
// already ended ctx from closing the connection before the stop function
// is stored.
func (c *Conn) watchContext(ctx context.Context) {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()

	c.stopWatchingContext = context.AfterFunc(ctx, func() {
		_ = c.closeTransport(context.Cause(ctx))
	})
}
