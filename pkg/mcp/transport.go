package mcp

import (
	"context"
	"fmt"
	"net/http"
	"sync"
)

// Transport carries JSON-RPC messages between the client and one server.
// The client calls Start once, then Send from any goroutine while one
// goroutine calls Receive, and finally Close.
type Transport interface {
	// Start connects to the server. ctx bounds the connection attempt only;
	// the connection lasts until Close.
	Start(ctx context.Context) error

	// Send delivers message to the server. Canceling ctx abandons the
	// delivery and any response stream opened for it.
	Send(ctx context.Context, message Message, options SendOptions) error

	// Receive returns the next message from the server. It returns an error
	// wrapping ErrTransportClosed once no further messages will arrive; any
	// other error reports a problem, such as an unparsable message, after
	// which receiving continues.
	Receive(ctx context.Context) (Message, error)

	// Close disconnects, releases the transport's resources and waits for
	// its goroutines. ctx bounds graceful shutdown work, such as terminating
	// an HTTP session.
	Close(ctx context.Context) error
}

// SendOptions are per-message transport settings.
type SendOptions struct {
	// Headers are request headers derived from tool arguments annotated with
	// x-mcp-header. Only HTTP transports that implement
	// ToolParameterHeaderTransport use them.
	Headers http.Header
}

// ProtocolVersionTransport is implemented by transports whose behavior
// depends on the negotiated protocol version, such as HTTP transports that
// send it in a header.
type ProtocolVersionTransport interface {
	SetProtocolVersion(version string)
}

// ProtocolDiscoveryTransport is implemented by transports that can probe the
// server with server/discover before falling back to initialize. Custom
// transports without it use initialize.
type ProtocolDiscoveryTransport interface {
	SupportsProtocolVersionDiscovery() bool
}

// ToolParameterHeaderTransport is implemented by transports that mirror tool
// arguments annotated with x-mcp-header into request headers.
type ToolParameterHeaderTransport interface {
	SupportsToolParameterHeaders() bool
}

// inbox queues received messages and reception errors for Receive. Producers
// stop delivering when their context ends; close marks the end of the
// stream.
type inbox struct {
	items chan inboxItem

	closeOnce sync.Once
	done      chan struct{}
	cause     error
}

type inboxItem struct {
	message Message
	err     error
}

const inboxCapacity = 64

func newInbox() *inbox {
	return &inbox{
		items: make(chan inboxItem, inboxCapacity),
		done:  make(chan struct{}),
	}
}

// deliverMessage queues message. It reports false when ctx ended or the
// inbox closed first.
func (b *inbox) deliverMessage(ctx context.Context, message Message) bool {
	return b.deliver(ctx, inboxItem{
		message: message,
	})
}

// deliverError queues a non-fatal reception error.
func (b *inbox) deliverError(ctx context.Context, err error) bool {
	return b.deliver(ctx, inboxItem{
		err: err,
	})
}

func (b *inbox) deliver(ctx context.Context, item inboxItem) bool {
	select {
	case b.items <- item:
		return true
	case <-b.done:
		return false
	case <-ctx.Done():
		return false
	}
}

// close ends the stream. Receive drains queued items first and then returns
// ErrTransportClosed, wrapping cause when it is not nil.
func (b *inbox) close(cause error) {
	b.closeOnce.Do(func() {
		b.cause = cause
		close(b.done)
	})
}

func (b *inbox) receive(ctx context.Context) (Message, error) {
	select {
	case item := <-b.items:
		return item.message, item.err
	default:
	}

	select {
	case item := <-b.items:
		return item.message, item.err
	case <-b.done:
		return Message{}, b.closedError()
	case <-ctx.Done():
		return Message{}, ctx.Err()
	}
}

func (b *inbox) closedError() error {
	if b.cause == nil {
		return ErrTransportClosed
	}

	return fmt.Errorf("%w: %w", ErrTransportClosed, b.cause)
}
