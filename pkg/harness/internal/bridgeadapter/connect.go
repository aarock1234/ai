package bridgeadapter

import (
	"context"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// Handshake bounds of connections that wait for the bridge's hello, as
// upstream.
const (
	maxOpenTimeout     = 10 * time.Second
	maxHelloTimeout    = 5 * time.Second
	retryDelayStep     = 250 * time.Millisecond
	maxRetryDelay      = time.Second
	minimumHelloBudget = time.Millisecond
)

// connector opens connections to one bridge endpoint.
type connector struct {
	identity adapterIdentity
	endpoint harness.PortEndpoint
	client   *http.Client

	// waitForHello makes each connection wait for the bridge-hello frame
	// before it carries commands, retrying within handshakeBudget. Some
	// sandboxes complete the WebSocket handshake long before the connection
	// reaches the bridge, and anything sent in that gap is lost.
	waitForHello    bool
	handshakeBudget time.Duration

	// userMessageResponses records the capability the latest hello
	// announced.
	userMessageResponses *atomic.Bool
}

// connect opens a connection, the bridge.ChannelOptions.Connect function.
func (c connector) connect(ctx context.Context) (bridge.Conn, error) {
	if !c.waitForHello {
		return c.dial(ctx)
	}

	return c.connectWithHello(ctx)
}

// dial opens a WebSocket connection governed by ctx.
func (c connector) dial(ctx context.Context) (bridge.Conn, error) {
	return bridge.Dial(ctx, c.endpoint, bridge.DialOptions{
		Client: c.client,
	})
}

// connectWithHello retries opening a connection that received the bridge's
// hello until the handshake budget runs out.
func (c connector) connectWithHello(ctx context.Context) (bridge.Conn, error) {
	deadline := time.Now().Add(c.handshakeBudget)

	var lastErr error
	attempt := 0

	for time.Now().Before(deadline) {
		attempt++

		conn, err := c.openAndAwaitHello(ctx, deadline)
		if err == nil {
			return conn, nil
		}

		if ctx.Err() != nil {
			return nil, context.Cause(ctx)
		}

		lastErr = err

		remaining := time.Until(deadline)
		if remaining <= 0 {
			break
		}

		if err := sleep(ctx, min(retryDelayStep*time.Duration(attempt), maxRetryDelay, remaining)); err != nil {
			return nil, err
		}
	}

	return nil, fmt.Errorf(
		"%s bridge did not complete the websocket handshake within %s after %d attempt(s): %w",
		c.identity.harnessID,
		c.handshakeBudget,
		attempt,
		lastErr,
	)
}

// openAndAwaitHello opens one connection and waits for the bridge-hello
// frame on it.
func (c connector) openAndAwaitHello(ctx context.Context, deadline time.Time) (bridge.Conn, error) {
	conn, err := c.dialWithin(ctx, min(maxOpenTimeout, time.Until(deadline)))
	if err != nil {
		return nil, err
	}

	helloTimeout := min(maxHelloTimeout, max(minimumHelloBudget, time.Until(deadline)))

	if err := c.awaitHello(ctx, conn, helloTimeout); err != nil {
		_ = conn.Close()

		return nil, err
	}

	return conn, nil
}

// dialWithin dials with the open timeout. The connection outlives the
// timeout: it lives until ctx ends or the connection is closed.
func (c connector) dialWithin(ctx context.Context, timeout time.Duration) (bridge.Conn, error) {
	connectionCtx, cancelConnection := context.WithCancel(ctx)

	type dialResult struct {
		conn bridge.Conn
		err  error
	}

	results := make(chan dialResult, 1)

	go func() {
		conn, err := c.dial(connectionCtx)
		results <- dialResult{
			conn: conn,
			err:  err,
		}
	}()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case result := <-results:
		if result.err != nil {
			cancelConnection()

			return nil, result.err
		}

		return &cancelOnClose{
			Conn:   result.conn,
			cancel: cancelConnection,
		}, nil
	case <-timer.C:
		cancelConnection()
		<-results

		return nil, fmt.Errorf("websocket open timed out after %s", timeout)
	}
}

// awaitHello reads frames until the bridge-hello frame arrives, recording
// its capabilities.
func (c connector) awaitHello(ctx context.Context, conn bridge.Conn, timeout time.Duration) error {
	hellos := make(chan error, 1)
	readerDone := make(chan struct{})

	go func() {
		defer close(readerDone)

		hellos <- c.readUntilHello(conn)
	}()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case err := <-hellos:
		return err
	case <-timer.C:
		_ = conn.Close()
		<-readerDone

		return fmt.Errorf("%s bridge did not send bridge-hello within %s", c.identity.harnessID, timeout)
	case <-ctx.Done():
		_ = conn.Close()
		<-readerDone

		return context.Cause(ctx)
	}
}

// readUntilHello reads messages until a bridge-hello frame, ignoring every
// other message.
func (c connector) readUntilHello(conn bridge.Conn) error {
	for {
		data, err := conn.Read()
		if err != nil {
			return fmt.Errorf("%s bridge closed before sending bridge-hello: %w", c.identity.harnessID, err)
		}

		frame, err := bridge.DecodeFrame(data)
		if err != nil {
			continue
		}

		hello, ok := frame.Message.(bridge.Hello)
		if !ok {
			continue
		}

		c.userMessageResponses.Store(hello.UserMessageResponses)

		return nil
	}
}

// cancelOnClose releases a connection's context when it is closed.
type cancelOnClose struct {
	bridge.Conn

	cancel context.CancelFunc
}

// Close closes the connection and releases its context.
func (c *cancelOnClose) Close() error {
	err := c.Conn.Close()
	c.cancel()

	return err
}

// sleep waits for d or until ctx ends.
func sleep(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}
