package bridge

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/harness"
)

// tokenQueryParameter carries the bridge token on the connection URL.
const tokenQueryParameter = "agent_bridge_token"

// Conn is a connection to a bridge carrying JSON text messages. Read is
// called from one goroutine at a time; Write and Close are safe from any
// goroutine.
type Conn interface {
	// Read returns the next message.
	Read() ([]byte, error)

	// Write sends one message.
	Write(data []byte) error

	// Close closes the connection. It is safe to call more than once.
	Close() error
}

// ConnectionSettings are the bridge connection settings every bridge-backed
// harness adapter exposes. Adapters embed them in their settings.
type ConnectionSettings struct {
	// Port overrides the port the bridge binds inside the sandbox. Zero uses
	// the first port the sandbox exposes.
	Port int

	// PortEndpoint overrides the endpoint used to reach the bridge. It is
	// required, together with Port, for a basic sandbox session.
	PortEndpoint *harness.PortEndpoint

	// StartupTimeout bounds the wait for the bridge to become ready. Zero
	// selects 120 seconds.
	StartupTimeout time.Duration

	// Reconnect bounds the reconnects after an established bridge connection
	// drops. Zero values select 30 seconds with exponential backoff from 50
	// milliseconds to 2 seconds.
	Reconnect ReconnectOptions

	// MintBridgeToken creates the token securing the bridge connection. Nil
	// uses NewToken, a random 32-byte hexadecimal token.
	MintBridgeToken harness.MintBridgeTokenFunc

	// HTTPClient sends the bridge's WebSocket handshakes and, for adapters
	// that read a native CLI login, its OAuth requests. Nil uses
	// http.DefaultClient.
	HTTPClient *http.Client
}

// DialOptions configure Dial.
type DialOptions struct {
	// Client sends the WebSocket handshake. Nil uses http.DefaultClient.
	Client *http.Client
}

// Dial opens a WebSocket connection to a bridge endpoint, sending the
// endpoint's headers. Ctx governs the handshake and the connection:
// cancelling it closes the connection.
func Dial(ctx context.Context, endpoint harness.PortEndpoint, options DialOptions) (Conn, error) {
	header := make(http.Header, len(endpoint.Headers))
	for name, value := range endpoint.Headers {
		header.Set(name, value)
	}

	conn, err := websocket.Dial(ctx, endpoint.URL, websocket.DialOptions{
		Client: options.Client,
		Header: header,
	})
	if err != nil {
		return nil, fmt.Errorf("connect to bridge: %w", err)
	}

	return &websocketConn{
		conn: conn,
	}, nil
}

// websocketConn adapts a WebSocket connection to Conn.
type websocketConn struct {
	conn *websocket.Conn
}

// Read returns the next data message.
func (c *websocketConn) Read() ([]byte, error) {
	_, data, err := c.conn.ReadMessage()

	return data, err
}

// Write sends a text message.
func (c *websocketConn) Write(data []byte) error {
	return c.conn.WriteMessage(websocket.TextMessage, data)
}

// Close sends a normal closure and closes the connection.
func (c *websocketConn) Close() error {
	return c.conn.Close(websocket.StatusNormalClosure, "")
}

// NewToken returns a random 32-byte hexadecimal bridge token, the default
// harness.MintBridgeTokenFunc.
func NewToken() string {
	random := make([]byte, 32)
	_, _ = rand.Read(random)

	return hex.EncodeToString(random)
}

// WithToken returns endpoint with token added to its URL query.
func WithToken(endpoint harness.PortEndpoint, token string) (harness.PortEndpoint, error) {
	parsed, err := url.Parse(endpoint.URL)
	if err != nil {
		return harness.PortEndpoint{}, fmt.Errorf("parse bridge endpoint: %w", err)
	}

	query := parsed.Query()
	query.Set(tokenQueryParameter, token)
	parsed.RawQuery = query.Encode()

	endpoint.URL = parsed.String()

	return endpoint, nil
}
