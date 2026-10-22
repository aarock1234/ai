package websocket

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strings"
)

// acceptGUID is appended to the handshake key to compute the accept value.
const acceptGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

// maxHandshakeBodyBytes bounds how much of a rejected handshake is read.
const maxHandshakeBodyBytes = 64 << 10

// DialOptions configure Dial.
type DialOptions struct {
	// Client sends the handshake. Nil uses http.DefaultClient. Its Timeout
	// bounds the handshake only. Its transport must support protocol
	// upgrades, as http.Transport does.
	Client *http.Client

	// Header is sent with the handshake.
	Header http.Header

	// Protocols are the requested subprotocols, in order of preference.
	Protocols []string

	// ReadLimit bounds one received message in bytes. Zero selects 32 MiB.
	ReadLimit int64
}

// HandshakeError reports a server that rejected or mishandled the opening
// handshake. It wraps ErrBadHandshake.
type HandshakeError struct {
	Message string

	// StatusCode, Header and Body describe the server response.
	StatusCode int
	Header     http.Header
	Body       []byte
}

// Error returns the failure and the status code.
func (e *HandshakeError) Error() string {
	return fmt.Sprintf("%s: %s (status %d)", ErrBadHandshake, e.Message, e.StatusCode)
}

// Unwrap returns ErrBadHandshake.
func (e *HandshakeError) Unwrap() error {
	return ErrBadHandshake
}

// Dial opens a WebSocket connection to rawURL, which may use the ws, wss,
// http or https scheme. Ctx governs the handshake and the connection:
// cancelling it closes the connection, and pending reads and writes then
// fail with the context's cause.
func Dial(ctx context.Context, rawURL string, options DialOptions) (*Conn, error) {
	target, err := httpURL(rawURL)
	if err != nil {
		return nil, err
	}

	key, err := handshakeKey()
	if err != nil {
		return nil, err
	}

	response, err := sendHandshake(ctx, target, key, options)
	if err != nil {
		return nil, err
	}

	stream, err := acceptHandshake(response, key, options.Protocols)
	if err != nil {
		return nil, err
	}

	readLimit := options.ReadLimit
	if readLimit <= 0 {
		readLimit = defaultReadLimit
	}

	conn := &Conn{
		reader:         bufio.NewReader(stream),
		stream:         stream,
		protocol:       response.Header.Get("Sec-WebSocket-Protocol"),
		responseHeader: response.Header,
		readLimit:      readLimit,
	}

	conn.watchContext(ctx)

	return conn, nil
}

// httpURL converts a ws or wss URL to the http or https URL net/http sends.
func httpURL(rawURL string) (string, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "", fmt.Errorf("parse websocket url: %w", err)
	}

	switch parsed.Scheme {
	case "ws":
		parsed.Scheme = "http"
	case "wss":
		parsed.Scheme = "https"
	case "http", "https":
		// net/http sends these schemes unchanged.
	default:
		return "", fmt.Errorf("websocket url has unsupported scheme %q", parsed.Scheme)
	}

	return parsed.String(), nil
}

// handshakeKey returns a random Sec-WebSocket-Key value.
func handshakeKey() (string, error) {
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return "", fmt.Errorf("generate websocket key: %w", err)
	}

	return base64.StdEncoding.EncodeToString(nonce[:]), nil
}

// sendHandshake sends the upgrade request. A client timeout applies to the
// handshake only: net/http would otherwise wrap the upgraded body and end
// the connection when the timeout elapses.
func sendHandshake(ctx context.Context, target, key string, options DialOptions) (*http.Response, error) {
	client := options.Client
	if client == nil {
		client = http.DefaultClient
	}

	handshakeClient := *client
	handshakeClient.Timeout = 0

	if client.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, client.Timeout)
		defer cancel()
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return nil, fmt.Errorf("build websocket handshake: %w", err)
	}

	request.Header = options.Header.Clone()
	if request.Header == nil {
		request.Header = make(http.Header)
	}

	request.Header.Set("Upgrade", "websocket")
	request.Header.Set("Connection", "Upgrade")
	request.Header.Set("Sec-WebSocket-Key", key)
	request.Header.Set("Sec-WebSocket-Version", "13")

	if len(options.Protocols) > 0 {
		request.Header.Set("Sec-WebSocket-Protocol", strings.Join(options.Protocols, ", "))
	}

	response, err := handshakeClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("send websocket handshake: %w", err)
	}

	return response, nil
}

// acceptHandshake validates the server response and returns the upgraded
// connection. Rejected handshakes close the response body.
func acceptHandshake(response *http.Response, key string, protocols []string) (io.ReadWriteCloser, error) {
	if response.StatusCode != http.StatusSwitchingProtocols {
		return nil, rejectedHandshake(response)
	}

	stream, ok := response.Body.(io.ReadWriteCloser)
	if !ok {
		closeBody(response.Body)

		return nil, invalidHandshake(response, "the http transport does not support protocol upgrades")
	}

	if message := handshakeProblem(response.Header, key, protocols); message != "" {
		closeBody(stream)

		return nil, invalidHandshake(response, message)
	}

	return stream, nil
}

// handshakeProblem describes what is wrong with an upgrade response, or
// returns an empty string when it is valid.
func handshakeProblem(header http.Header, key string, protocols []string) string {
	if !strings.EqualFold(header.Get("Upgrade"), "websocket") {
		return "missing upgrade header"
	}

	if !hasToken(header.Get("Connection"), "upgrade") {
		return "missing connection upgrade header"
	}

	if header.Get("Sec-WebSocket-Accept") != acceptValue(key) {
		return "invalid accept value"
	}

	if header.Get("Sec-WebSocket-Extensions") != "" {
		return "unrequested extension"
	}

	protocol := header.Get("Sec-WebSocket-Protocol")
	if protocol != "" && !slices.Contains(protocols, protocol) {
		return fmt.Sprintf("unrequested subprotocol %q", protocol)
	}

	return ""
}

// acceptValue returns the Sec-WebSocket-Accept value expected for key.
func acceptValue(key string) string {
	digest := sha1.Sum([]byte(key + acceptGUID))

	return base64.StdEncoding.EncodeToString(digest[:])
}

// hasToken reports whether a comma-separated header value contains token,
// ignoring case.
func hasToken(value, token string) bool {
	for part := range strings.SplitSeq(value, ",") {
		if strings.EqualFold(strings.TrimSpace(part), token) {
			return true
		}
	}

	return false
}

// rejectedHandshake reads a non-upgrade response into a HandshakeError.
func rejectedHandshake(response *http.Response) error {
	defer closeBody(response.Body)

	body, _ := io.ReadAll(io.LimitReader(response.Body, maxHandshakeBodyBytes))

	return &HandshakeError{
		Message:    "server did not switch protocols",
		StatusCode: response.StatusCode,
		Header:     response.Header,
		Body:       body,
	}
}

func invalidHandshake(response *http.Response, message string) error {
	return &HandshakeError{
		Message:    message,
		StatusCode: response.StatusCode,
		Header:     response.Header,
	}
}

// closeBody closes a body that is abandoned. Its error is ignored: the
// handshake has already failed.
func closeBody(body io.Closer) {
	_ = body.Close()
}
