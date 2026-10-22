package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sync"

	"github.com/aarock1234/ai/internal/origin"
	"github.com/aarock1234/ai/internal/sse"
)

// SSETransportOptions configure NewSSETransport.
type SSETransportOptions struct {
	// URL is the Server-Sent Events endpoint. It is required.
	URL string

	// Headers are sent with every request.
	Headers http.Header

	// HTTPClient sends requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// AuthProvider authorizes requests with OAuth, as for HTTPTransport.
	AuthProvider OAuthClientProvider

	// FollowRedirects follows HTTP redirects. By default a redirect fails
	// the request.
	FollowRedirects bool
}

// SSETransport implements the legacy HTTP with Server-Sent Events
// transport: a GET stream announces a POST endpoint with an endpoint event
// and carries every message from the server.
type SSETransport struct {
	url        *url.URL
	settings   httpSettings
	client     *http.Client
	authorizer *authorizer

	inbox  *inbox
	reader sync.WaitGroup

	mu              sync.Mutex
	started         bool
	closed          bool
	connected       bool
	endpoint        *url.URL
	protocolVersion string
	cancelStream    context.CancelFunc
}

// NewSSETransport returns a legacy SSE transport for options.URL.
func NewSSETransport(options SSETransportOptions) (*SSETransport, error) {
	endpoint, err := parseAbsoluteURL(options.URL)
	if err != nil {
		return nil, &ClientError{
			Message: "invalid mcp sse transport url",
			Cause:   err,
		}
	}

	client := newHTTPClient(options.HTTPClient, options.FollowRedirects)

	transport := &SSETransport{
		url: endpoint,
		settings: httpSettings{
			headers:      options.Headers,
			client:       client,
			authProvider: options.AuthProvider,
		},
		client:          client,
		inbox:           newInbox(),
		protocolVersion: LatestLegacyProtocolVersion,
	}

	if options.AuthProvider != nil {
		transport.authorizer = &authorizer{
			provider:  options.AuthProvider,
			serverURL: endpoint.String(),
			client:    options.HTTPClient,
		}
	}

	return transport, nil
}

// SetProtocolVersion records the negotiated protocol version, which is sent
// in the Mcp-Protocol-Version header.
func (t *SSETransport) SetProtocolVersion(version string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.protocolVersion = version
}

// Start opens the event stream and waits for the endpoint event. ctx bounds
// the wait; the stream lasts until Close.
func (t *SSETransport) Start(ctx context.Context) error {
	t.mu.Lock()
	if t.started {
		t.mu.Unlock()

		return &ClientError{
			Message: "mcp sse transport already started",
		}
	}

	t.started = true
	streamCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	t.cancelStream = cancel
	t.mu.Unlock()

	// Abandon the connection if ctx ends before the endpoint arrives.
	stopWatching := context.AfterFunc(ctx, cancel)
	defer stopWatching()

	response, err := t.openStream(streamCtx)
	if err != nil {
		cancel()

		return t.startFailure(ctx, err)
	}

	ready := make(chan error, 1)

	t.reader.Go(func() {
		defer closeBody(response.Body)

		t.readStream(streamCtx, response.Body, ready)
	})

	if err := <-ready; err != nil {
		cancel()

		return t.startFailure(ctx, err)
	}

	if !stopWatching() {
		return ctx.Err()
	}

	return nil
}

func (t *SSETransport) startFailure(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}

	return err
}

// openStream sends the GET request, running the OAuth flow once after a
// 401 response, and checks the response.
func (t *SSETransport) openStream(ctx context.Context) (*http.Response, error) {
	response, err := t.getStream(ctx)
	if err != nil {
		return nil, err
	}

	if response.StatusCode == http.StatusUnauthorized && t.authorizer != nil {
		closeBody(response.Body)

		if err := t.authorizer.recoverUnauthorized(ctx, response); err != nil {
			return nil, err
		}

		response, err = t.getStream(ctx)
		if err != nil {
			return nil, err
		}
	}

	if response.StatusCode >= 200 && response.StatusCode <= 299 {
		return response, nil
	}

	defer closeBody(response.Body)

	return nil, t.streamFailure(response)
}

func (t *SSETransport) getStream(ctx context.Context) (*http.Response, error) {
	base := http.Header{
		"Accept": {
			"text/event-stream",
		},
	}

	headers, err := requestHeaders(ctx, t.settings, base, t.currentProtocolVersion())
	if err != nil {
		return nil, err
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, t.url.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("create mcp sse request: %w", err)
	}

	request.Header = headers

	response, err := t.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("mcp sse transport error: get %s: %w", t.url, err)
	}

	return response, nil
}

func (t *SSETransport) streamFailure(response *http.Response) error {
	body, _ := readErrorBody(response)

	message := fmt.Sprintf("mcp sse transport error: %d %s", response.StatusCode, http.StatusText(response.StatusCode))
	if body != "" {
		message += ": " + body
	}

	if response.StatusCode == http.StatusMethodNotAllowed {
		message += ". this server does not support sse transport. try using the http transport instead"
	}

	return &ClientError{
		Message:      message,
		StatusCode:   response.StatusCode,
		URL:          t.url.String(),
		ResponseBody: body,
	}
}

// readStream reads the event stream. It reports the outcome of connecting
// on ready, and closes the inbox when the stream ends.
func (t *SSETransport) readStream(ctx context.Context, body io.Reader, ready chan<- error) {
	connected := false

	for event, err := range sse.Read(body) {
		if err != nil {
			t.endStream(ctx, connected, ready, fmt.Errorf("mcp sse transport error: read event stream: %w", err))

			return
		}

		if event.Type == "endpoint" {
			if connected {
				continue
			}

			if err := t.connect(event.Data); err != nil {
				t.endStream(ctx, false, ready, err)

				return
			}

			connected = true
			ready <- nil

			continue
		}

		if isMessageEvent(event.Type) && !deliverEventMessage(ctx, t.inbox, event.Data, "mcp sse transport error: failed to parse message") {
			t.endStream(ctx, connected, ready, nil)

			return
		}
	}

	t.endStream(ctx, connected, ready, nil)
}

// connect records the POST endpoint, which must have the stream's origin.
func (t *SSETransport) connect(data string) error {
	endpoint, err := t.url.Parse(data)
	if err != nil {
		return fmt.Errorf("mcp sse transport error: invalid endpoint %q: %w", data, err)
	}

	if !origin.SameURL(endpoint, t.url) {
		return &ClientError{
			Message: "mcp sse transport error: endpoint origin does not match connection origin: " + origin.Of(endpoint),
		}
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	t.endpoint = endpoint
	t.connected = true

	return nil
}

// endStream marks the transport disconnected. Before the endpoint event the
// failure is reported to Start; afterwards it closes the inbox.
func (t *SSETransport) endStream(ctx context.Context, connected bool, ready chan<- error, cause error) {
	t.mu.Lock()
	t.connected = false
	t.mu.Unlock()

	if !connected {
		if cause == nil {
			cause = errors.New("mcp sse transport error: connection closed before endpoint event was received")
		}

		ready <- cause

		return
	}

	if ctx.Err() == nil && cause == nil {
		cause = errors.New("mcp sse transport error: connection closed unexpectedly")
	}

	t.inbox.close(cause)
}

// Receive returns the next message from the server.
func (t *SSETransport) Receive(ctx context.Context) (Message, error) {
	return t.inbox.receive(ctx)
}

// Close ends the event stream and waits for its reader.
func (t *SSETransport) Close(_ context.Context) error {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()

		return nil
	}

	t.closed = true
	t.connected = false
	t.endpoint = nil
	cancel := t.cancelStream
	t.mu.Unlock()

	if cancel != nil {
		cancel()
	}

	t.reader.Wait()
	t.inbox.close(nil)

	return nil
}

// Send POSTs message to the endpoint announced by the server, running the
// OAuth flow once after a 401 response.
func (t *SSETransport) Send(ctx context.Context, message Message, _ SendOptions) error {
	endpoint, err := t.currentEndpoint()
	if err != nil {
		return err
	}

	body, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("encode json-rpc message: %w", err)
	}

	response, authorization, err := t.postMessage(ctx, endpoint, body)
	if err != nil {
		return err
	}

	if response.StatusCode == http.StatusUnauthorized && t.authorizer != nil {
		closeBody(response.Body)

		if err := t.recoverPost(ctx, response, authorization); err != nil {
			return err
		}

		response, _, err = t.postMessage(ctx, endpoint, body)
		if err != nil {
			return err
		}
	}
	defer closeBody(response.Body)

	if response.StatusCode >= 200 && response.StatusCode <= 299 {
		return nil
	}

	text, _ := readErrorBody(response)

	return &ClientError{
		Message:      fmt.Sprintf("mcp sse transport error: posting to endpoint (http %d): %s", response.StatusCode, text),
		StatusCode:   response.StatusCode,
		URL:          endpoint.String(),
		ResponseBody: text,
	}
}

// recoverPost retries with a token that changed since the request was sent,
// and otherwise runs the OAuth flow.
func (t *SSETransport) recoverPost(ctx context.Context, response *http.Response, sentAuthorization string) error {
	tokens, err := t.authorizer.provider.Tokens(ctx)
	if err != nil {
		return fmt.Errorf("load oauth tokens: %w", err)
	}

	if tokens != nil && tokens.AccessToken != "" && sentAuthorization != "Bearer "+tokens.AccessToken {
		return nil
	}

	return t.authorizer.recoverUnauthorized(ctx, response)
}

func (t *SSETransport) currentEndpoint() (*url.URL, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.endpoint == nil || !t.connected {
		return nil, &ClientError{
			Message: "mcp sse transport error: not connected",
			Cause:   ErrConnectionClosed,
		}
	}

	return t.endpoint, nil
}

func (t *SSETransport) currentProtocolVersion() string {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.protocolVersion
}

// postMessage sends body and returns the response with the Authorization
// header it carried.
func (t *SSETransport) postMessage(ctx context.Context, endpoint *url.URL, body []byte) (*http.Response, string, error) {
	base := http.Header{
		"Content-Type": {
			"application/json",
		},
	}

	headers, err := requestHeaders(ctx, t.settings, base, t.currentProtocolVersion())
	if err != nil {
		return nil, "", err
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewReader(body))
	if err != nil {
		return nil, "", fmt.Errorf("create mcp sse request: %w", err)
	}

	request.Header = headers

	response, err := t.client.Do(request)
	if err != nil {
		return nil, "", fmt.Errorf("mcp sse transport error: post to %s: %w", endpoint, err)
	}

	return response, headers.Get("Authorization"), nil
}

var _ interface {
	Transport
	ProtocolVersionTransport
} = (*SSETransport)(nil)
