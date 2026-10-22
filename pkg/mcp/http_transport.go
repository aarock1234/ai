package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/internal/sse"
)

// maxMessageBytes bounds one JSON-RPC message or batch received over HTTP or
// stdio.
const maxMessageBytes = 128 << 20

// Reconnection of the inbound Server-Sent Events stream.
const (
	initialReconnectionDelay    = time.Second
	maxReconnectionDelay        = 30 * time.Second
	reconnectionDelayGrowFactor = 1.5
	maxReconnectionAttempts     = 2
)

// HTTPTransportOptions configure NewHTTPTransport.
type HTTPTransportOptions struct {
	// URL is the MCP endpoint. It is required.
	URL string

	// Headers are sent with every request.
	Headers http.Header

	// HTTPClient sends requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// AuthProvider authorizes requests with OAuth. On a 401 response the
	// transport runs Auth once and retries; when the user must be redirected
	// first, the request fails with ErrUnauthorized.
	AuthProvider OAuthClientProvider

	// FollowRedirects follows HTTP redirects. By default a redirect fails
	// the request.
	FollowRedirects bool

	// InitialSessionID resumes a Streamable HTTP session after
	// initialization; use it with ClientOptions.InitialInitializeResult.
	InitialSessionID string

	// InitialProtocolVersion is sent before the protocol is negotiated.
	// Empty uses LatestProtocolVersion.
	InitialProtocolVersion string

	// OnSessionIDChange is called when the server creates, changes or
	// clears the session ID; a cleared ID is empty.
	OnSessionIDChange func(sessionID string)

	// OnSessionExpired is called when the server answers 404 for a session
	// ID. The transport clears the ID before reporting the request failure.
	OnSessionExpired func(sessionID string)

	// KeepSessionOnClose skips the DELETE request that terminates the
	// session on Close, so that the application can reattach to it later.
	KeepSessionOnClose bool
}

// HTTPTransport implements the Streamable HTTP transport: messages are
// POSTed to the endpoint and answered with JSON or a Server-Sent Events
// stream, and with the legacy protocol an optional GET stream carries
// server-initiated messages.
type HTTPTransport struct {
	url                     *url.URL
	settings                httpSettings
	client                  *http.Client
	authorizer              *authorizer
	onSessionIDChange       func(sessionID string)
	onSessionExpired        func(sessionID string)
	terminateSessionOnClose bool

	inbox   *inbox
	streams sync.WaitGroup

	mu                sync.Mutex
	lifetime          context.Context
	cancelLifetime    context.CancelFunc
	started           bool
	closed            bool
	sessionID         string
	protocolVersion   string
	inbound           *inboundStream
	lastEventID       string
	reconnectAttempts int
}

// NewHTTPTransport returns a Streamable HTTP transport for options.URL.
func NewHTTPTransport(options HTTPTransportOptions) (*HTTPTransport, error) {
	endpoint, err := parseAbsoluteURL(options.URL)
	if err != nil {
		return nil, &ClientError{
			Message: "invalid mcp http transport url",
			Cause:   err,
		}
	}

	protocolVersion := options.InitialProtocolVersion
	if protocolVersion == "" {
		protocolVersion = LatestProtocolVersion
	}

	client := newHTTPClient(options.HTTPClient, options.FollowRedirects)

	transport := &HTTPTransport{
		url: endpoint,
		settings: httpSettings{
			headers:      options.Headers,
			client:       client,
			authProvider: options.AuthProvider,
		},
		client:                  client,
		onSessionIDChange:       options.OnSessionIDChange,
		onSessionExpired:        options.OnSessionExpired,
		terminateSessionOnClose: !options.KeepSessionOnClose,
		inbox:                   newInbox(),
		sessionID:               options.InitialSessionID,
		protocolVersion:         protocolVersion,
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

// SupportsProtocolVersionDiscovery reports that the client may probe with
// server/discover.
func (t *HTTPTransport) SupportsProtocolVersionDiscovery() bool {
	return true
}

// SupportsToolParameterHeaders reports that x-mcp-header arguments are sent
// as headers.
func (t *HTTPTransport) SupportsToolParameterHeaders() bool {
	return true
}

// SetProtocolVersion records the negotiated protocol version. The legacy
// protocol opens the inbound stream; the stateless protocol closes it.
func (t *HTTPTransport) SetProtocolVersion(version string) {
	t.mu.Lock()
	t.protocolVersion = version
	active := t.started && !t.closed
	modern := version == LatestProtocolVersion

	if active && modern && t.inbound != nil {
		t.inbound.cancel()
		t.inbound = nil
	}
	t.mu.Unlock()

	if active && !modern {
		t.startInbound()
	}
}

// Start prepares the transport and, with a legacy protocol version, opens
// the inbound stream.
func (t *HTTPTransport) Start(ctx context.Context) error {
	t.mu.Lock()
	if t.started {
		t.mu.Unlock()

		return &ClientError{
			Message: "mcp http transport already started",
		}
	}

	t.started = true
	t.lifetime, t.cancelLifetime = context.WithCancel(context.WithoutCancel(ctx))
	modern := t.protocolVersion == LatestProtocolVersion
	t.mu.Unlock()

	if !modern {
		t.startInbound()
	}

	return nil
}

// Receive returns the next message from the server.
func (t *HTTPTransport) Receive(ctx context.Context) (Message, error) {
	return t.inbox.receive(ctx)
}

// Close stops all streams, terminates the session with a DELETE request
// unless KeepSessionOnClose is set, and waits for the transport's
// goroutines.
func (t *HTTPTransport) Close(ctx context.Context) error {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()

		return nil
	}

	t.closed = true
	started := t.started
	sessionID := t.sessionID
	modern := t.protocolVersion == LatestProtocolVersion

	if t.cancelLifetime != nil {
		t.cancelLifetime()
	}
	t.mu.Unlock()

	if started && !modern && sessionID != "" && t.terminateSessionOnClose {
		t.terminateSession(ctx, sessionID)
	}

	t.streams.Wait()
	t.inbox.close(nil)

	return nil
}

// terminateSession asks the server to end the session. Failures are
// ignored: the session expires on its own.
func (t *HTTPTransport) terminateSession(ctx context.Context, sessionID string) {
	headers, err := requestHeaders(ctx, t.settings, nil, t.currentProtocolVersion())
	if err != nil {
		return
	}

	headers.Set("Mcp-Session-Id", sessionID)

	request, err := http.NewRequestWithContext(ctx, http.MethodDelete, t.url.String(), nil)
	if err != nil {
		return
	}

	request.Header = headers

	response, err := t.client.Do(request)
	if err != nil {
		return
	}

	closeBody(response.Body)
}

// Send POSTs message. JSON responses are delivered before Send returns;
// a Server-Sent Events response is read in the background until it ends, ctx
// ends, or the transport closes.
func (t *HTTPTransport) Send(ctx context.Context, message Message, options SendOptions) error {
	lifetime, err := t.activeLifetime()
	if err != nil {
		return err
	}

	requestCtx, cancel := mergeContexts(ctx, lifetime)

	post, err := t.post(requestCtx, message, options)
	if err != nil {
		cancel()

		return err
	}

	stream, err := t.handlePostResponse(requestCtx, message, post)
	if err != nil || stream == nil {
		cancel()

		return err
	}

	return t.readStreamInBackground(requestCtx, cancel, stream)
}

func (t *HTTPTransport) activeLifetime() (context.Context, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.started || t.closed {
		return nil, &ClientError{
			Message: "mcp http transport is not connected",
			Cause:   ErrConnectionClosed,
		}
	}

	return t.lifetime, nil
}

func (t *HTTPTransport) currentProtocolVersion() string {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.protocolVersion
}

func (t *HTTPTransport) isModern() bool {
	return t.currentProtocolVersion() == LatestProtocolVersion
}

// postResult is a POST response with the session ID the request carried.
type postResult struct {
	response  *http.Response
	sessionID string
}

// post sends message, running the OAuth flow and retrying once after a 401
// response.
func (t *HTTPTransport) post(ctx context.Context, message Message, options SendOptions) (postResult, error) {
	result, err := t.postOnce(ctx, message, options)
	if err != nil || !t.needsAuthorization(result.response) {
		return result, err
	}

	closeBody(result.response.Body)

	if err := t.authorizer.recoverUnauthorized(ctx, result.response); err != nil {
		return postResult{}, err
	}

	return t.postOnce(ctx, message, options)
}

func (t *HTTPTransport) needsAuthorization(response *http.Response) bool {
	return response.StatusCode == http.StatusUnauthorized && t.authorizer != nil
}

func (t *HTTPTransport) postOnce(ctx context.Context, message Message, options SendOptions) (postResult, error) {
	headers, sessionID, err := t.postHeaders(ctx, message, options)
	if err != nil {
		return postResult{}, err
	}

	body, err := json.Marshal(message)
	if err != nil {
		return postResult{}, fmt.Errorf("encode json-rpc message: %w", err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, t.url.String(), bytes.NewReader(body))
	if err != nil {
		return postResult{}, fmt.Errorf("create mcp http request: %w", err)
	}

	request.Header = headers

	response, err := t.client.Do(request)
	if err != nil {
		return postResult{}, fmt.Errorf("mcp http transport error: post to %s: %w", t.url, err)
	}

	t.applySessionID(response)

	return postResult{
		response:  response,
		sessionID: sessionID,
	}, nil
}

// postHeaders returns the headers of a POST and the session ID it carries.
func (t *HTTPTransport) postHeaders(ctx context.Context, message Message, options SendOptions) (http.Header, string, error) {
	t.mu.Lock()
	protocolVersion := t.protocolVersion
	sessionID := t.sessionID
	t.mu.Unlock()

	modern := protocolVersion == LatestProtocolVersion

	if message.Method == "initialize" {
		sessionID = ""
	}

	base := http.Header{
		"Content-Type": {
			"application/json",
		},
		"Accept": {
			"application/json, text/event-stream",
		},
	}

	if modern {
		maps.Copy(base, options.Headers)

		if message.Kind() == MessageKindRequest {
			addStandardRequestHeaders(base, message)
		}
	}

	headers, err := requestHeaders(ctx, t.settings, base, protocolVersion)
	if err != nil {
		return nil, "", err
	}

	if !modern && sessionID != "" {
		headers.Set("Mcp-Session-Id", sessionID)
	}

	return headers, sessionID, nil
}

// addStandardRequestHeaders adds Mcp-Method and, for requests naming a tool,
// prompt or resource, Mcp-Name.
func addStandardRequestHeaders(headers http.Header, message Message) {
	headers.Set("Mcp-Method", message.Method)

	var params struct {
		Name *string `json:"name"`
		URI  *string `json:"uri"`
	}

	if message.Params == nil || json.Unmarshal(message.Params, &params) != nil {
		return
	}

	var name *string
	switch message.Method {
	case "resources/read":
		name = params.URI
	case "tools/call", "prompts/get":
		name = params.Name
	}

	if name != nil {
		headers.Set("Mcp-Name", encodeHeaderValue(*name))
	}
}

// handlePostResponse delivers or reports the response to a POST. It returns
// the body of a Server-Sent Events response, which the caller reads and
// closes, and closes every other body.
func (t *HTTPTransport) handlePostResponse(ctx context.Context, message Message, post postResult) (io.ReadCloser, error) {
	response := post.response

	if response.StatusCode == http.StatusAccepted {
		closeBody(response.Body)

		// A server without an inbound stream before initialization may
		// accept one now.
		if !t.isModern() {
			t.startInbound()
		}

		return nil, nil
	}

	if response.StatusCode < 200 || response.StatusCode > 299 {
		defer closeBody(response.Body)

		return nil, t.failedPost(ctx, message, post)
	}

	// Notifications expect no response, although some servers acknowledge
	// them with 200 instead of 202.
	if message.ID == nil {
		closeBody(response.Body)

		return nil, nil
	}

	contentType := response.Header.Get("Content-Type")

	switch {
	case strings.Contains(contentType, "application/json"):
		defer closeBody(response.Body)

		return nil, t.deliverJSONResponse(ctx, response.Body)
	case strings.Contains(contentType, "text/event-stream"):
		return response.Body, nil
	default:
		closeBody(response.Body)

		return nil, &ClientError{
			Message:    "mcp http transport error: unexpected content type: " + contentType,
			StatusCode: response.StatusCode,
			URL:        t.url.String(),
		}
	}
}

// failedPost delivers a JSON-RPC error carried by a failed response and
// otherwise returns a *ClientError describing the failure.
func (t *HTTPTransport) failedPost(ctx context.Context, message Message, post postResult) error {
	response := post.response
	body, read := readErrorBody(response)

	if message.ID != nil && read {
		if errorMessage, err := ParseMessage([]byte(body)); err == nil && errorMessage.Kind() == MessageKindError {
			if errorMessage.ID == nil {
				errorMessage.ID = message.ID
			}

			t.inbox.deliverMessage(ctx, errorMessage)

			return nil
		}
	}

	errorText := fmt.Sprintf("mcp http transport error: posting to endpoint (http %d): %s", response.StatusCode, body)

	if response.StatusCode == http.StatusNotFound && !t.isModern() {
		if post.sessionID != "" {
			t.expireSession(post.sessionID)

			errorText += ". the mcp session expired. create a new client without an initial session id to start a fresh session"
		} else {
			errorText += ". this server does not support http transport. try using the sse transport instead"
		}
	}

	return &ClientError{
		Message:      errorText,
		StatusCode:   response.StatusCode,
		URL:          t.url.String(),
		ResponseBody: body,
	}
}

// deliverJSONResponse delivers one message or a batch.
func (t *HTTPTransport) deliverJSONResponse(ctx context.Context, body io.Reader) error {
	messages, err := decodeMessages(body)
	if err != nil {
		return &ClientError{
			Message: "mcp http transport error: failed to parse response",
			Cause:   err,
		}
	}

	for _, message := range messages {
		if !t.inbox.deliverMessage(ctx, message) {
			return ctx.Err()
		}
	}

	return nil
}

// decodeMessages decodes a JSON-RPC message or a batch of messages.
func decodeMessages(body io.Reader) ([]Message, error) {
	data, err := io.ReadAll(io.LimitReader(body, maxMessageBytes))
	if err != nil {
		return nil, err
	}

	if !isJSONArray(bytes.TrimSpace(data)) {
		message, err := ParseMessage(data)
		if err != nil {
			return nil, err
		}

		return []Message{
			message,
		}, nil
	}

	var items []json.RawMessage
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}

	messages := make([]Message, 0, len(items))
	for _, item := range items {
		message, err := ParseMessage(item)
		if err != nil {
			return nil, err
		}

		messages = append(messages, message)
	}

	return messages, nil
}

// readStreamInBackground reads a response stream on a goroutine that Close
// waits for. The goroutine owns cancel and body.
func (t *HTTPTransport) readStreamInBackground(ctx context.Context, cancel context.CancelFunc, body io.ReadCloser) error {
	read := func() {
		defer cancel()
		defer closeBody(body)

		t.readEvents(ctx, body, false)
	}

	if !t.spawn(read) {
		cancel()
		closeBody(body)

		return &ClientError{
			Message: "mcp http transport is not connected",
			Cause:   ErrConnectionClosed,
		}
	}

	return nil
}

// spawn runs work on a goroutine that Close waits for, unless the
// transport is closed.
func (t *HTTPTransport) spawn(work func()) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return false
	}

	t.streams.Go(work)

	return true
}

// readEvents delivers the messages of a Server-Sent Events stream until it
// ends. It reports whether the stream failed, as opposed to ending. The
// event IDs of a resumable stream are kept for reconnecting.
func (t *HTTPTransport) readEvents(ctx context.Context, body io.Reader, resumable bool) bool {
	for event, err := range sse.Read(body) {
		if err != nil {
			if ctx.Err() == nil {
				t.inbox.deliverError(ctx, fmt.Errorf("mcp http transport error: read event stream: %w", err))
			}

			return ctx.Err() == nil
		}

		if resumable && event.ID != "" {
			t.recordEventID(event.ID)
		}

		if !isMessageEvent(event.Type) {
			continue
		}

		if !deliverEventMessage(ctx, t.inbox, event.Data, "mcp http transport error: failed to parse message") {
			return false
		}
	}

	return false
}

func isMessageEvent(eventType string) bool {
	return eventType == "" || eventType == "message"
}

// deliverEventMessage parses an event's data and delivers the message, or
// the parse failure. It reports false when delivery stopped.
func deliverEventMessage(ctx context.Context, destination *inbox, data string, failure string) bool {
	message, err := ParseMessage([]byte(data))
	if err != nil {
		return destination.deliverError(ctx, &ClientError{
			Message: failure,
			Cause:   err,
		})
	}

	return destination.deliverMessage(ctx, message)
}

func (t *HTTPTransport) recordEventID(id string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.lastEventID = id
}

// applySessionID records the session ID the server assigned, with the
// legacy protocol.
func (t *HTTPTransport) applySessionID(response *http.Response) {
	sessionID := response.Header.Get("Mcp-Session-Id")
	if sessionID == "" || t.isModern() {
		return
	}

	t.setSessionID(sessionID)
}

func (t *HTTPTransport) setSessionID(sessionID string) {
	t.mu.Lock()
	changed := t.sessionID != sessionID
	t.sessionID = sessionID
	t.mu.Unlock()

	if changed && t.onSessionIDChange != nil {
		t.onSessionIDChange(sessionID)
	}
}

// expireSession clears an expired session ID and reports it.
func (t *HTTPTransport) expireSession(sessionID string) {
	t.mu.Lock()
	current := t.sessionID == sessionID
	t.mu.Unlock()

	if current {
		t.setSessionID("")
	}

	if t.onSessionExpired != nil {
		t.onSessionExpired(sessionID)
	}
}

// startInbound opens the optional inbound stream of the legacy protocol
// unless it is open or the transport is closed.
func (t *HTTPTransport) startInbound() {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.started || t.closed || t.inbound != nil || t.protocolVersion == LatestProtocolVersion {
		return
	}

	ctx, cancel := context.WithCancel(t.lifetime)
	stream := &inboundStream{
		cancel: cancel,
	}
	t.inbound = stream

	t.streams.Go(func() {
		defer t.finishInbound(stream)

		t.runInbound(ctx)
	})
}

// inboundStream is the running inbound stream of the legacy protocol.
type inboundStream struct {
	cancel context.CancelFunc
}

// finishInbound forgets the inbound stream so that a later 202 response may
// open a new one.
func (t *HTTPTransport) finishInbound(stream *inboundStream) {
	stream.cancel()

	t.mu.Lock()
	defer t.mu.Unlock()

	if t.inbound == stream {
		t.inbound = nil
	}
}

// runInbound receives the inbound stream, reconnecting with the last event
// ID after failures up to maxReconnectionAttempts times.
func (t *HTTPTransport) runInbound(ctx context.Context) {
	resumeToken := ""

	for {
		if !t.receiveInbound(ctx, resumeToken) || ctx.Err() != nil {
			return
		}

		delay, ok := t.nextReconnectionDelay()
		if !ok {
			t.inbox.deliverError(ctx, &ClientError{
				Message: fmt.Sprintf("mcp http transport error: maximum reconnection attempts (%d) exceeded", maxReconnectionAttempts),
			})

			return
		}

		if poll.Wait(ctx, delay) != nil {
			return
		}

		t.mu.Lock()
		resumeToken = t.lastEventID
		t.mu.Unlock()
	}
}

// receiveInbound opens and reads the inbound stream once. It reports whether
// the stream failed in a way that warrants reconnecting.
func (t *HTTPTransport) receiveInbound(ctx context.Context, resumeToken string) bool {
	sessionID := t.currentSessionID()

	response, reconnect, err := t.openInbound(ctx, resumeToken)
	if err != nil {
		if ctx.Err() != nil {
			return false
		}

		t.inbox.deliverError(ctx, err)

		return reconnect
	}
	defer closeBody(response.Body)

	if response.StatusCode == http.StatusMethodNotAllowed {
		return false
	}

	if response.StatusCode < 200 || response.StatusCode > 299 {
		if response.StatusCode == http.StatusNotFound && sessionID != "" {
			t.expireSession(sessionID)
		}

		t.inbox.deliverError(ctx, &ClientError{
			Message:    fmt.Sprintf("mcp http transport error: get sse failed: %d %s", response.StatusCode, http.StatusText(response.StatusCode)),
			StatusCode: response.StatusCode,
			URL:        t.url.String(),
		})

		return false
	}

	t.mu.Lock()
	t.reconnectAttempts = 0
	t.mu.Unlock()

	return t.readEvents(ctx, response.Body, true)
}

// openInbound sends the GET request, running the OAuth flow once after a
// 401 response. reconnect reports whether a failure was a network failure.
func (t *HTTPTransport) openInbound(ctx context.Context, resumeToken string) (response *http.Response, reconnect bool, err error) {
	response, err = t.getInbound(ctx, resumeToken)
	if err != nil {
		return nil, true, err
	}

	if !t.needsAuthorization(response) {
		return response, false, nil
	}

	closeBody(response.Body)

	if err := t.authorizer.recoverUnauthorized(ctx, response); err != nil {
		return nil, false, err
	}

	response, err = t.getInbound(ctx, resumeToken)
	if err != nil {
		return nil, true, err
	}

	return response, false, nil
}

func (t *HTTPTransport) getInbound(ctx context.Context, resumeToken string) (*http.Response, error) {
	base := http.Header{
		"Accept": {
			"text/event-stream",
		},
	}

	headers, err := requestHeaders(ctx, t.settings, base, t.currentProtocolVersion())
	if err != nil {
		return nil, err
	}

	if sessionID := t.currentSessionID(); sessionID != "" {
		headers.Set("Mcp-Session-Id", sessionID)
	}

	if resumeToken != "" {
		headers.Set("Last-Event-Id", resumeToken)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, t.url.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("create mcp http request: %w", err)
	}

	request.Header = headers

	response, err := t.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("mcp http transport error: get %s: %w", t.url, err)
	}

	t.applySessionID(response)

	return response, nil
}

func (t *HTTPTransport) currentSessionID() string {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.sessionID
}

// nextReconnectionDelay returns the backoff before the next reconnection,
// or false when the attempts are exhausted.
func (t *HTTPTransport) nextReconnectionDelay() (time.Duration, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.reconnectAttempts >= maxReconnectionAttempts {
		return 0, false
	}

	growth := math.Pow(reconnectionDelayGrowFactor, float64(t.reconnectAttempts))
	delay := min(time.Duration(float64(initialReconnectionDelay)*growth), maxReconnectionDelay)

	t.reconnectAttempts++

	return delay, true
}

var _ interface {
	Transport
	ProtocolVersionTransport
	ProtocolDiscoveryTransport
	ToolParameterHeaderTransport
} = (*HTTPTransport)(nil)
