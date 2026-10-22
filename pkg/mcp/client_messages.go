package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// cancelNotificationTimeout bounds sending notifications/cancelled after the
// caller abandoned a request.
const cancelNotificationTimeout = 5 * time.Second

// Keys of the protocol metadata added to every request of the stateless
// protocol.
const (
	metaProtocolVersion    = "io.modelcontextprotocol/protocolVersion"
	metaClientCapabilities = "io.modelcontextprotocol/clientCapabilities"
	metaClientInfo         = "io.modelcontextprotocol/clientInfo"
	metaServerInfo         = "io.modelcontextprotocol/serverInfo"
)

// rpcCall describes one request: its method, parameters to encode, optional
// HTTP headers, and the result value to decode into.
type rpcCall struct {
	method  string
	params  any
	headers http.Header
	result  any
}

// request sends call and waits for its response. Canceling ctx abandons the
// request and notifies the server.
func (c *Client) request(ctx context.Context, call rpcCall) error {
	if err := c.checkRequestAllowed(call.method); err != nil {
		return err
	}

	if ctx.Err() != nil {
		return abortedRequestError(ctx)
	}

	params, err := c.encodeRequestParams(call.params)
	if err != nil {
		return err
	}

	id := c.nextRequestID.Add(1) - 1

	outcome, err := c.registerPending(id)
	if err != nil {
		return err
	}
	defer c.unregisterPending(id)

	requestCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	message := newRequestMessage(NumberID(id), call.method, params)

	sendOptions := SendOptions{
		Headers: call.headers,
	}

	if err := c.transport.Send(requestCtx, message, sendOptions); err != nil {
		if ctx.Err() != nil {
			c.notifyCancelled(ctx, id, call.method)

			return abortedRequestError(ctx)
		}

		return err
	}

	select {
	case response := <-outcome:
		return decodeResponse(response, call.result, c.isModernProtocol())
	case <-ctx.Done():
		c.notifyCancelled(ctx, id, call.method)

		return abortedRequestError(ctx)
	}
}

func abortedRequestError(ctx context.Context) error {
	return &ClientError{
		Message: "request was aborted",
		Cause:   context.Cause(ctx),
	}
}

func (c *Client) checkRequestAllowed(method string) error {
	c.mu.Lock()
	closed := c.closed || c.stopping
	c.mu.Unlock()

	if closed {
		return &ClientError{
			Message: "attempted to send a request from a closed client",
			Cause:   ErrConnectionClosed,
		}
	}

	return c.assertCapability(method)
}

// assertCapability rejects requests for features the server did not
// advertise.
func (c *Client) assertCapability(method string) error {
	capabilities := c.serverCapabilities()

	switch method {
	case "initialize", "server/discover":
		return nil
	case "completion/complete":
		return requireCapability(capabilities.Completions != nil, "completions")
	case "events/list", "events/subscribe", "events/unsubscribe":
		return requireCapability(capabilities.Events != nil, "events")
	case "tools/list", "tools/call":
		return requireCapability(capabilities.Tools != nil, "tools")
	case "resources/list", "resources/read", "resources/templates/list":
		return requireCapability(capabilities.Resources != nil, "resources")
	case "prompts/list", "prompts/get":
		return requireCapability(capabilities.Prompts != nil, "prompts")
	default:
		return &ClientError{
			Message: "unsupported method: " + method,
		}
	}
}

func requireCapability(supported bool, feature string) error {
	if supported {
		return nil
	}

	return &ClientError{
		Message: "server does not support " + feature,
	}
}

func (c *Client) isModernProtocol() bool {
	era, _ := c.protocol()

	return era == protocolEraModern
}

// encodeRequestParams encodes params and, in the stateless protocol, adds
// the protocol metadata the server needs on every request.
func (c *Client) encodeRequestParams(params any) (json.RawMessage, error) {
	encoded, err := encodeParams(params)
	if err != nil {
		return nil, err
	}

	era, version := c.protocol()
	if era != protocolEraModern {
		return encoded, nil
	}

	return addProtocolMeta(encoded, map[string]any{
		metaProtocolVersion:    version,
		metaClientCapabilities: c.capabilities,
		metaClientInfo:         c.clientInfo,
	})
}

// encodeParams encodes params, omitting parameters that are absent or
// empty.
func encodeParams(params any) (json.RawMessage, error) {
	if params == nil {
		return nil, nil
	}

	encoded, err := json.Marshal(params)
	if err != nil {
		return nil, &ClientError{
			Message: "failed to encode request parameters",
			Cause:   err,
		}
	}

	if string(encoded) == "{}" {
		return nil, nil
	}

	return encoded, nil
}

// addProtocolMeta merges entries into the _meta member of the params object.
func addProtocolMeta(params json.RawMessage, entries map[string]any) (json.RawMessage, error) {
	members := make(map[string]json.RawMessage)
	if params != nil {
		if err := json.Unmarshal(params, &members); err != nil {
			return nil, fmt.Errorf("decode request parameters: %w", err)
		}
	}

	meta := make(map[string]any)
	if existing, ok := members["_meta"]; ok {
		var existingMeta map[string]json.RawMessage
		if err := json.Unmarshal(existing, &existingMeta); err != nil {
			return nil, fmt.Errorf("decode request _meta: %w", err)
		}

		for key, value := range existingMeta {
			meta[key] = value
		}
	}

	for key, value := range entries {
		meta[key] = value
	}

	encodedMeta, err := json.Marshal(meta)
	if err != nil {
		return nil, fmt.Errorf("encode request _meta: %w", err)
	}

	members["_meta"] = encodedMeta

	return json.Marshal(members)
}

// decodeResponse converts a response into the result value or a
// ClientError.
func decodeResponse(outcome responseOutcome, result any, modern bool) error {
	if outcome.err != nil {
		return outcome.err
	}

	if rpcError := outcome.message.Error; rpcError != nil {
		code := rpcError.Code

		return &ClientError{
			Message: rpcError.Message,
			Code:    &code,
			Data:    rpcError.Data,
			Cause:   rpcError,
		}
	}

	raw := outcome.message.Result

	if err := checkResultType(raw, modern); err != nil {
		return err
	}

	if err := json.Unmarshal(raw, result); err != nil {
		return &ClientError{
			Message: "failed to parse server response",
			Cause:   err,
		}
	}

	if validator, ok := result.(resultValidator); ok {
		if err := validator.validate(raw); err != nil {
			return &ClientError{
				Message: "failed to parse server response",
				Cause:   err,
			}
		}
	}

	return nil
}

// checkResultType rejects results that the client cannot complete: modern
// results without a resultType and requests for additional input.
func checkResultType(raw json.RawMessage, modern bool) error {
	var envelope struct {
		ResultType *string `json:"resultType"`
	}

	if err := json.Unmarshal(raw, &envelope); err != nil {
		return &ClientError{
			Message: "failed to parse server response",
			Cause:   err,
		}
	}

	if modern && envelope.ResultType == nil {
		return &ClientError{
			Message: "modern mcp result is missing resultType",
		}
	}

	if envelope.ResultType != nil && *envelope.ResultType == "input_required" {
		return &ClientError{
			Message: "server requested additional input, but multi round-trip requests are not supported yet",
		}
	}

	return nil
}

// notify sends a notification.
func (c *Client) notify(ctx context.Context, method string, params any) error {
	encoded, err := encodeParams(params)
	if err != nil {
		return err
	}

	message := newNotificationMessage(method, encoded)

	return c.transport.Send(ctx, message, SendOptions{})
}

// cancelledParams are the parameters of notifications/cancelled.
type cancelledParams struct {
	RequestID ID     `json:"requestId"`
	Reason    string `json:"reason,omitempty"`
}

// notifyCancelled tells the server that the client abandoned request id. The
// initialize request must not be cancelled. The notification is sent in the
// background, bounded by cancelNotificationTimeout.
func (c *Client) notifyCancelled(ctx context.Context, id int64, method string) {
	if method == "initialize" {
		return
	}

	params := cancelledParams{
		RequestID: NumberID(id),
		Reason:    context.Cause(ctx).Error(),
	}

	c.spawn(func() {
		sendCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cancelNotificationTimeout)
		defer cancel()

		if err := c.notify(sendCtx, "notifications/cancelled", params); err != nil {
			c.reportError(fmt.Errorf("send cancellation of request %d: %w", id, err))
		}
	})
}

func (c *Client) registerPending(id int64) (chan responseOutcome, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed || c.stopping {
		return nil, &ClientError{
			Message: "attempted to send a request from a closed client",
			Cause:   ErrConnectionClosed,
		}
	}

	outcome := make(chan responseOutcome, 1)
	c.pending[id] = outcome

	return outcome, nil
}

func (c *Client) unregisterPending(id int64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.pending, id)
}

// takePending removes and returns the pending request that id answers.
func (c *Client) takePending(id ID) (chan responseOutcome, bool) {
	number, ok := id.correlationNumber()
	if !ok {
		return nil, false
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	outcome, found := c.pending[number]
	if found {
		delete(c.pending, number)
	}

	return outcome, found
}

// readMessages receives messages until the transport closes. It runs on the
// client's reader goroutine.
func (c *Client) readMessages() {
	for {
		message, err := c.transport.Receive(c.lifetime)
		if err != nil {
			if errors.Is(err, ErrTransportClosed) || c.lifetime.Err() != nil {
				c.connectionClosed()

				return
			}

			c.reportError(err)

			continue
		}

		c.dispatch(message)
	}
}

func (c *Client) dispatch(message Message) {
	switch message.Kind() {
	case MessageKindRequest:
		c.spawn(func() {
			c.handleServerRequest(message)
		})
	case MessageKindNotification:
		c.handleNotification(message)
	default:
		c.handleResponse(message)
	}
}

func (c *Client) handleResponse(message Message) {
	if message.ID == nil {
		c.reportError(&ClientError{
			Message: "protocol error: received a response without a message id: " + encodeForError(message),
		})

		return
	}

	outcome, found := c.takePending(*message.ID)
	if !found {
		c.reportError(&ClientError{
			Message: "protocol error: received a response for an unknown message id: " + encodeForError(message),
		})

		return
	}

	outcome <- responseOutcome{
		message: message,
	}
}

func (c *Client) handleNotification(message Message) {
	if message.Method == "notifications/cancelled" {
		c.cancelServerRequest(message.Params)
	}

	notification := Notification{
		Method: message.Method,
		Params: message.Params,
	}

	if c.onNotification != nil {
		c.onNotification(c.lifetime, notification)

		return
	}

	if message.Method == "notifications/cancelled" {
		return
	}

	c.reportError(&ClientError{
		Message: "unsupported message type: notification " + message.Method,
	})
}

func encodeForError(message Message) string {
	encoded, err := json.Marshal(message)
	if err != nil {
		return fmt.Sprintf("%+v", message)
	}

	return string(encoded)
}
