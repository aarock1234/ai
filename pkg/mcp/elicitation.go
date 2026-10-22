package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

// ElicitationRequest asks the user, through the client, for information
// described by RequestedSchema.
type ElicitationRequest struct {
	Message string `json:"message"`

	// RequestedSchema is the JSON Schema of the requested content.
	RequestedSchema json.RawMessage `json:"requestedSchema,omitempty"`

	Meta map[string]json.RawMessage `json:"_meta,omitempty"`
}

// ElicitAction is the user's response to an elicitation request.
type ElicitAction string

// Elicitation actions.
const (
	ElicitActionAccept  ElicitAction = "accept"
	ElicitActionDecline ElicitAction = "decline"
	ElicitActionCancel  ElicitAction = "cancel"
)

// ElicitResult is the client's answer to an elicitation request.
type ElicitResult struct {
	Action ElicitAction `json:"action"`

	// Content is the submitted data when Action is ElicitActionAccept.
	Content map[string]any `json:"content,omitempty"`

	Meta map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r ElicitResult) validate() error {
	switch r.Action {
	case ElicitActionAccept, ElicitActionDecline, ElicitActionCancel:
		return nil
	default:
		return fmt.Errorf("invalid elicitation action %q", r.Action)
	}
}

// ElicitationHandler answers elicitation/create requests from the server.
// ctx is canceled when the server cancels the request or the client closes.
type ElicitationHandler func(ctx context.Context, request ElicitationRequest) (ElicitResult, error)

// HandleElicitation registers the handler for elicitation requests. Advertise
// ClientCapabilities.Elicitation so that the server sends them. Without a
// handler, the client answers them with a method-not-found error.
func (c *Client) HandleElicitation(handler ElicitationHandler) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.elicitationHandler = handler
}

// handleServerRequest answers one request from the server. It runs on its
// own goroutine so that a slow handler does not block message reading.
func (c *Client) handleServerRequest(message Message) {
	ctx, cancel := context.WithCancel(c.lifetime)
	defer cancel()

	key := message.ID.String()
	c.trackServerRequest(key, cancel)
	defer c.untrackServerRequest(key)

	response := c.answerServerRequest(ctx, message)

	// A request the server cancelled must not be answered.
	if ctx.Err() != nil {
		return
	}

	if err := c.transport.Send(c.lifetime, response, SendOptions{}); err != nil {
		c.reportError(fmt.Errorf("respond to server request %s: %w", message.Method, err))
	}
}

func (c *Client) answerServerRequest(ctx context.Context, message Message) Message {
	id := *message.ID

	switch message.Method {
	case "ping":
		return newResultMessage(id, json.RawMessage(`{}`))
	case "elicitation/create":
		return c.answerElicitation(ctx, id, message.Params)
	default:
		return newErrorMessage(id, RPCError{
			Code:    errorCodeMethodNotFound,
			Message: "unsupported request method: " + message.Method,
		})
	}
}

func (c *Client) answerElicitation(ctx context.Context, id ID, params json.RawMessage) Message {
	c.mu.Lock()
	handler := c.elicitationHandler
	c.mu.Unlock()

	if handler == nil {
		return newErrorMessage(id, RPCError{
			Code:    errorCodeMethodNotFound,
			Message: "no elicitation handler registered on client",
		})
	}

	request, err := decodeElicitationRequest(params)
	if err != nil {
		return newErrorMessage(id, RPCError{
			Code:    errorCodeInvalidParams,
			Message: "invalid elicitation request: " + err.Error(),
		})
	}

	result, err := handler(ctx, request)
	if err == nil {
		err = result.validate()
	}

	if err != nil {
		c.reportError(fmt.Errorf("handle elicitation request: %w", err))

		return newErrorMessage(id, RPCError{
			Code:    errorCodeInternalError,
			Message: err.Error(),
		})
	}

	encoded, err := json.Marshal(result)
	if err != nil {
		return newErrorMessage(id, RPCError{
			Code:    errorCodeInternalError,
			Message: "failed to encode elicitation result",
		})
	}

	return newResultMessage(id, encoded)
}

func decodeElicitationRequest(params json.RawMessage) (ElicitationRequest, error) {
	if params == nil {
		return ElicitationRequest{}, errors.New("params are required")
	}

	if err := requireFields(params, "message"); err != nil {
		return ElicitationRequest{}, err
	}

	var request ElicitationRequest
	if err := json.Unmarshal(params, &request); err != nil {
		return ElicitationRequest{}, err
	}

	return request, nil
}

func (c *Client) trackServerRequest(key string, cancel context.CancelFunc) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.serverRequests[key] = cancel
}

func (c *Client) untrackServerRequest(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.serverRequests, key)
}

// cancelServerRequest stops the handler of the request named by a
// notifications/cancelled message.
func (c *Client) cancelServerRequest(params json.RawMessage) {
	var cancelled cancelledParams
	if err := json.Unmarshal(params, &cancelled); err != nil {
		c.reportError(fmt.Errorf("decode notifications/cancelled: %w", err))

		return
	}

	c.mu.Lock()
	cancel, ok := c.serverRequests[cancelled.RequestID.String()]
	c.mu.Unlock()

	if ok {
		cancel()
	}
}
