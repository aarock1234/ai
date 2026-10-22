package mcp

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
)

// ErrConnectionClosed reports that the connection to the server closed while
// a request was pending or before it was sent.
var ErrConnectionClosed = errors.New("mcp connection closed")

// ErrTransportClosed is returned by Transport.Receive once no further
// messages will arrive.
var ErrTransportClosed = errors.New("mcp transport closed")

// ErrUnauthorized reports that the server rejected the request and OAuth
// authorization did not complete, for example because the user must first be
// redirected to the authorization server.
var ErrUnauthorized = errors.New("unauthorized")

// ClientError reports a failure of the MCP client: a JSON-RPC error from the
// server, a failed HTTP exchange, an invalid response or a protocol
// violation.
type ClientError struct {
	Message string

	// Code is the JSON-RPC error code from the server's error response, such
	// as -32601 (method not found) or -32602 (invalid params). It is nil for
	// failures without a JSON-RPC error.
	Code *int

	// Data is the data member of the server's JSON-RPC error, if any.
	Data json.RawMessage

	// StatusCode is the HTTP status of the failed response for HTTP
	// transports, and zero otherwise.
	StatusCode int

	// URL is the endpoint of the failed HTTP request.
	URL string

	// ResponseBody is the body of the failed HTTP response, when it could be
	// read.
	ResponseBody string

	Cause error
}

// Error returns the message and, when present, the cause.
func (e *ClientError) Error() string {
	if e.Cause == nil {
		return "mcp: " + e.Message
	}

	return fmt.Sprintf("mcp: %s: %v", e.Message, e.Cause)
}

// Unwrap returns the cause.
func (e *ClientError) Unwrap() error {
	return e.Cause
}

// OAuthErrorCode is the error code of an OAuth error response.
type OAuthErrorCode string

// OAuth error codes the authorization flow distinguishes. Other codes from
// authorization servers are preserved as received.
const (
	OAuthServerError                 OAuthErrorCode = "server_error"
	OAuthInvalidClient               OAuthErrorCode = "invalid_client"
	OAuthInvalidGrant                OAuthErrorCode = "invalid_grant"
	OAuthUnauthorizedClient          OAuthErrorCode = "unauthorized_client"
	OAuthAuthorizationServerMismatch OAuthErrorCode = "authorization_server_mismatch"
)

// OAuthError reports a failure in the OAuth flow. Code is empty for failures
// that are not OAuth error responses, such as an endpoint that may not be
// requested.
type OAuthError struct {
	Code    OAuthErrorCode
	Message string

	// URI is the error_uri of an OAuth error response.
	URI string

	Cause error
}

// Error returns the code and message.
func (e *OAuthError) Error() string {
	message := "mcp oauth"
	if e.Code != "" {
		message += " " + string(e.Code)
	}

	if e.Message != "" {
		message += ": " + e.Message
	}

	if e.Cause != nil {
		message += fmt.Sprintf(": %v", e.Cause)
	}

	return message
}

// Unwrap returns the cause.
func (e *OAuthError) Unwrap() error {
	return e.Cause
}

// hasOAuthErrorCode reports whether err is an OAuthError with one of codes.
func hasOAuthErrorCode(err error, codes ...OAuthErrorCode) bool {
	oauthErr, ok := errors.AsType[*OAuthError](err)
	if !ok {
		return false
	}

	return slices.Contains(codes, oauthErr.Code)
}

func authorizationServerMismatch(message string) *OAuthError {
	return &OAuthError{
		Code:    OAuthAuthorizationServerMismatch,
		Message: message,
	}
}
