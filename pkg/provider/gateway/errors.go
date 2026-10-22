package gateway

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/pkg/model"
)

// ErrorType classifies a gateway Error.
type ErrorType string

// Error types. The first eight are reported by the gateway; the others are
// raised by this package.
const (
	ErrorAuthentication   ErrorType = "authentication_error"
	ErrorInvalidRequest   ErrorType = "invalid_request_error"
	ErrorRateLimit        ErrorType = "rate_limit_exceeded"
	ErrorModelNotFound    ErrorType = "model_not_found"
	ErrorNotFound         ErrorType = "not_found"
	ErrorInternalServer   ErrorType = "internal_server_error"
	ErrorFailedDependency ErrorType = "failed_dependency"
	ErrorForbidden        ErrorType = "forbidden"

	// ErrorTimeout reports a request that timed out on the client.
	ErrorTimeout ErrorType = "timeout_error"

	// ErrorResponse reports an error response the gateway did not describe
	// in its documented format.
	ErrorResponse ErrorType = "response_error"
)

// Error is a failed AI Gateway request. Test for it with errors.As; Cause
// holds the underlying *model.APICallError when the gateway responded.
type Error struct {
	Type    ErrorType
	Message string

	// StatusCode is the HTTP status, or the status the error type implies.
	StatusCode int

	// GenerationID identifies the failed generation, when reported.
	GenerationID string

	// IsRetryable reports whether repeating the request may succeed.
	IsRetryable bool

	// ModelID is the unknown model of a model_not_found error.
	ModelID string

	// RuleID is the routing rule that rejected a forbidden request.
	RuleID string

	// Response is the undocumented error body of a response_error.
	Response json.RawMessage

	Cause error
}

// Error returns the message, followed by the generation ID when known.
func (e *Error) Error() string {
	if e.GenerationID == "" {
		return e.Message
	}

	return fmt.Sprintf("%s [%s]", e.Message, e.GenerationID)
}

// Unwrap returns the underlying cause.
func (e *Error) Unwrap() error {
	return e.Cause
}

// Contextual authentication messages, by how the request was authenticated.
const (
	apiKeyAuthenticationMessage = `AI Gateway authentication failed: Invalid API key or token.

Create a new API key: https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%2Fapi-keys

Provide an API key or Vercel access token via 'apiKey' option or 'AI_GATEWAY_API_KEY' environment variable.`

	oidcAuthenticationMessage = `AI Gateway authentication failed: Invalid OIDC token.

Run 'npx vercel link' to link your project, then 'vc env pull' to fetch the token.

Alternatively, use an API key: https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%2Fapi-keys
or pass a Vercel access token via the 'apiKey' option.`

	missingAuthenticationMessage = `AI Gateway authentication failed: No authentication provided.

Option 1 - API key:
Create an API key: https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%2Fapi-keys
Provide via 'apiKey' option or 'AI_GATEWAY_API_KEY' environment variable.

Option 2 - Vercel access token:
Pass a Vercel personal access token or Vercel app access token via the 'apiKey' option.

Option 3 - OIDC token:
Run 'npx vercel link' to link your project, then 'vc env pull' to fetch the token.`

	timeoutMessage = `Gateway request timed out: %s

    This is a client-side timeout. To resolve this, increase your timeout configuration: https://vercel.com/docs/ai-gateway/capabilities/video-generation#extending-timeouts-for-node.js`
)

// errorBody is the documented gateway error response.
type errorBody struct {
	Error        *errorDetail `json:"error"`
	GenerationID *string      `json:"generationId"`
}

type errorDetail struct {
	Message *string         `json:"message"`
	Type    *string         `json:"type"`
	Param   json.RawMessage `json:"param"`
}

type modelNotFoundParam struct {
	ModelID *string `json:"modelId"`
}

type forbiddenParam struct {
	RuleID *string `json:"ruleId"`
}

// asGatewayError converts a failed call into an *Error. Gateway errors and
// failures that are not API calls, including cancellation, are returned
// unchanged.
func asGatewayError(err error, method authMethod) error {
	if err == nil {
		return nil
	}

	if _, ok := errors.AsType[*Error](err); ok {
		return err
	}

	apiErr, ok := errors.AsType[*model.APICallError](err)
	if !ok {
		return err
	}

	if apiErr.StatusCode == 0 && isTimeout(apiErr.Cause) {
		return &Error{
			Type:        ErrorTimeout,
			Message:     fmt.Sprintf(timeoutMessage, apiErr.Message),
			StatusCode:  http.StatusRequestTimeout,
			IsRetryable: true,
			Cause:       err,
		}
	}

	statusCode := apiErr.StatusCode
	if statusCode == 0 {
		statusCode = http.StatusInternalServerError
	}

	var retryable *bool
	if apiErr.IsRetryable && (apiErr.StatusCode == 0 || apiErr.StatusCode < 400) {
		retryable = new(true)
	}

	return errorFromResponse(errorResponseSettings{
		body:       []byte(apiErr.ResponseBody),
		statusCode: statusCode,
		cause:      err,
		method:     method,
		retryable:  retryable,
	})
}

// timeoutError is implemented by transport errors that can report a
// timeout.
type timeoutError interface {
	error
	Timeout() bool
}

func isTimeout(err error) bool {
	timeout, ok := errors.AsType[timeoutError](err)

	return ok && timeout.Timeout()
}

// errorResponseSettings describe a failed response.
type errorResponseSettings struct {
	body       []byte
	statusCode int
	cause      error
	method     authMethod

	// retryable overrides the retryability implied by the status code.
	retryable *bool
}

// errorFromResponse builds the Error of a documented error response, or a
// response_error when the body has another shape.
func errorFromResponse(settings errorResponseSettings) *Error {
	var body errorBody
	decodeErr := json.Unmarshal(settings.body, &body)

	if decodeErr != nil || body.Error == nil || body.Error.Message == nil {
		return responseError(settings, body.GenerationID)
	}

	gatewayErr := &Error{
		Type:        ErrorType(stringValue(body.Error.Type)),
		Message:     *body.Error.Message,
		StatusCode:  settings.statusCode,
		IsRetryable: retryableStatus(settings.statusCode, settings.retryable),
		Cause:       settings.cause,
	}

	if body.GenerationID != nil {
		gatewayErr.GenerationID = *body.GenerationID
	}

	switch gatewayErr.Type {
	case ErrorAuthentication:
		gatewayErr.Message = authenticationMessage(settings.method)

	case ErrorModelNotFound:
		var param modelNotFoundParam
		if json.Unmarshal(body.Error.Param, &param) == nil {
			gatewayErr.ModelID = stringValue(param.ModelID)
		}

	case ErrorForbidden:
		var param forbiddenParam
		if json.Unmarshal(body.Error.Param, &param) == nil {
			gatewayErr.RuleID = stringValue(param.RuleID)
		}

	case ErrorInvalidRequest, ErrorRateLimit, ErrorNotFound, ErrorInternalServer, ErrorFailedDependency:

	default:
		gatewayErr.Type = ErrorInternalServer
	}

	return gatewayErr
}

// responseError reports an error body in an undocumented format.
func responseError(settings errorResponseSettings, generationID *string) *Error {
	gatewayErr := &Error{
		Type:        ErrorResponse,
		Message:     "Invalid error response format: Gateway request failed",
		StatusCode:  settings.statusCode,
		IsRetryable: retryableStatus(settings.statusCode, settings.retryable),
		Cause:       settings.cause,
	}

	if generationID != nil {
		gatewayErr.GenerationID = *generationID
	}

	if trimmed := bytes.TrimSpace(settings.body); json.Valid(trimmed) {
		gatewayErr.Response = trimmed
	} else if len(trimmed) > 0 {
		encoded, err := json.Marshal(string(trimmed))
		if err == nil {
			gatewayErr.Response = encoded
		}
	}

	return gatewayErr
}

// authenticationErrorSettings describe a failed authentication.
type authenticationErrorSettings struct {
	method     authMethod
	statusCode int
	cause      error
}

// authenticationError explains how to authenticate for the method used.
func authenticationError(settings authenticationErrorSettings) *Error {
	return &Error{
		Type:        ErrorAuthentication,
		Message:     authenticationMessage(settings.method),
		StatusCode:  settings.statusCode,
		IsRetryable: model.RetryableStatus(settings.statusCode),
		Cause:       settings.cause,
	}
}

func authenticationMessage(method authMethod) string {
	switch method {
	case authAPIKey:
		return apiKeyAuthenticationMessage
	case authOIDC:
		return oidcAuthenticationMessage
	default:
		return missingAuthenticationMessage
	}
}

func retryableStatus(statusCode int, override *bool) bool {
	if override != nil {
		return *override
	}

	return model.RetryableStatus(statusCode)
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}
