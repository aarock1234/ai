package codemode

import (
	"encoding/json"
	"fmt"
	"time"
)

// ErrorCode is the stable machine-readable code of an Error.
type ErrorCode string

// Error codes, matching upstream's codes.
const (
	// ErrorCodeGeneric is a code mode failure without a more specific code.
	ErrorCodeGeneric ErrorCode = "CODE_MODE_ERROR"

	// ErrorCodeTimeout reports that the invocation exceeded its timeout.
	ErrorCodeTimeout ErrorCode = "CODE_MODE_TIMEOUT"

	// ErrorCodeAborted reports that the caller's context ended the
	// invocation.
	ErrorCodeAborted ErrorCode = "CODE_MODE_ABORTED"

	// ErrorCodeConcurrency reports that the process-wide worker cap set by
	// SetMaxWorkers was reached.
	ErrorCodeConcurrency ErrorCode = "CODE_MODE_CONCURRENCY_LIMIT"

	// ErrorCodeSourceTooLarge reports a program larger than
	// ExecutionPolicy.MaxSourceBytes.
	ErrorCodeSourceTooLarge ErrorCode = "CODE_MODE_SOURCE_TOO_LARGE"

	// ErrorCodeBridgeLimit reports a program that exceeded the host tool
	// call limits.
	ErrorCodeBridgeLimit ErrorCode = "CODE_MODE_BRIDGE_LIMIT"

	// ErrorCodeDetachedBridgeRequest reports a program that did not await a
	// host tool call it made, or returned while one was still running.
	ErrorCodeDetachedBridgeRequest ErrorCode = "CODE_MODE_DETACHED_BRIDGE_REQUEST"

	// ErrorCodeProtocol reports a malformed or mismatched continuation,
	// interrupt or resolution.
	ErrorCodeProtocol ErrorCode = "CODE_MODE_PROTOCOL_ERROR"

	// ErrorCodeSerialization reports a value that is not JSON-serializable
	// or exceeds its size limit.
	ErrorCodeSerialization ErrorCode = "CODE_MODE_SERIALIZATION_ERROR"

	// ErrorCodeTool reports a host tool call that could not be made, such
	// as an unknown tool or invalid input.
	ErrorCodeTool ErrorCode = "CODE_MODE_TOOL_ERROR"

	// ErrorCodeToolApprovalRequired reports a call that needs approval when
	// no approval callback is configured.
	ErrorCodeToolApprovalRequired ErrorCode = "CODE_MODE_TOOL_APPROVAL_REQUIRED"

	// ErrorCodeToolApprovalDenied reports a call whose approval was denied.
	ErrorCodeToolApprovalDenied ErrorCode = "CODE_MODE_TOOL_APPROVAL_DENIED"

	// ErrorCodeHostTool reports a host tool whose Execute function failed.
	// The program only sees a generic message; Cause holds the failure.
	ErrorCodeHostTool ErrorCode = "CODE_MODE_HOST_TOOL_ERROR"
)

// Error is a code mode failure. The program sees Message and Code when a
// host tool call fails; the Go caller can inspect every field with
// errors.AsType.
type Error struct {
	Code ErrorCode

	// Message describes the failure. It is also the message of the
	// JavaScript error that the program receives.
	Message string

	// ToolName, ToolCallID and Input identify the host tool call, for
	// tool errors.
	ToolName   string
	ToolCallID string
	Input      json.RawMessage

	// Reason explains a denied approval.
	Reason string

	// Limit and Size report the exceeded limit and the observed size, in
	// bytes, calls, workers or milliseconds, for limit errors.
	Limit int64
	Size  int64

	// Cause is the underlying failure, such as the error a host tool
	// returned or the context's cause.
	Cause error
}

// Error implements error.
func (e *Error) Error() string {
	return "code mode: " + e.Message
}

// Unwrap returns the cause.
func (e *Error) Unwrap() error {
	return e.Cause
}

// ScriptError is an exception that the program threw and did not catch,
// including syntax errors.
type ScriptError struct {
	// Name is the error name, such as "TypeError" or "SyntaxError".
	Name    string
	Message string

	// Stack is the JavaScript stack trace, when the exception has one.
	// Program locations are reported as code-mode.js.
	Stack string
}

// Error implements error.
func (e *ScriptError) Error() string {
	if e.Stack != "" {
		return "code mode program failed: " + e.Stack
	}

	if e.Name == "" {
		return "code mode program failed: " + e.Message
	}

	return fmt.Sprintf("code mode program failed: %s: %s", e.Name, e.Message)
}

func newTimeoutError(timeout time.Duration) *Error {
	return &Error{
		Code:    ErrorCodeTimeout,
		Message: fmt.Sprintf("execution timed out after %s", timeout),
		Limit:   timeout.Milliseconds(),
	}
}

func newAbortedError(cause error) *Error {
	return &Error{
		Code:    ErrorCodeAborted,
		Message: "execution was aborted",
		Cause:   cause,
	}
}

func newProtocolError(message string) *Error {
	return &Error{
		Code:    ErrorCodeProtocol,
		Message: message,
	}
}

func newBridgeLimitError(message string, limit int) *Error {
	return &Error{
		Code:    ErrorCodeBridgeLimit,
		Message: message,
		Limit:   int64(limit),
	}
}

func newApprovalDeniedError(call PendingInterruption, reason string) *Error {
	message := fmt.Sprintf("tool %q approval was denied", call.ToolName)
	if reason != "" {
		message += ": " + reason
	}

	return &Error{
		Code:       ErrorCodeToolApprovalDenied,
		Message:    message,
		ToolName:   call.ToolName,
		ToolCallID: call.ToolCallID,
		Input:      call.Input,
		Reason:     reason,
	}
}
