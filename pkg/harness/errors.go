package harness

import (
	"errors"
)

// ErrHarness is wrapped by every error type of this package, the Go form of
// the upstream HarnessError base class: errors.Is(err, ErrHarness) reports
// whether a failure originated in or was signalled by a harness.
var ErrHarness = errors.New("harness error")

// defaultClientErrorMessage replaces error details sent to clients.
const defaultClientErrorMessage = "An error occurred."

// Error is a general harness failure.
type Error struct {
	Message string
	Cause   error
}

// Error returns the message.
func (e *Error) Error() string {
	return e.Message
}

// Unwrap returns ErrHarness and the cause.
func (e *Error) Unwrap() []error {
	return harnessCauses(e.Cause)
}

// CapabilityUnsupportedError reports a request the adapter or the sandbox
// cannot satisfy, such as manual compaction from an adapter that only
// compacts automatically, or a port endpoint from a sandbox without ports.
type CapabilityUnsupportedError struct {
	// Message is the complete human-readable message.
	Message string

	// HarnessID optionally identifies the adapter or sandbox provider.
	HarnessID string

	Cause error
}

// Error returns the message.
func (e *CapabilityUnsupportedError) Error() string {
	return e.Message
}

// Unwrap returns ErrHarness and the cause.
func (e *CapabilityUnsupportedError) Unwrap() []error {
	return harnessCauses(e.Cause)
}

// HistoryUnavailableError reports an adapter that supports history reads but
// cannot reach the runtime's store from the current environment, for
// example because the store lives inside a remote sandbox. A conversation
// without recorded messages is not an error.
type HistoryUnavailableError struct {
	Message   string
	HarnessID string
	Cause     error
}

// Error returns the message.
func (e *HistoryUnavailableError) Error() string {
	return e.Message
}

// Unwrap returns ErrHarness and the cause.
func (e *HistoryUnavailableError) Unwrap() []error {
	return harnessCauses(e.Cause)
}

// SandboxAuthenticationError reports a sandbox provider that cannot
// authenticate or authorize the creation or resumption of a sandbox. The
// message explains how to configure credentials; Cause keeps the underlying
// failure.
type SandboxAuthenticationError struct {
	Message           string
	SandboxProviderID string
	Cause             error
}

// Error returns the message.
func (e *SandboxAuthenticationError) Error() string {
	return e.Message
}

// Unwrap returns ErrHarness and the cause.
func (e *SandboxAuthenticationError) Unwrap() []error {
	return harnessCauses(e.Cause)
}

// ErrorMessage returns a client-safe message for an error produced by a
// harness turn. The messages of CapabilityUnsupportedError,
// SandboxAuthenticationError and Error are preserved; every other error is
// masked so that provider, sandbox and server details are not exposed.
func ErrorMessage(err error) string {
	if capability, ok := errors.AsType[*CapabilityUnsupportedError](err); ok {
		return capability.Message
	}

	if authentication, ok := errors.AsType[*SandboxAuthenticationError](err); ok {
		return authentication.Message
	}

	if general, ok := errors.AsType[*Error](err); ok {
		return general.Message
	}

	return defaultClientErrorMessage
}

// harnessCauses returns the errors a harness error wraps.
func harnessCauses(cause error) []error {
	if cause == nil {
		return []error{
			ErrHarness,
		}
	}

	return []error{
		ErrHarness,
		cause,
	}
}
