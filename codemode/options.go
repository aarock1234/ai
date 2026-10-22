package codemode

import (
	"context"
	"encoding/json"
	"time"
)

// Default execution limits, matching upstream's defaults.
const (
	DefaultTimeout                   = 30 * time.Second
	DefaultMaxCallStackDepth         = 1024
	DefaultMaxResultBytes            = 1024 * 1024
	DefaultMaxConsoleOutputBytes     = 64 * 1024
	DefaultMaxSourceBytes            = 256 * 1024
	DefaultMaxToolInputBytes         = 1024 * 1024
	DefaultMaxToolOutputBytes        = 4 * 1024 * 1024
	DefaultMaxBridgeRequests         = 256
	DefaultMaxInFlightBridgeRequests = 32
)

// Options configure NewTool, NewCallerTool and Run.
type Options struct {
	ExecutionPolicy      ExecutionPolicy
	ContinuationSecurity ContinuationSecurity
	Approval             ApprovalOptions

	// OnConsole receives each console message the program writes, within
	// ExecutionPolicy.MaxConsoleOutputBytes. It runs on the goroutine that
	// executes the program, so it must not block for long.
	OnConsole func(ctx context.Context, entry ConsoleEntry)
}

// ExecutionPolicy limits one sandbox invocation. A zero field uses its
// default.
//
// Upstream's memoryLimitBytes has no equivalent: the Go interpreter cannot
// bound the heap of one program. Upstream's maxStackSizeBytes becomes a call
// depth, which is what the interpreter limits.
type ExecutionPolicy struct {
	// Timeout bounds the whole invocation, including host tool calls. The
	// default is DefaultTimeout.
	Timeout time.Duration

	// MaxCallStackDepth bounds the JavaScript call depth. The default is
	// DefaultMaxCallStackDepth.
	MaxCallStackDepth int

	// MaxResultBytes bounds the JSON encoding of the returned value.
	MaxResultBytes int

	// MaxConsoleOutputBytes bounds the console output that is kept. Later
	// output is dropped and the result is marked as truncated.
	MaxConsoleOutputBytes int

	// MaxSourceBytes bounds the program source.
	MaxSourceBytes int

	// MaxToolInputBytes bounds the JSON encoding of each host tool input.
	MaxToolInputBytes int

	// MaxToolOutputBytes bounds the JSON encoding of each host tool output
	// and of each interrupt resolution.
	MaxToolOutputBytes int

	// MaxBridgeRequests bounds the number of host tool calls.
	MaxBridgeRequests int

	// MaxInFlightBridgeRequests bounds the number of host tool calls that
	// run at the same time.
	MaxInFlightBridgeRequests int
}

// ApprovalMode selects how tools that need approval are handled.
type ApprovalMode string

// Approval modes. The empty mode means ApprovalModeCallback.
const (
	// ApprovalModeCallback asks ApprovalOptions.OnApprovalRequired and fails
	// the call when it is nil.
	ApprovalModeCallback ApprovalMode = "callback"

	// ApprovalModeInterrupt stops the invocation with an approval Interrupt
	// that ContinueApproval resumes.
	ApprovalModeInterrupt ApprovalMode = "interrupt"
)

// ApprovalOptions configure host tools whose NeedsApproval function reports
// that a call needs approval.
type ApprovalOptions struct {
	Mode ApprovalMode

	// OnApprovalRequired decides a call in ApprovalModeCallback. It may be
	// called concurrently for calls that run at the same time.
	OnApprovalRequired func(ctx context.Context, request ApprovalRequest) (ApprovalDecision, error)
}

// ApprovalRequest describes a host tool call awaiting approval.
type ApprovalRequest struct {
	ToolName   string
	Input      json.RawMessage
	ToolCallID string
}

// ApprovalDecision approves or denies a host tool call. It is also the JSON
// resolution of an approval interrupt.
type ApprovalDecision struct {
	Approved bool `json:"approved"`

	// Reason optionally explains a denial.
	Reason string `json:"reason,omitempty"`
}

// ConsoleLevel is the console method a message was written with.
type ConsoleLevel string

// Console levels.
const (
	ConsoleLevelLog   ConsoleLevel = "log"
	ConsoleLevelInfo  ConsoleLevel = "info"
	ConsoleLevelWarn  ConsoleLevel = "warn"
	ConsoleLevelError ConsoleLevel = "error"
	ConsoleLevelDebug ConsoleLevel = "debug"
)

// ConsoleEntry is one console message. Arguments are joined with spaces;
// strings are written as they are and other values as JSON.
type ConsoleEntry struct {
	Level   ConsoleLevel
	Message string
}

// executionLimits is an ExecutionPolicy with the defaults applied.
type executionLimits struct {
	timeout                   time.Duration
	maxCallStackDepth         int
	maxResultBytes            int
	maxConsoleOutputBytes     int
	maxSourceBytes            int
	maxToolInputBytes         int
	maxToolOutputBytes        int
	maxBridgeRequests         int
	maxInFlightBridgeRequests int
}

func resolveExecutionLimits(policy ExecutionPolicy) executionLimits {
	return executionLimits{
		timeout:                   positiveOr(policy.Timeout, DefaultTimeout),
		maxCallStackDepth:         positiveOr(policy.MaxCallStackDepth, DefaultMaxCallStackDepth),
		maxResultBytes:            positiveOr(policy.MaxResultBytes, DefaultMaxResultBytes),
		maxConsoleOutputBytes:     positiveOr(policy.MaxConsoleOutputBytes, DefaultMaxConsoleOutputBytes),
		maxSourceBytes:            positiveOr(policy.MaxSourceBytes, DefaultMaxSourceBytes),
		maxToolInputBytes:         positiveOr(policy.MaxToolInputBytes, DefaultMaxToolInputBytes),
		maxToolOutputBytes:        positiveOr(policy.MaxToolOutputBytes, DefaultMaxToolOutputBytes),
		maxBridgeRequests:         positiveOr(policy.MaxBridgeRequests, DefaultMaxBridgeRequests),
		maxInFlightBridgeRequests: positiveOr(policy.MaxInFlightBridgeRequests, DefaultMaxInFlightBridgeRequests),
	}
}

func positiveOr[T int | time.Duration](value, fallback T) T {
	if value > 0 {
		return value
	}

	return fallback
}
