package harness

import (
	"time"
)

// DebugLevel is the severity of a diagnostic.
type DebugLevel string

// Debug levels, from most to least severe.
const (
	DebugError DebugLevel = "error"
	DebugWarn  DebugLevel = "warn"
	DebugInfo  DebugLevel = "info"
	DebugDebug DebugLevel = "debug"
	DebugTrace DebugLevel = "trace"
)

// DebugConfig configures diagnostics. Applications set it on AgentSettings,
// where unset fields fall back to the HARNESS_DEBUG, HARNESS_DEBUG_LEVEL and
// HARNESS_DEBUG_SUBSYSTEMS environment variables; adapters receive the
// resolved configuration in Observability and send it to their bridge.
//
// Go adaptation: upstream has separate, identical emission (adapter) and
// consumption (host) types; this package uses one type for both.
type DebugConfig struct {
	// Enabled turns diagnostics on. Nil falls back to the environment.
	Enabled *bool `json:"enabled,omitempty"`

	// Level is the threshold; diagnostics at or above it are emitted. Empty
	// falls back to the environment, then DebugDebug.
	Level DebugLevel `json:"level,omitempty"`

	// Subsystems filters structured events by dotted prefix: "bridge"
	// matches "bridge.turn". Captured console lines are not filtered.
	Subsystems []string `json:"subsystems,omitempty"`
}

// DiagnosticKind distinguishes captured console lines from structured
// events.
type DiagnosticKind string

// Diagnostic kinds.
const (
	DiagnosticLog   DiagnosticKind = "log"
	DiagnosticEvent DiagnosticKind = "event"
)

// Diagnostic is one diagnostic an adapter reports, such as a console line
// captured inside the sandbox or a structured bridge event. Diagnostics stay
// per-line and are never folded into telemetry spans.
type Diagnostic struct {
	Level DebugLevel `json:"level"`

	// Message is the console line or the event message.
	Message string `json:"message"`

	// Subsystem is dotted; captured console lines use
	// "sandbox.log.<source>".
	Subsystem string `json:"subsystem"`

	Kind DiagnosticKind `json:"kind"`

	// Source and Stream describe a captured console line. Stream is "stdout"
	// or "stderr".
	Source string `json:"source,omitempty"`
	Stream string `json:"stream,omitempty"`

	// Attrs and Error belong to structured events.
	Attrs map[string]any   `json:"attrs,omitempty"`
	Error *DiagnosticError `json:"error,omitempty"`

	// SessionID identifies the harness session the diagnostic came from.
	SessionID string `json:"sessionId,omitempty"`

	Timestamp time.Time `json:"timestamp"`
}

// DiagnosticError is the error payload of a structured diagnostic.
type DiagnosticError struct {
	Name    string `json:"name,omitempty"`
	Message string `json:"message"`
	Stack   string `json:"stack,omitempty"`
}

// Observability is the diagnostics wiring the agent hands to Harness.Start
// when diagnostics are enabled.
type Observability struct {
	// Debug gates what the adapter captures and emits.
	Debug DebugConfig

	// Report receives each diagnostic. It is safe for concurrent use.
	Report func(diagnostic Diagnostic)
}
