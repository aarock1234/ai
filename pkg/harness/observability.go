package harness

import (
	"fmt"
	"io"
	"os"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
)

// DiagnosticConsumer is implemented by telemetry integrations that also
// want the per-line diagnostics stream. The agent calls IngestDiagnostic
// for every forwarded diagnostic in addition to the ai.Telemetry events, so
// a single reporter registered in AgentSettings.Telemetry receives both.
type DiagnosticConsumer interface {
	IngestDiagnostic(diagnostic Diagnostic)
}

// resolvedDebugConfig is a debug configuration with its defaults applied.
type resolvedDebugConfig struct {
	enabled    bool
	level      DebugLevel
	subsystems []string
}

// resolveDebugConfig applies the HARNESS_DEBUG, HARNESS_DEBUG_LEVEL and
// HARNESS_DEBUG_SUBSYSTEMS environment variables to the fields the
// configuration leaves unset. Code is always authoritative.
func resolveDebugConfig(config *DebugConfig, getenv func(key string) string) resolvedDebugConfig {
	resolved := resolvedDebugConfig{
		enabled:    isTruthy(getenv("HARNESS_DEBUG")),
		level:      DebugLevel(getenv("HARNESS_DEBUG_LEVEL")),
		subsystems: parseList(getenv("HARNESS_DEBUG_SUBSYSTEMS")),
	}

	if config != nil {
		if config.Enabled != nil {
			resolved.enabled = *config.Enabled
		}

		if config.Level != "" {
			resolved.level = config.Level
		}

		if config.Subsystems != nil {
			resolved.subsystems = slices.Clone(config.Subsystems)
		}
	}

	if resolved.level == "" {
		resolved.level = DebugDebug
	}

	return resolved
}

// observabilitySetup are the agent settings diagnostics depend on.
type observabilitySetup struct {
	debug     *DebugConfig
	onLog     func(diagnostic Diagnostic)
	telemetry ai.TelemetryOptions
	stderr    io.Writer
}

// buildObservability returns the diagnostics wiring handed to Harness.Start,
// or nil when diagnostics are disabled. Report fans each diagnostic out to
// stderr, OnLog and the telemetry integrations that consume diagnostics.
func buildObservability(setup observabilitySetup) *Observability {
	resolved := resolveDebugConfig(setup.debug, os.Getenv)
	if !resolved.enabled {
		return nil
	}

	var consumers []DiagnosticConsumer
	for _, integration := range setup.telemetry.Integrations {
		if consumer, ok := integration.(DiagnosticConsumer); ok {
			consumers = append(consumers, consumer)
		}
	}

	stderr := setup.stderr
	if stderr == nil {
		stderr = os.Stderr
	}

	enabled := true

	return &Observability{
		Debug: DebugConfig{
			Enabled:    &enabled,
			Level:      resolved.level,
			Subsystems: resolved.subsystems,
		},
		Report: func(diagnostic Diagnostic) {
			_, _ = io.WriteString(stderr, formatDiagnostic(diagnostic))

			if setup.onLog != nil {
				setup.onLog(diagnostic)
			}

			for _, consumer := range consumers {
				consumer.IngestDiagnostic(diagnostic)
			}
		},
	}
}

// formatDiagnostic renders a diagnostic as one stderr line.
func formatDiagnostic(diagnostic Diagnostic) string {
	parts := []string{
		fmt.Sprintf("[harness:%s]", diagnostic.Level),
	}

	if diagnostic.Subsystem != "" {
		parts = append(parts, diagnostic.Subsystem)
	}

	if diagnostic.Message != "" {
		parts = append(parts, diagnostic.Message)
	}

	line := strings.Join(parts, " ")

	if diagnostic.Error != nil {
		name := diagnostic.Error.Name
		if name == "" {
			name = "Error"
		}

		line += fmt.Sprintf(" (%s: %s)", name, diagnostic.Error.Message)
	}

	return line + "\n"
}

// isTruthy reports whether an environment value enables a flag.
func isTruthy(value string) bool {
	switch strings.ToLower(value) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// parseList splits a comma-separated list, dropping empty items. It returns
// nil for an empty list.
func parseList(value string) []string {
	var items []string

	for item := range strings.SplitSeq(value, ",") {
		if trimmed := strings.TrimSpace(item); trimmed != "" {
			items = append(items, trimmed)
		}
	}

	return items
}
