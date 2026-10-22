// Package devtools records AI SDK calls for local debugging, a port of the
// Vercel AI SDK DevTools. NewTelemetry captures GenerateText and StreamText
// calls through the ai.Telemetry lifecycle, NewMiddleware captures the
// language model calls of a wrapped model, and NewServer with
// Server.ListenAndServe runs a local viewer that reads the recordings.
//
// DevTools is intended for local development only. NewTelemetry and
// NewMiddleware return ErrProduction in production builds. Go adaptation:
// Go programs have no NODE_ENV convention, so a production build is marked
// explicitly at compile time with the production build tag:
//
//	go build -tags production ./cmd/app
//
// The guard cannot then be bypassed by the deployment environment. For
// parity with upstream, and for services deployed next to Node.js tooling,
// NODE_ENV=production is honored as well.
//
// Recordings are stored in the JSON file .devtools/generations.json of the
// project directory, in the same format as upstream, so the upstream viewer
// (npx @ai-sdk/devtools) can display them too. The file holds runs, one per
// multi-step interaction, and steps, one per language model call:
//
//	{
//	  "runs": [{"id", "started_at", "parent_run_id", "parent_step_id", "function_id"}],
//	  "steps": [{"id", "run_id", "step_number", "type", "model_id", "provider",
//	             "started_at", "duration_ms", "input", "output", "usage", "error",
//	             "raw_request", "raw_response", "raw_chunks", "provider_options"}]
//	}
//
// The step fields input, output, usage, raw_request, raw_response,
// raw_chunks and provider_options are JSON documents encoded as strings.
// After each change the recorder notifies a running viewer on
// localhost:4983, or on the port in AI_SDK_DEVTOOLS_PORT.
package devtools

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/internal/wirejson"
)

// DefaultPort is the port of the viewer when neither an option nor the
// AI_SDK_DEVTOOLS_PORT environment variable sets one.
const DefaultPort = 4983

// portEnvironmentVariable overrides the viewer port.
const portEnvironmentVariable = "AI_SDK_DEVTOOLS_PORT"

// ErrProduction reports that DevTools was configured in a production build:
// a binary built with the production build tag, or a process whose NODE_ENV
// environment variable is "production".
var ErrProduction = errors.New("devtools should not be used in production")

// nodeEnvironmentVariable is the variable upstream checks for production.
const nodeEnvironmentVariable = "NODE_ENV"

// requireDevelopment returns ErrProduction in a production build. Component
// names what to remove, such as "NewTelemetry".
func requireDevelopment(component string) error {
	if !isProduction() {
		return nil
	}

	return errors.Join(ErrProduction, errors.New("remove "+component+" from production builds"))
}

// isProduction reports whether the binary was built with the production
// build tag or NODE_ENV is "production".
func isProduction() bool {
	return isProductionBuild || os.Getenv(nodeEnvironmentVariable) == "production"
}

// resolvePort returns port when it is set, and otherwise the port of the
// AI_SDK_DEVTOOLS_PORT environment variable or DefaultPort.
func resolvePort(port int) int {
	if port > 0 {
		return port
	}

	configured, err := strconv.Atoi(os.Getenv(portEnvironmentVariable))
	if err != nil || configured <= 0 {
		return DefaultPort
	}

	return configured
}

// newRunID returns a run ID that starts with the UTC start time, so that run
// IDs sort chronologically, followed by a random suffix.
func newRunID(now time.Time) string {
	timestamp := strings.NewReplacer(
		"-", "",
		":", "",
		"T", "",
		".", "",
		"Z", "",
	).Replace(wirejson.Timestamp(now))

	return timestamp[:17] + "-" + id.New("")[:8]
}

// newStepID returns a random step ID.
func newStepID() string {
	return id.New("step")
}
