// Package bridgeadapter holds the host-side machinery shared by the
// bridge-backed harness adapters of this module (harness/claudecode,
// harness/codex, harness/opencode, harness/deepagents and harness/acp):
// spawning, launching or reattaching the in-sandbox bridge, routing its
// frames into prompt turns, and the session lifecycle. Upstream duplicates
// this code in each adapter; the adapters supply only their runtime-specific
// start commands, authentication and lifecycle data.
package bridgeadapter

import (
	"time"

	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// defaultStartupTimeout bounds the bridge startup when the connection
// settings leave it unset, as upstream.
const defaultStartupTimeout = 120 * time.Second

// startupTimeout returns the configured startup timeout or the default.
func startupTimeout(settings bridge.ConnectionSettings) time.Duration {
	if settings.StartupTimeout <= 0 {
		return defaultStartupTimeout
	}

	return settings.StartupTimeout
}

// Coordinates locate a live bridge, so that a later process can reattach to
// it instead of spawning a new one. They are stored in lifecycle state data
// under "bridge".
type Coordinates struct {
	Port            int    `json:"port"`
	Token           string `json:"token"`
	LastSeenEventID int64  `json:"lastSeenEventId"`
	SandboxID       string `json:"sandboxId,omitempty"`
}

// Strategy is how a session reached its bridge.
type Strategy string

// Strategies.
const (
	// StrategyFresh spawned a bridge for a new session.
	StrategyFresh Strategy = "fresh"

	// StrategyAttach reattached to a bridge that is still running.
	StrategyAttach Strategy = "attach"

	// StrategyReplay respawned the bridge, which replays the finished turn
	// from its event log on disk.
	StrategyReplay Strategy = "replay"

	// StrategyRerun respawned the bridge without a turn to replay; the
	// runtime continues its own thread from the persisted state.
	StrategyRerun Strategy = "rerun"
)

// IsRespawn reports whether the bridge was respawned for a resumed session.
func (s Strategy) IsRespawn() bool {
	return s == StrategyReplay || s == StrategyRerun
}
