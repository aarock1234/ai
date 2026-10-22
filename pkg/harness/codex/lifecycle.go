package codex

import (
	"encoding/json"
	"fmt"
	"maps"

	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// lifecycleStateSchema is the JSON Schema of the adapter's lifecycle state
// data.
const lifecycleStateSchema = `{
  "type": "object",
  "properties": {
    "threadId": {
      "type": "string"
    },
    "turnConfigurationFingerprint": {
      "type": "string"
    },
    "bridge": {
      "type": "object",
      "properties": {
        "port": {
          "type": "number"
        },
        "token": {
          "type": "string"
        },
        "lastSeenEventId": {
          "type": "number"
        },
        "sandboxId": {
          "type": "string"
        }
      },
      "required": ["port", "token", "lastSeenEventId"]
    },
    "sandboxCredentialEnvironment": {
      "type": "object",
      "additionalProperties": {
        "type": "string"
      }
    }
  }
}`

// stateData is the adapter-defined data of a lifecycle state.
type stateData struct {
	// ThreadID is the app-server thread to resume with thread/resume when the
	// bridge is respawned.
	ThreadID string `json:"threadId,omitempty"`

	// TurnConfigurationFingerprint identifies the instructions and tools the
	// thread was started with, so that a changed configuration starts a new
	// thread.
	TurnConfigurationFingerprint string `json:"turnConfigurationFingerprint,omitempty"`

	// Bridge locates the live bridge after Detach and SuspendTurn.
	Bridge *bridgeadapter.Coordinates `json:"bridge,omitempty"`

	// SandboxCredentialEnvironment holds the credential placeholders of a
	// brokered session, reused on resume so that the installed request
	// transformations keep matching.
	SandboxCredentialEnvironment map[string]string `json:"sandboxCredentialEnvironment,omitempty"`
}

// stopStateData merges the bridge's stop reply, which carries the thread ID,
// with the adapter's own data. A bridge that could not reply loses the
// thread, so Codex starts a new thread on the preserved working directory.
func stopStateData(reply json.RawMessage, own stateData) (json.RawMessage, error) {
	merged := make(map[string]json.RawMessage)

	var replyFields map[string]json.RawMessage
	if err := json.Unmarshal(reply, &replyFields); err == nil {
		maps.Copy(merged, replyFields)
	}

	if own.TurnConfigurationFingerprint != "" {
		encoded, err := json.Marshal(own.TurnConfigurationFingerprint)
		if err != nil {
			return nil, err
		}

		merged["turnConfigurationFingerprint"] = encoded
	}

	if own.SandboxCredentialEnvironment != nil {
		encoded, err := json.Marshal(own.SandboxCredentialEnvironment)
		if err != nil {
			return nil, err
		}

		merged["sandboxCredentialEnvironment"] = encoded
	}

	data, err := json.Marshal(merged)
	if err != nil {
		return nil, fmt.Errorf("encode codex lifecycle state: %w", err)
	}

	return data, nil
}
