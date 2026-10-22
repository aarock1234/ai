package opencode

import (
	"encoding/json"
	"fmt"
	"maps"

	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// lifecycleStateSchema is the JSON Schema of the adapter's lifecycle state
// data. It is a loose object: a stopped session's data holds whatever the
// bridge returned, while a detached or suspended session's data also carries
// the live bridge coordinates.
const lifecycleStateSchema = `{
  "type": "object",
  "properties": {
    "openCodeSessionId": {
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
	// OpenCodeSessionID is the OpenCode session to resume when the bridge
	// is respawned.
	OpenCodeSessionID string `json:"openCodeSessionId,omitempty"`

	// Bridge locates the live bridge after Detach and SuspendTurn.
	Bridge *bridgeadapter.Coordinates `json:"bridge,omitempty"`

	// SandboxCredentialEnvironment holds the credential placeholders of a
	// brokered session, reused on resume so that the installed request
	// transformations keep matching.
	SandboxCredentialEnvironment map[string]string `json:"sandboxCredentialEnvironment,omitempty"`
}

// stopStateData merges the bridge's stop reply, which carries the OpenCode
// session, with the credential placeholders.
func stopStateData(reply json.RawMessage, credentialEnvironment map[string]string) (json.RawMessage, error) {
	merged := make(map[string]json.RawMessage)

	var replyFields map[string]json.RawMessage
	if err := json.Unmarshal(reply, &replyFields); err == nil {
		maps.Copy(merged, replyFields)
	}

	if credentialEnvironment != nil {
		encoded, err := json.Marshal(credentialEnvironment)
		if err != nil {
			return nil, err
		}

		merged["sandboxCredentialEnvironment"] = encoded
	}

	data, err := json.Marshal(merged)
	if err != nil {
		return nil, fmt.Errorf("encode opencode lifecycle state: %w", err)
	}

	return data, nil
}
