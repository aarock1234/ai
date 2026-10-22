package claudecode

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
    },
    "claudeSessionId": {
      "type": "string"
    }
  }
}`

// stateData is the adapter-defined data of a lifecycle state.
type stateData struct {
	// Bridge locates the live bridge after Detach and SuspendTurn.
	Bridge *bridgeadapter.Coordinates `json:"bridge,omitempty"`

	// SandboxCredentialEnvironment holds the credential placeholders of a
	// brokered session, reused on resume so that the installed request
	// transformations keep matching.
	SandboxCredentialEnvironment map[string]string `json:"sandboxCredentialEnvironment,omitempty"`

	// ClaudeSessionID is the exact Claude conversation to resume. It names
	// the conversation instead of relying on the SDK's continue flag, which
	// picks the most recent conversation of the working directory. It is
	// also reported as harnessMetadata["claude-code"].sessionId on finish
	// parts.
	ClaudeSessionID string `json:"claudeSessionId,omitempty"`
}

// stopStateData merges the bridge's stop reply with the adapter's own data.
// The reply takes precedence over the recorded conversation, which only
// fills in for replies without it.
func stopStateData(reply json.RawMessage, own stateData) (json.RawMessage, error) {
	merged := make(map[string]json.RawMessage)

	if own.ClaudeSessionID != "" {
		encoded, err := json.Marshal(own.ClaudeSessionID)
		if err != nil {
			return nil, err
		}

		merged["claudeSessionId"] = encoded
	}

	var replyFields map[string]json.RawMessage
	if err := json.Unmarshal(reply, &replyFields); err == nil {
		maps.Copy(merged, replyFields)
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
		return nil, fmt.Errorf("encode claude code lifecycle state: %w", err)
	}

	return data, nil
}
