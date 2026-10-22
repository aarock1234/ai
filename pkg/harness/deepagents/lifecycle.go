package deepagents

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
)

// lifecycleStateSchema is the JSON Schema of the adapter's lifecycle state
// data.
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
}

// isResumeStart reports whether the session starts from lifecycle state.
func isResumeStart(options harness.StartOptions) bool {
	return options.ContinueFrom != nil || options.ResumeFrom != nil
}

// encodeStateData encodes lifecycle state data.
func encodeStateData(data stateData) (json.RawMessage, error) {
	encoded, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("encode deepagents lifecycle state: %w", err)
	}

	return encoded, nil
}
