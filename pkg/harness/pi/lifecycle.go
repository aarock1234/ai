package pi

import (
	"encoding/json"
	"fmt"
)

// lifecycleStateSchema is the JSON Schema of the adapter's lifecycle state
// data. sessionFileName is upstream's field; runtime locates a Pi process
// that keeps running in the sandbox.
const lifecycleStateSchema = `{
  "type": "object",
  "properties": {
    "sessionFileName": {
      "type": "string",
      "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]*\\.jsonl?$"
    },
    "runtime": {
      "type": "object",
      "properties": {
        "launchId": {
          "type": "string",
          "pattern": "^[a-f0-9]{32}$"
        },
        "outboxOffset": {
          "type": "integer",
          "minimum": 0
        },
        "inputSequence": {
          "type": "integer",
          "minimum": 0
        },
        "configurationFingerprint": {
          "type": "string"
        },
        "turnActive": {
          "type": "boolean"
        },
        "pendingDialogs": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "requestId": {
                "type": "string"
              },
              "kind": {
                "type": "string"
              },
              "toolCallId": {
                "type": "string"
              }
            },
            "required": ["requestId", "kind", "toolCallId"]
          }
        },
        "translator": {
          "type": "object"
        },
        "turnBaseline": {
          "type": "object"
        },
        "toolsSignature": {
          "type": "string"
        },
        "model": {
          "type": "object"
        },
        "modelSelected": {
          "type": "boolean"
        }
      },
      "required": ["launchId", "outboxOffset", "inputSequence", "configurationFingerprint"]
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
	// SessionFileName is the basename of Pi's session file in the session's
	// private directory, which a cold restart opens.
	SessionFileName string `json:"sessionFileName,omitempty"`

	// Runtime locates the live Pi process after Detach and SuspendTurn.
	Runtime *liveRuntime `json:"runtime,omitempty"`

	// SandboxCredentialEnvironment holds the credential placeholders of a
	// brokered session, reused on resume so that the installed request
	// transformations keep matching.
	SandboxCredentialEnvironment map[string]string `json:"sandboxCredentialEnvironment,omitempty"`
}

// liveRuntime is the cursor of a Pi process that keeps running, with the
// host state needed to continue its turn.
type liveRuntime struct {
	LaunchID      string `json:"launchId"`
	OutboxOffset  int64  `json:"outboxOffset"`
	InputSequence int64  `json:"inputSequence"`

	// ConfigurationFingerprint identifies the launch configuration. A resume
	// with other settings restarts Pi from its session file.
	ConfigurationFingerprint string `json:"configurationFingerprint"`

	// TurnActive reports a turn that was running when the host froze.
	TurnActive bool `json:"turnActive,omitempty"`

	// PendingDialogs are the host-input requests Pi is waiting on.
	PendingDialogs []pendingDialog `json:"pendingDialogs,omitempty"`

	Translator   *translator    `json:"translator,omitempty"`
	TurnBaseline *sessionTokens `json:"turnBaseline,omitempty"`

	ToolsSignature string     `json:"toolsSignature,omitempty"`
	Model          *modelInfo `json:"model,omitempty"`
	ModelSelected  bool       `json:"modelSelected,omitempty"`
}

// pendingDialog is an extension UI dialog waiting for host input.
type pendingDialog struct {
	RequestID  string `json:"requestId"`
	Kind       string `json:"kind"`
	ToolCallID string `json:"toolCallId"`
}

// validate checks the identifiers that become sandbox paths.
func (d stateData) validate() error {
	if d.SessionFileName != "" && !isSafeSessionFileName(d.SessionFileName) {
		return fmt.Errorf("invalid pi session file name: %s", d.SessionFileName)
	}

	if d.Runtime != nil && !launchIDPattern.MatchString(d.Runtime.LaunchID) {
		return fmt.Errorf("invalid pi runtime launch id: %s", d.Runtime.LaunchID)
	}

	return nil
}

// encodeState encodes lifecycle state data.
func encodeState(data stateData) (json.RawMessage, error) {
	encoded, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("encode pi lifecycle state: %w", err)
	}

	return encoded, nil
}
