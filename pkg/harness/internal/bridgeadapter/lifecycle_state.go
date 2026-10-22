package bridgeadapter

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/jsonvalue"
)

// DecodeStateData decodes the adapter-defined data of the lifecycle state a
// session continues or resumes from. A fresh session, or data that is not a
// JSON object, yields the zero data, as upstream reads only object data.
// Object data whose fields do not match T is reported, naming the adapter
// by displayName.
func DecodeStateData[T any](options harness.StartOptions, displayName string) (T, error) {
	var data T

	raw := LifecycleData(options)
	if !jsonvalue.IsObject(raw) {
		return data, nil
	}

	if err := json.Unmarshal(raw, &data); err != nil {
		var zero T

		return zero, fmt.Errorf("decode %s lifecycle state: %w", displayName, err)
	}

	return data, nil
}

// LifecycleData returns the data of the continuation or resume state, or
// nil for a fresh session.
func LifecycleData(options harness.StartOptions) json.RawMessage {
	if options.ContinueFrom != nil {
		return options.ContinueFrom.Data
	}

	if options.ResumeFrom != nil {
		return options.ResumeFrom.Data
	}

	return nil
}
