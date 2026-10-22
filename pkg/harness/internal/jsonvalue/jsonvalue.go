// Package jsonvalue classifies the opaque JSON values the harness packages
// exchange with runtimes and lifecycle state.
package jsonvalue

import "encoding/json"

// IsObject reports whether raw encodes a JSON object.
func IsObject(raw json.RawMessage) bool {
	var object map[string]json.RawMessage

	return json.Unmarshal(raw, &object) == nil && object != nil
}
