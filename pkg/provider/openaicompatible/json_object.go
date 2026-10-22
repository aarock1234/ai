package openaicompatible

import (
	"encoding/json"
	"maps"

	"github.com/aarock1234/ai/internal/jsonobject"
)

// encodeExtended encodes value as a JSON object and then sets each member of
// extra, replacing members of the same name. Prompt messages and parts use it
// to carry caller-supplied provider extensions.
func encodeExtended(value any, extra map[string]json.RawMessage) ([]byte, error) {
	if len(extra) == 0 {
		return json.Marshal(value)
	}

	members, err := jsonobject.Members(value)
	if err != nil {
		return nil, err
	}

	maps.Copy(members, extra)

	return json.Marshal(members)
}
