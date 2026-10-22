package openaicompatible

import (
	"encoding/json"
	"errors"
	"fmt"
)

// errMissingChoices reports a chunk that is neither a delta nor an error.
var errMissingChoices = errors.New("chunk has no choices")

// streamOptions asks the provider to report usage in the stream.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// setStreamMembers enables streaming and, when includeUsage is set, usage
// reporting. These members replace passthrough options of the same name.
func setStreamMembers(body map[string]json.RawMessage, includeUsage bool) error {
	body["stream"] = json.RawMessage("true")
	delete(body, "stream_options")

	if !includeUsage {
		return nil
	}

	encoded, err := json.Marshal(streamOptions{
		IncludeUsage: true,
	})
	if err != nil {
		return fmt.Errorf("encode stream options: %w", err)
	}

	body["stream_options"] = encoded

	return nil
}
