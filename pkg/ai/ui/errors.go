package ui

import (
	"fmt"
)

// MessageStreamError reports a chunk that does not fit the message being
// built, such as a text delta without a preceding text start.
type MessageStreamError struct {
	// ChunkType is the type of the offending chunk.
	ChunkType string

	// ChunkID is the ID, tool call ID or approval ID of the chunk.
	ChunkID string

	Message string
}

// Error returns the description of the mismatch.
func (e *MessageStreamError) Error() string {
	return fmt.Sprintf("ui message stream %s chunk %q: %s", e.ChunkType, e.ChunkID, e.Message)
}

// ValidationError reports a UI message value that does not match its
// schema or the expected message structure.
type ValidationError struct {
	// Field locates the value, such as "messages[0].parts[1].input".
	Field string

	// EntityName and EntityID name the tool or data part and the ID of the
	// tool call, part or message, when known.
	EntityName string
	EntityID   string

	Cause error
}

// Error returns the location and the cause.
func (e *ValidationError) Error() string {
	return fmt.Sprintf("invalid %s: %v", e.Field, e.Cause)
}

// Unwrap returns the cause.
func (e *ValidationError) Unwrap() error {
	return e.Cause
}
