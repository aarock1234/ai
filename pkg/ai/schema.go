package ai

import (
	"encoding/json"

	"github.com/aarock1234/ai/internal/schema"
)

// SchemaFor generates a JSON Schema from Go type T via reflection.
// Use this to build the InputSchema field of a Tool definition.
//
// This is a top-level generic function because Go does not support
// generic methods.
func SchemaFor[T any]() json.RawMessage {
	return schema.Generate[T]()
}
