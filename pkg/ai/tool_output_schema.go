package ai

import (
	"encoding/json"
	"reflect"

	"github.com/aarock1234/ai/internal/schema"
)

// generateOutputSchema returns the JSON Schema of a tool output type, or nil
// when the type is an interface or its schema cannot be inferred. Outputs
// are encoded by encoding/json, which leaves out omitempty fields and writes
// nil slices and maps as null.
func generateOutputSchema[Output any]() json.RawMessage {
	if reflect.TypeFor[Output]().Kind() == reflect.Interface {
		return nil
	}

	outputSchema, err := schema.GenerateWithOptions[Output](schema.GenerateOptions{
		Writer: schema.ProgramWriter,
	})
	if err != nil {
		return nil
	}

	return outputSchema
}
