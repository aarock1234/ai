package ai

import (
	"encoding/json"
	"reflect"

	"github.com/aarock1234/ai/internal/schema"
)

// generateOutputSchema returns the JSON Schema of a tool output type, or nil
// when the type is an interface or its schema cannot be inferred. Outputs
// are encoded by encoding/json, which writes nil slices and maps as null,
// so the schema accepts null for them.
func generateOutputSchema[Output any]() json.RawMessage {
	if reflect.TypeFor[Output]().Kind() == reflect.Interface {
		return nil
	}

	outputSchema, err := schema.GenerateWithOptions[Output](schema.GenerateOptions{
		NullableCollections: true,
	})
	if err != nil {
		return nil
	}

	return outputSchema
}
