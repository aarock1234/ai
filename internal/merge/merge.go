// Package merge combines provider-specific option and metadata objects. The
// values inside model.ProviderOptions and model.ProviderMetadata may be typed
// structs, maps or raw JSON, so objects are compared by their JSON form.
package merge

import (
	"encoding/json"
	"maps"
	"reflect"
)

// Deep returns base with overrides applied recursively, following the
// upstream mergeObjects rules: nested objects merge key by key, while arrays,
// scalars and nulls replace the base value. A nil override keeps the base
// value. Neither input is modified. The result is nil when both inputs are
// nil.
func Deep[Values ~map[string]any](base, overrides Values) Values {
	if overrides == nil {
		return base
	}

	if base == nil {
		return overrides
	}

	merged := make(Values, len(base)+len(overrides))
	maps.Copy(merged, base)

	for key, override := range overrides {
		if override == nil {
			continue
		}

		merged[key] = deepValue(merged[key], override)
	}

	return merged
}

// Shallow returns base with the top-level keys of overrides replacing those
// of base. Both values must encode to JSON objects; it reports false
// otherwise.
func Shallow(base, overrides any) (map[string]any, bool) {
	baseObject, ok := Object(base)
	if !ok {
		return nil, false
	}

	overrideObject, ok := Object(overrides)
	if !ok {
		return nil, false
	}

	merged := make(map[string]any, len(baseObject)+len(overrideObject))
	maps.Copy(merged, baseObject)
	maps.Copy(merged, overrideObject)

	return merged, true
}

// Object returns value as a JSON object. Maps with string keys are returned
// directly; structs and raw JSON are converted through their JSON encoding.
// It reports false for nil values and for values that are not objects.
func Object(value any) (map[string]any, bool) {
	switch typed := value.(type) {
	case nil:
		return nil, false
	case map[string]any:
		return typed, true
	case json.RawMessage:
		return decodeObject(typed)
	}

	if !mayEncodeObject(value) {
		return nil, false
	}

	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, false
	}

	return decodeObject(encoded)
}

func deepValue(base, override any) any {
	baseObject, baseIsObject := Object(base)
	overrideObject, overrideIsObject := Object(override)

	if !baseIsObject || !overrideIsObject {
		return override
	}

	return Deep(baseObject, overrideObject)
}

func decodeObject(encoded []byte) (map[string]any, bool) {
	var object map[string]any
	if err := json.Unmarshal(encoded, &object); err != nil || object == nil {
		return nil, false
	}

	return object, true
}

// mayEncodeObject reports whether value's kind can produce a JSON object, so
// that scalars and slices are not encoded only to be rejected.
func mayEncodeObject(value any) bool {
	kind := reflect.Indirect(reflect.ValueOf(value)).Kind()

	return kind == reflect.Struct || kind == reflect.Map
}
