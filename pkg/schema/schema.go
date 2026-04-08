// Package schema provides JSON Schema generation from Go types via
// reflection. It supports struct fields with json tags, basic types,
// slices, nested structs, pointers (nullable), and a jsonschema
// struct tag for descriptions and enums.
package schema

import (
	"encoding/json"
	"reflect"
	"strings"
	"time"
)

var timeType = reflect.TypeFor[time.Time]()

// Generate produces a JSON Schema for type T. It examines struct
// fields, json tags, and optional jsonschema tags to build the schema.
//
// Supported types: string, int*, float*, bool, time.Time, slices,
// nested structs, pointers (nullable).
//
// The jsonschema struct tag supports:
//
//	`jsonschema:"description=text"`
//	`jsonschema:"enum=a|b|c"`
//	`jsonschema:"description=text,enum=a|b|c"`
//
// This is a top-level generic function because Go does not support
// generic methods.
func Generate[T any]() json.RawMessage {
	t := reflect.TypeFor[T]()

	schema := generateType(t)
	data, _ := json.Marshal(schema)

	return data
}

// schemaMap is a JSON Schema object represented as an ordered map.
type schemaMap = map[string]any

// generateType produces a schema for a single Go type.
func generateType(t reflect.Type) schemaMap {
	// Dereference pointer.
	if t.Kind() == reflect.Pointer {
		inner := generateType(t.Elem())
		inner["nullable"] = true

		return inner
	}

	// Special-case time.Time.
	if t == timeType {
		return schemaMap{
			"type":   "string",
			"format": "date-time",
		}
	}

	switch t.Kind() {
	case reflect.String:
		return schemaMap{"type": "string"}

	case reflect.Bool:
		return schemaMap{"type": "boolean"}

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return schemaMap{"type": "integer"}

	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return schemaMap{"type": "integer"}

	case reflect.Float32, reflect.Float64:
		return schemaMap{"type": "number"}

	case reflect.Slice:
		return schemaMap{
			"type":  "array",
			"items": generateType(t.Elem()),
		}

	case reflect.Struct:
		return generateStruct(t)

	default:
		return schemaMap{}
	}
}

// generateStruct produces a JSON Schema object for a struct type.
func generateStruct(t reflect.Type) schemaMap {
	properties := schemaMap{}
	var required []string

	for i := range t.NumField() {
		field := t.Field(i)

		// Skip unexported fields.
		if !field.IsExported() {
			continue
		}

		// Determine JSON field name.
		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}

		name, opts, _ := strings.Cut(jsonTag, ",")
		if name == "" {
			name = field.Name
		}

		// Check if omitempty/omitzero.
		optional := strings.Contains(opts, "omitempty") || strings.Contains(opts, "omitzero")

		fieldSchema := generateType(field.Type)

		// Apply jsonschema tag.
		jsTag := field.Tag.Get("jsonschema")
		if jsTag != "" {
			applyJSONSchemaTag(fieldSchema, jsTag)
		}

		properties[name] = fieldSchema

		if !optional && field.Type.Kind() != reflect.Pointer {
			required = append(required, name)
		}
	}

	schema := schemaMap{
		"type":                 "object",
		"properties":           properties,
		"additionalProperties": false,
	}

	if len(required) > 0 {
		schema["required"] = required
	}

	return schema
}

// applyJSONSchemaTag parses the jsonschema struct tag and applies
// description and enum values to the schema.
func applyJSONSchemaTag(schema schemaMap, tag string) {
	for _, part := range strings.Split(tag, ",") {
		part = strings.TrimSpace(part)

		if strings.HasPrefix(part, "description=") {
			schema["description"] = strings.TrimPrefix(part, "description=")
		}

		if strings.HasPrefix(part, "enum=") {
			values := strings.Split(strings.TrimPrefix(part, "enum="), "|")
			schema["enum"] = values
		}
	}
}
