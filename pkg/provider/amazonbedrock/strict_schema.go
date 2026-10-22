package amazonbedrock

import (
	"bytes"
	"encoding/json"
	"slices"
)

// strictSchemaNode holds the JSON Schema keywords that can nest schemas.
type strictSchemaNode struct {
	Type                 json.RawMessage `json:"type"`
	AdditionalProperties json.RawMessage `json:"additionalProperties"`

	Properties        map[string]json.RawMessage `json:"properties"`
	PatternProperties map[string]json.RawMessage `json:"patternProperties"`
	Definitions       map[string]json.RawMessage `json:"definitions"`
	Defs              map[string]json.RawMessage `json:"$defs"`
	Dependencies      map[string]json.RawMessage `json:"dependencies"`

	PropertyNames json.RawMessage `json:"propertyNames"`
	Contains      json.RawMessage `json:"contains"`
	Not           json.RawMessage `json:"not"`
	If            json.RawMessage `json:"if"`
	Then          json.RawMessage `json:"then"`
	Else          json.RawMessage `json:"else"`
	Items         json.RawMessage `json:"items"`

	AnyOf []json.RawMessage `json:"anyOf"`
	AllOf []json.RawMessage `json:"allOf"`
	OneOf []json.RawMessage `json:"oneOf"`
}

// isStrictSchemaCompatible reports whether every object in a schema sets
// additionalProperties to false, as Bedrock requires for strict tools.
// Boolean schemas are compatible; schemas that cannot be read are not.
func isStrictSchemaCompatible(schema json.RawMessage) bool {
	trimmed := bytes.TrimSpace(schema)
	if len(trimmed) == 0 || isJSONBoolean(trimmed) {
		return true
	}

	var node strictSchemaNode
	if err := json.Unmarshal(trimmed, &node); err != nil {
		return false
	}

	if describesObjectType(node.Type) && !bytes.Equal(bytes.TrimSpace(node.AdditionalProperties), []byte("false")) {
		return false
	}

	schemaMaps := []map[string]json.RawMessage{
		node.Properties,
		node.PatternProperties,
		node.Definitions,
		node.Defs,
	}

	for _, schemaMap := range schemaMaps {
		for _, definition := range schemaMap {
			if !isStrictSchemaCompatible(definition) {
				return false
			}
		}
	}

	for _, dependency := range node.Dependencies {
		if !isJSONArray(dependency) && !isStrictSchemaCompatible(dependency) {
			return false
		}
	}

	nested := []json.RawMessage{
		node.PropertyNames,
		node.Contains,
		node.Not,
		node.If,
		node.Then,
		node.Else,
	}

	if !allCompatible(nested) || !itemsCompatible(node.Items) {
		return false
	}

	return allCompatible(node.AnyOf) && allCompatible(node.AllOf) && allCompatible(node.OneOf)
}

// itemsCompatible checks a single items schema or a tuple of them.
func itemsCompatible(items json.RawMessage) bool {
	if !isJSONArray(items) {
		return isStrictSchemaCompatible(items)
	}

	var tuple []json.RawMessage
	if err := json.Unmarshal(items, &tuple); err != nil {
		return false
	}

	return allCompatible(tuple)
}

func allCompatible(schemas []json.RawMessage) bool {
	for _, schema := range schemas {
		if !isStrictSchemaCompatible(schema) {
			return false
		}
	}

	return true
}

// describesObjectType reports whether a type keyword is "object" or a list
// that includes it.
func describesObjectType(schemaType json.RawMessage) bool {
	var single string
	if err := json.Unmarshal(schemaType, &single); err == nil {
		return single == "object"
	}

	var list []string
	if err := json.Unmarshal(schemaType, &list); err == nil {
		return slices.Contains(list, "object")
	}

	return false
}

func isJSONBoolean(value []byte) bool {
	return bytes.Equal(value, []byte("true")) || bytes.Equal(value, []byte("false"))
}

func isJSONArray(value json.RawMessage) bool {
	trimmed := bytes.TrimSpace(value)

	return len(trimmed) > 0 && trimmed[0] == '['
}
