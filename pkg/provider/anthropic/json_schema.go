package anthropic

import (
	"bytes"
	"encoding/json"
	"strings"
	"unicode"

	"github.com/aarock1234/ai/internal/jsonobject"
)

// supportedStringFormats are the string formats Anthropic's constrained
// decoder accepts.
var supportedStringFormats = map[string]bool{
	"date-time": true,
	"time":      true,
	"date":      true,
	"duration":  true,
	"email":     true,
	"hostname":  true,
	"uri":       true,
	"ipv4":      true,
	"ipv6":      true,
	"uuid":      true,
}

// descriptionConstraintKeys are keywords Anthropic rejects in structured
// output schemas. Their values move into the description instead.
var descriptionConstraintKeys = []string{
	"minimum",
	"maximum",
	"exclusiveMinimum",
	"exclusiveMaximum",
	"multipleOf",
	"minLength",
	"maxLength",
	"pattern",
	"minItems",
	"maxItems",
	"uniqueItems",
	"minProperties",
	"maxProperties",
	"not",
}

// copiedSchemaKeys are kept unchanged, in this order.
var copiedSchemaKeys = []string{
	"$schema",
	"$id",
	"title",
	"description",
}

// sanitizeJSONSchema removes JSON Schema keywords that Anthropic rejects in
// output_config.format.schema. Applications still validate output against
// the original schema; this only relaxes what the constrained decoder sees.
func sanitizeJSONSchema(schema json.RawMessage) json.RawMessage {
	return sanitizeDefinition(schema)
}

// sanitizeDefinition sanitizes a schema object and passes boolean and other
// non-object definitions through.
func sanitizeDefinition(definition json.RawMessage) json.RawMessage {
	schema, err := jsonobject.Parse(definition)
	if err != nil {
		return definition
	}

	sanitized, err := sanitizeSchema(schema).Encode()
	if err != nil {
		return definition
	}

	return sanitized
}

func sanitizeSchema(schema jsonobject.Object) jsonobject.Object {
	if ref, found := presentValue(schema, "$ref"); found {
		return jsonobject.Object{
			{
				Key:   "$ref",
				Value: ref,
			},
		}
	}

	result := jsonobject.Object{}
	add := func(key string, value json.RawMessage) {
		result = append(result, jsonobject.Member{
			Key:   key,
			Value: value,
		})
	}

	for _, key := range copiedSchemaKeys {
		if value, found := presentValue(schema, key); found {
			add(key, value)
		}
	}

	for _, key := range []string{"default", "const"} {
		if value, found := schema.Get(key); found {
			add(key, value)
		}
	}

	for _, key := range []string{"enum", "type"} {
		if value, found := presentValue(schema, key); found {
			add(key, value)
		}
	}

	if anyOf, found := presentValue(schema, "anyOf"); found {
		add("anyOf", sanitizeDefinitionList(anyOf))
	} else if oneOf, found := presentValue(schema, "oneOf"); found {
		add("anyOf", sanitizeDefinitionList(oneOf))
	}

	if allOf, found := presentValue(schema, "allOf"); found {
		add("allOf", sanitizeDefinitionList(allOf))
	}

	for _, key := range []string{"definitions", "$defs"} {
		if definitions, found := presentValue(schema, key); found {
			add(key, sanitizeDefinitionMap(definitions))
		}
	}

	if describesObject(schema) {
		if properties, found := presentValue(schema, "properties"); found {
			add("properties", sanitizeDefinitionMap(properties))
		}

		add("additionalProperties", json.RawMessage("false"))

		if required, found := presentValue(schema, "required"); found {
			add("required", required)
		}
	}

	if items, found := presentValue(schema, "items"); found {
		add("items", sanitizeItems(items))
	}

	format, hasFormat := schema.GetString("format")
	if hasFormat && supportedStringFormats[format] {
		add("format", mustEncodeJSON(format))
	}

	if constraints := constraintDescription(schema); constraints != "" {
		result = withDescription(result, constraints)
	}

	return result
}

// presentValue returns the value of key unless it is absent or null.
func presentValue(schema jsonobject.Object, key string) (json.RawMessage, bool) {
	value, found := schema.Get(key)
	if !found || isJSONNull(value) {
		return nil, false
	}

	return value, true
}

func describesObject(schema jsonobject.Object) bool {
	schemaType, _ := schema.GetString("type")
	_, hasProperties := presentValue(schema, "properties")

	return schemaType == "object" || hasProperties
}

func sanitizeItems(items json.RawMessage) json.RawMessage {
	if isJSONArray(items) {
		return sanitizeDefinitionList(items)
	}

	return sanitizeDefinition(items)
}

func sanitizeDefinitionList(list json.RawMessage) json.RawMessage {
	var definitions []json.RawMessage
	if err := json.Unmarshal(list, &definitions); err != nil {
		return list
	}

	sanitized := make([]json.RawMessage, 0, len(definitions))
	for _, definition := range definitions {
		sanitized = append(sanitized, sanitizeDefinition(definition))
	}

	return mustEncodeJSON(sanitized)
}

func sanitizeDefinitionMap(definitions json.RawMessage) json.RawMessage {
	object, err := jsonobject.Parse(definitions)
	if err != nil {
		return definitions
	}

	sanitized := jsonobject.Object{}
	for _, member := range object {
		sanitized = append(sanitized, jsonobject.Member{
			Key:   member.Key,
			Value: sanitizeDefinition(member.Value),
		})
	}

	encoded, err := sanitized.Encode()
	if err != nil {
		return definitions
	}

	return encoded
}

// constraintDescription describes the rejected constraints of a schema, such
// as "min length: 3; format: binary.".
func constraintDescription(schema jsonobject.Object) string {
	var descriptions []string

	for _, key := range descriptionConstraintKeys {
		value, found := presentValue(schema, key)
		if !found || string(value) == "false" {
			continue
		}

		descriptions = append(descriptions, constraintName(key)+": "+constraintValue(value))
	}

	format, hasFormat := schema.GetString("format")
	if hasFormat && !supportedStringFormats[format] {
		descriptions = append(descriptions, "format: "+format)
	}

	if len(descriptions) == 0 {
		return ""
	}

	return strings.Join(descriptions, "; ") + "."
}

// withDescription appends constraints to the schema description, adding the
// description when the schema has none.
func withDescription(result jsonobject.Object, constraints string) jsonobject.Object {
	for index, member := range result {
		if member.Key != "description" {
			continue
		}

		description := constraintValue(member.Value) + "\n" + constraints
		result[index].Value = mustEncodeJSON(description)

		return result
	}

	return append(result, jsonobject.Member{
		Key:   "description",
		Value: mustEncodeJSON(constraints),
	})
}

// constraintName spells a camelCase keyword as lowercase words, such as
// "min length" for minLength.
func constraintName(key string) string {
	var name strings.Builder
	for _, character := range key {
		if unicode.IsUpper(character) {
			name.WriteByte(' ')
			name.WriteRune(unicode.ToLower(character))

			continue
		}

		name.WriteRune(character)
	}

	return name.String()
}

func constraintValue(value json.RawMessage) string {
	var text string
	if err := json.Unmarshal(value, &text); err == nil {
		return text
	}

	var compact bytes.Buffer
	if err := json.Compact(&compact, value); err != nil {
		return string(value)
	}

	return compact.String()
}

func isJSONNull(value json.RawMessage) bool {
	return string(bytes.TrimSpace(value)) == "null"
}

func isJSONArray(value json.RawMessage) bool {
	trimmed := bytes.TrimSpace(value)

	return len(trimmed) > 0 && trimmed[0] == '['
}
