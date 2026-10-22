package openai

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// localReferencePattern matches "#/definitions/<name>" and "#/$defs/<name>".
var localReferencePattern = regexp.MustCompile(`^#/(definitions|\$defs)/(.+)$`)

// Schema keywords whose values are subschemas, grouped by shape.
var (
	schemaRecordKeywords = []string{
		"properties",
		"patternProperties",
		"definitions",
		"$defs",
	}

	schemaDefinitionKeywords = []string{
		"additionalProperties",
		"additionalItems",
		"contains",
		"not",
		"if",
		"then",
		"else",
	}

	schemaListKeywords = []string{
		"allOf",
		"anyOf",
		"oneOf",
	}
)

// normalizeJSONSchema prepares a JSON Schema for OpenAI structured outputs,
// for both the chat and responses APIs. OpenAI rejects some keywords that only
// constrain validation, so they are removed with a compatibility warning:
//
//   - propertyNames with a string schema (other propertyNames fail),
//   - pattern values that use regex lookaround.
//
// A singleton allOf holding only a $ref, which Zod emits for recursive
// references, becomes a direct $ref; at the root it is expanded, because
// OpenAI requires an object root. Member order is preserved because OpenAI
// generates properties in schema order.
func normalizeJSONSchema(schema json.RawMessage) (json.RawMessage, []model.Warning, error) {
	if len(bytes.TrimSpace(schema)) == 0 {
		return schema, nil, nil
	}

	var normalizer schemaNormalizer

	normalized, err := normalizer.definition(schema, true)
	if err != nil {
		return nil, nil, err
	}

	return normalized, normalizer.warnings(), nil
}

// schemaNormalizer records which unsupported keywords it removed.
type schemaNormalizer struct {
	removedPropertyNames     bool
	removedLookaroundPattern bool
}

func (n *schemaNormalizer) warnings() []model.Warning {
	var warnings []model.Warning

	if n.removedPropertyNames {
		warnings = append(warnings, model.CompatibilityWarning(
			"JSON Schema propertyNames",
			"OpenAI does not support JSON Schema propertyNames. It was removed before sending the schema, so OpenAI will not enforce property-name constraints.",
		))
	}

	if n.removedLookaroundPattern {
		warnings = append(warnings, model.CompatibilityWarning(
			"JSON Schema pattern with regex lookaround",
			"OpenAI does not support regex lookaround in JSON Schema patterns. The pattern was removed before sending the schema, so OpenAI will not enforce that constraint.",
		))
	}

	return warnings
}

// definition normalizes a schema object. Boolean schemas and other values
// that are not objects are returned unchanged.
func (n *schemaNormalizer) definition(raw json.RawMessage, isRoot bool) (json.RawMessage, error) {
	object, err := jsonobject.Parse(raw)
	if errors.Is(err, jsonobject.ErrNotObject) {
		return raw, nil
	}

	if err != nil {
		return nil, fmt.Errorf("%w: decode JSON schema: %w", model.ErrInvalidArgument, err)
	}

	normalized, err := n.schema(object, isRoot)
	if err != nil {
		return nil, err
	}

	return normalized.Encode()
}

// subschema normalizes a schema nested inside another schema.
func (n *schemaNormalizer) subschema(raw json.RawMessage) (json.RawMessage, error) {
	return n.definition(raw, false)
}

// schema normalizes one schema object and its subschemas.
func (n *schemaNormalizer) schema(schema jsonobject.Object, isRoot bool) (jsonobject.Object, error) {
	schema, err := n.removeUnsupported(schema)
	if err != nil {
		return nil, err
	}

	schema, err = n.normalizeSubschemas(schema)
	if err != nil {
		return nil, err
	}

	reference, ok := singletonReference(schema)
	if !ok {
		return schema, nil
	}

	withoutAllOf := schema.Without("allOf")

	if !isRoot {
		return withoutAllOf.With("$ref", jsonobject.EncodeString(reference)), nil
	}

	// OpenAI requires an object at the root, so a local root reference is
	// expanded instead of being sent as a direct $ref.
	referenced, ok := localReferenceSchema(reference, schema)
	if !ok {
		return schema, nil
	}

	return referenced.Merge(withoutAllOf), nil
}

// removeUnsupported drops string propertyNames constraints and lookaround
// patterns.
func (n *schemaNormalizer) removeUnsupported(schema jsonobject.Object) (jsonobject.Object, error) {
	propertyNames, ok := schema.Get("propertyNames")
	if ok && !isJSONNull(propertyNames) {
		if !isStringSchema(propertyNames) {
			return nil, fmt.Errorf("%w: JSON Schema propertyNames that does not use a string schema", model.ErrUnsupported)
		}

		n.removedPropertyNames = true
	}

	schema = schema.Without("propertyNames")

	pattern, ok := schema.GetString("pattern")
	if ok && containsRegexLookaround(pattern) {
		schema = schema.Without("pattern")
		n.removedLookaroundPattern = true
	}

	return schema, nil
}

// normalizeSubschemas normalizes every keyword that holds subschemas.
func (n *schemaNormalizer) normalizeSubschemas(schema jsonobject.Object) (jsonobject.Object, error) {
	normalized := make(jsonobject.Object, 0, len(schema))

	for _, member := range schema {
		value, err := n.keyword(member.Key, member.Value)
		if err != nil {
			return nil, err
		}

		normalized = append(normalized, jsonobject.Member{
			Key:   member.Key,
			Value: value,
		})
	}

	return normalized, nil
}

// keyword normalizes the value of one schema keyword.
func (n *schemaNormalizer) keyword(key string, value json.RawMessage) (json.RawMessage, error) {
	if isJSONNull(value) {
		return value, nil
	}

	switch {
	case slices.Contains(schemaRecordKeywords, key):
		return mapSchemaRecord(value, n.subschema)

	case slices.Contains(schemaDefinitionKeywords, key):
		return n.subschema(value)

	case slices.Contains(schemaListKeywords, key):
		return n.list(value)

	case key == "items":
		if isJSONArray(value) {
			return n.list(value)
		}

		return n.subschema(value)

	case key == "dependencies":
		return mapSchemaRecord(value, n.dependency)

	default:
		return value, nil
	}
}

// dependency normalizes a schema dependency and keeps property-list
// dependencies unchanged.
func (n *schemaNormalizer) dependency(value json.RawMessage) (json.RawMessage, error) {
	if isJSONArray(value) {
		return value, nil
	}

	return n.subschema(value)
}

// list normalizes an array of subschemas.
func (n *schemaNormalizer) list(value json.RawMessage) (json.RawMessage, error) {
	var definitions []json.RawMessage
	if err := json.Unmarshal(value, &definitions); err != nil {
		return value, nil
	}

	for index, definition := range definitions {
		normalized, err := n.subschema(definition)
		if err != nil {
			return nil, err
		}

		definitions[index] = normalized
	}

	return json.Marshal(definitions)
}

// mapSchemaRecord transforms every value of an object of subschemas, keeping
// key order. Values that are not objects are returned unchanged.
func mapSchemaRecord(
	value json.RawMessage,
	transform func(json.RawMessage) (json.RawMessage, error),
) (json.RawMessage, error) {
	record, err := jsonobject.Parse(value)
	if err != nil {
		return value, nil
	}

	for index, member := range record {
		transformed, err := transform(member.Value)
		if err != nil {
			return nil, err
		}

		record[index].Value = transformed
	}

	return record.Encode()
}

// singletonReference returns the reference of an allOf that holds exactly one
// schema consisting only of a $ref. Intersections are not rewritten because
// only a single reference equals a direct $ref.
func singletonReference(schema jsonobject.Object) (string, bool) {
	allOf, ok := schema.Get("allOf")
	if !ok {
		return "", false
	}

	var definitions []json.RawMessage
	if err := json.Unmarshal(allOf, &definitions); err != nil || len(definitions) != 1 {
		return "", false
	}

	only, err := jsonobject.Parse(definitions[0])
	if err != nil || len(only) != 1 {
		return "", false
	}

	return only.GetString("$ref")
}

// localReferenceSchema returns the "#/definitions/..." or "#/$defs/..."
// definition a reference points to within the root schema.
func localReferenceSchema(reference string, root jsonobject.Object) (jsonobject.Object, bool) {
	match := localReferencePattern.FindStringSubmatch(reference)
	if match == nil {
		return nil, false
	}

	keyword := match[1]
	name := strings.NewReplacer("~1", "/", "~0", "~").Replace(match[2])

	definitionsValue, ok := root.Get(keyword)
	if !ok {
		return nil, false
	}

	definitions, err := jsonobject.Parse(definitionsValue)
	if err != nil {
		return nil, false
	}

	definition, ok := definitions.Get(name)
	if !ok {
		return nil, false
	}

	referenced, err := jsonobject.Parse(definition)
	if err != nil {
		return nil, false
	}

	return referenced, true
}

// isStringSchema reports whether value is an object schema whose type is
// "string".
func isStringSchema(value json.RawMessage) bool {
	object, err := jsonobject.Parse(value)
	if err != nil {
		return false
	}

	schemaType, _ := object.GetString("type")

	return schemaType == "string"
}

// containsRegexLookaround reports whether pattern has a lookahead or
// lookbehind group outside a character class.
func containsRegexLookaround(pattern string) bool {
	escaped := false
	inCharacterClass := false

	for index := 0; index < len(pattern); index++ {
		character := pattern[index]

		switch {
		case escaped:
			escaped = false
		case character == '\\':
			escaped = true
		case character == '[':
			inCharacterClass = true
		case character == ']':
			inCharacterClass = false
		case !inCharacterClass && isLookaroundStart(pattern[index:]):
			return true
		}
	}

	return false
}

// isLookaroundStart reports whether text starts with "(?=", "(?!", "(?<=" or
// "(?<!".
func isLookaroundStart(text string) bool {
	for _, prefix := range []string{"(?=", "(?!", "(?<=", "(?<!"} {
		if strings.HasPrefix(text, prefix) {
			return true
		}
	}

	return false
}

func isJSONNull(value json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(value), []byte("null"))
}

func isJSONArray(value json.RawMessage) bool {
	trimmed := bytes.TrimSpace(value)

	return len(trimmed) > 0 && trimmed[0] == '['
}
