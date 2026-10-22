package google

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// JSON Schema is arbitrary JSON, so the sanitizer works on raw keyword values.
type schemaKeywords map[string]json.RawMessage

// sanitizeResponseSchema replaces "const" with a single-value "enum" in every
// subschema location that Google supports, because responseJsonSchema does
// not accept "const". Other keywords are preserved.
func sanitizeResponseSchema(schema json.RawMessage) (json.RawMessage, error) {
	sanitized, err := sanitizeSchema(schema)
	if err != nil {
		return nil, fmt.Errorf("%w: response format schema: %w", model.ErrInvalidArgument, err)
	}

	return sanitized, nil
}

func sanitizeSchema(schema json.RawMessage) (json.RawMessage, error) {
	var keywords schemaKeywords
	if err := json.Unmarshal(schema, &keywords); err != nil {
		return nil, err
	}

	if constValue, ok := keywords["const"]; ok {
		delete(keywords, "const")
		keywords["enum"] = json.RawMessage("[" + string(constValue) + "]")
	}

	transforms := map[string]func(json.RawMessage) (json.RawMessage, error){
		"properties":           sanitizeDefinitionMap,
		"$defs":                sanitizeDefinitionMap,
		"items":                sanitizeItems,
		"additionalProperties": sanitizeDefinition,
		"anyOf":                sanitizeDefinitionList,
		"oneOf":                sanitizeDefinitionList,
	}

	for keyword, transform := range transforms {
		value, ok := keywords[keyword]
		if !ok {
			continue
		}

		if !wirejson.HasValue(value) {
			delete(keywords, keyword)

			continue
		}

		sanitized, err := transform(value)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", keyword, err)
		}

		keywords[keyword] = sanitized
	}

	return json.Marshal(keywords)
}

// sanitizeDefinition sanitizes a subschema, which may be a boolean schema.
func sanitizeDefinition(definition json.RawMessage) (json.RawMessage, error) {
	if isBooleanSchema(definition) {
		return definition, nil
	}

	return sanitizeSchema(definition)
}

func sanitizeDefinitionMap(definitions json.RawMessage) (json.RawMessage, error) {
	var named map[string]json.RawMessage
	if err := json.Unmarshal(definitions, &named); err != nil {
		return nil, err
	}

	for name, definition := range named {
		sanitized, err := sanitizeDefinition(definition)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}

		named[name] = sanitized
	}

	return json.Marshal(named)
}

func sanitizeDefinitionList(definitions json.RawMessage) (json.RawMessage, error) {
	var list []json.RawMessage
	if err := json.Unmarshal(definitions, &list); err != nil {
		return nil, err
	}

	for index, definition := range list {
		sanitized, err := sanitizeDefinition(definition)
		if err != nil {
			return nil, fmt.Errorf("%d: %w", index, err)
		}

		list[index] = sanitized
	}

	return json.Marshal(list)
}

// sanitizeItems accepts a single items schema or a tuple of schemas.
func sanitizeItems(items json.RawMessage) (json.RawMessage, error) {
	if bytes.HasPrefix(bytes.TrimSpace(items), []byte("[")) {
		return sanitizeDefinitionList(items)
	}

	return sanitizeDefinition(items)
}

func isBooleanSchema(definition json.RawMessage) bool {
	trimmed := bytes.TrimSpace(definition)

	return bytes.Equal(trimmed, []byte("true")) || bytes.Equal(trimmed, []byte("false"))
}
