package policyopawasm

import (
	"context"
	"encoding/json"
	"fmt"

	"go.yaml.in/yaml/v3"
)

// yamlIsValid reports whether a string is valid YAML, and is undefined for
// other values.
func yamlIsValid(_ context.Context, arguments []json.RawMessage) (json.RawMessage, error) {
	var text string

	isString := decodeArgument(arguments, 0, &text) == nil
	if !isString {
		return nil, nil
	}

	var document any
	valid := yaml.Unmarshal([]byte(text), &document) == nil

	return json.Marshal(valid)
}

// yamlMarshal encodes a JSON value as a YAML document.
func yamlMarshal(_ context.Context, arguments []json.RawMessage) (json.RawMessage, error) {
	var value any
	if err := decodeArgument(arguments, 0, &value); err != nil {
		return nil, err
	}

	encoded, err := yaml.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode yaml: %w", err)
	}

	return json.Marshal(string(encoded))
}

// yamlUnmarshal decodes a YAML document into its JSON value. Mappings must
// have string keys, as JSON objects do.
func yamlUnmarshal(_ context.Context, arguments []json.RawMessage) (json.RawMessage, error) {
	var text string
	if err := decodeArgument(arguments, 0, &text); err != nil {
		return nil, err
	}

	var document any
	if err := yaml.Unmarshal([]byte(text), &document); err != nil {
		return nil, fmt.Errorf("decode yaml: %w", err)
	}

	encoded, err := json.Marshal(document)
	if err != nil {
		return nil, fmt.Errorf("convert yaml to json: %w", err)
	}

	return encoded, nil
}
