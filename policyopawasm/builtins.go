package policyopawasm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"regexp"
)

// BuiltinFunc implements an OPA built-in function that a policy compiled to
// WebAssembly calls on the host. It receives the JSON values of the
// arguments and returns the JSON value of the result; a nil result leaves
// the call undefined.
type BuiltinFunc func(ctx context.Context, arguments []json.RawMessage) (json.RawMessage, error)

// defaultBuiltins are the host built-in functions that
// @open-policy-agent/opa-wasm provides.
var defaultBuiltins = map[string]BuiltinFunc{
	"json.is_valid":  jsonIsValid,
	"regex.split":    regexSplit,
	"sprintf":        sprintf,
	"yaml.is_valid":  yamlIsValid,
	"yaml.marshal":   yamlMarshal,
	"yaml.unmarshal": yamlUnmarshal,
}

// jsonIsValid reports whether a string is valid JSON, and is undefined for
// other values.
func jsonIsValid(_ context.Context, arguments []json.RawMessage) (json.RawMessage, error) {
	var text string

	isString := decodeArgument(arguments, 0, &text) == nil
	if !isString {
		return nil, nil
	}

	return json.Marshal(json.Valid([]byte(text)))
}

// regexSplit splits a string around the matches of a regular expression.
func regexSplit(_ context.Context, arguments []json.RawMessage) (json.RawMessage, error) {
	var pattern, text string

	if err := decodeArgument(arguments, 0, &pattern); err != nil {
		return nil, err
	}

	if err := decodeArgument(arguments, 1, &text); err != nil {
		return nil, err
	}

	expression, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("compile pattern: %w", err)
	}

	return json.Marshal(expression.Split(text, -1))
}

// sprintf formats an array of values with Go format verbs, as OPA's own
// sprintf built-in function does.
func sprintf(_ context.Context, arguments []json.RawMessage) (json.RawMessage, error) {
	var format string
	var values []json.RawMessage

	if err := decodeArgument(arguments, 0, &format); err != nil {
		return nil, err
	}

	if err := decodeArgument(arguments, 1, &values); err != nil {
		return nil, err
	}

	operands := make([]any, 0, len(values))
	for _, value := range values {
		operands = append(operands, sprintfOperand(value))
	}

	return json.Marshal(fmt.Sprintf(format, operands...))
}

// sprintfOperand converts a JSON value to the Go value OPA formats: numbers
// become integers when they are whole, strings and booleans keep their
// type, and other values are formatted as JSON text.
func sprintfOperand(value json.RawMessage) any {
	decoder := json.NewDecoder(bytes.NewReader(value))
	decoder.UseNumber()

	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		return string(value)
	}

	switch typed := decoded.(type) {
	case json.Number:
		return sprintfNumber(typed)
	case string, bool:
		return typed
	default:
		return string(value)
	}
}

// sprintfNumber returns an integer for whole numbers and a float otherwise.
func sprintfNumber(number json.Number) any {
	if integer, err := number.Int64(); err == nil {
		return integer
	}

	if integer, ok := new(big.Int).SetString(number.String(), 10); ok {
		return integer
	}

	if float, err := number.Float64(); err == nil {
		return float
	}

	return number.String()
}

// decodeArgument decodes the argument at index into target.
func decodeArgument(arguments []json.RawMessage, index int, target any) error {
	if index >= len(arguments) {
		return fmt.Errorf("missing argument %d", index+1)
	}

	if err := json.Unmarshal(arguments[index], target); err != nil {
		return fmt.Errorf("decode argument %d: %w", index+1, err)
	}

	return nil
}
