package ui

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
)

// encodeJSON encodes value like JSON.stringify: without HTML escaping and
// without a trailing newline.
func encodeJSON(value any) ([]byte, error) {
	var buffer bytes.Buffer

	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)

	if err := encoder.Encode(value); err != nil {
		return nil, err
	}

	return bytes.TrimSuffix(buffer.Bytes(), []byte("\n")), nil
}

// marshalWithType encodes fields as a JSON object whose first member is the
// "type" discriminator. fields must encode to an object and must not be the
// type whose MarshalJSON method calls marshalWithType.
func marshalWithType(typeName string, fields any) ([]byte, error) {
	encodedFields, err := encodeJSON(fields)
	if err != nil {
		return nil, err
	}

	encodedType, err := encodeJSON(typeName)
	if err != nil {
		return nil, err
	}

	var buffer bytes.Buffer
	buffer.WriteString(`{"type":`)
	buffer.Write(encodedType)

	if bytes.Equal(encodedFields, []byte("{}")) {
		buffer.WriteByte('}')

		return buffer.Bytes(), nil
	}

	buffer.WriteByte(',')
	buffer.Write(encodedFields[1:])

	return buffer.Bytes(), nil
}

// decodeType returns the "type" discriminator of a JSON object.
func decodeType(data []byte) (string, error) {
	var discriminator struct {
		Type *string `json:"type"`
	}

	if err := json.Unmarshal(data, &discriminator); err != nil {
		return "", err
	}

	if discriminator.Type == nil {
		return "", errors.New("missing type")
	}

	return *discriminator.Type, nil
}

// encodeValue encodes an in-process value, such as a tool output, as JSON.
// Nil encodes as null.
func encodeValue(value any) (json.RawMessage, error) {
	if raw, ok := value.(json.RawMessage); ok && raw != nil {
		return raw, nil
	}

	encoded, err := encodeJSON(value)
	if err != nil {
		return nil, err
	}

	return encoded, nil
}

// isJSONNull reports whether raw is absent or the JSON null.
func isJSONNull(raw json.RawMessage) bool {
	return len(bytes.TrimSpace(raw)) == 0 || string(bytes.TrimSpace(raw)) == "null"
}

// equalJSON reports whether two JSON documents hold the same data. Absent
// values only equal absent values.
func equalJSON(first, second json.RawMessage) bool {
	if first == nil || second == nil {
		return first == nil && second == nil
	}

	firstValue, firstErr := decodeJSONValue(first)
	secondValue, secondErr := decodeJSONValue(second)

	if firstErr != nil || secondErr != nil {
		return false
	}

	return reflect.DeepEqual(firstValue, secondValue)
}

// decodeJSONValue decodes raw keeping numbers exact.
func decodeJSONValue(raw json.RawMessage) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()

	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}

	return value, nil
}
