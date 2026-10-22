// Package jsonobject edits JSON objects without reordering their members.
//
// Providers rewrite opaque JSON, such as tool input and JSON Schema, where
// member order matters: models generate properties in schema order, and
// replayed prompts must stay byte-stable for prompt caching. The operations
// mirror JavaScript object semantics, so ports of upstream object spreads keep
// the same key order.
package jsonobject

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

var (
	// ErrNotObject reports JSON that is valid but is not an object.
	ErrNotObject = errors.New("value is not a JSON object")

	// ErrInvalidValue reports a member value that is not valid JSON.
	ErrInvalidValue = errors.New("invalid JSON object member value")
)

// Member is one member of an Object.
type Member struct {
	Key   string
	Value json.RawMessage
}

// Object is a JSON object whose members keep their order. The zero value is
// an empty object.
type Object []Member

// Parse decodes raw as a JSON object. It returns an error wrapping
// ErrNotObject when raw holds another JSON value. As in JavaScript, a
// duplicate key keeps its first position and takes its last value.
func Parse(raw []byte) (Object, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))

	opening, err := decoder.Token()
	if err != nil {
		return nil, err
	}

	if opening != json.Delim('{') {
		return nil, ErrNotObject
	}

	object := Object{}
	for decoder.More() {
		member, err := decodeMember(decoder)
		if err != nil {
			return nil, err
		}

		object = object.With(member.Key, member.Value)
	}

	if _, err := decoder.Token(); err != nil {
		return nil, err
	}

	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("unexpected data after JSON object")
	}

	return object, nil
}

// IsObject reports whether raw is a JSON object.
func IsObject(raw []byte) bool {
	_, err := Parse(raw)

	return err == nil
}

func decodeMember(decoder *json.Decoder) (Member, error) {
	keyToken, err := decoder.Token()
	if err != nil {
		return Member{}, err
	}

	key, ok := keyToken.(string)
	if !ok {
		return Member{}, fmt.Errorf("unexpected object key %v", keyToken)
	}

	var value json.RawMessage
	if err := decoder.Decode(&value); err != nil {
		return Member{}, err
	}

	return Member{
		Key:   key,
		Value: value,
	}, nil
}

// UnmarshalJSON decodes a JSON object, keeping its member order.
func (o *Object) UnmarshalJSON(data []byte) error {
	object, err := Parse(data)
	if err != nil {
		return err
	}

	*o = object

	return nil
}

// MarshalJSON encodes the members in order. An empty member value encodes
// as null, and a member value that is not valid JSON fails with an error
// wrapping ErrInvalidValue.
func (o Object) MarshalJSON() ([]byte, error) {
	var buffer bytes.Buffer
	buffer.WriteByte('{')

	for index, member := range o {
		if index > 0 {
			buffer.WriteByte(',')
		}

		key, err := json.Marshal(member.Key)
		if err != nil {
			return nil, err
		}

		value, err := memberValue(member)
		if err != nil {
			return nil, err
		}

		buffer.Write(key)
		buffer.WriteByte(':')
		buffer.Write(value)
	}

	buffer.WriteByte('}')

	return buffer.Bytes(), nil
}

// memberValue returns the JSON value of a member: null when it is empty,
// and an error when it is not valid JSON.
func memberValue(member Member) (json.RawMessage, error) {
	if len(member.Value) == 0 {
		return json.RawMessage("null"), nil
	}

	if !json.Valid(member.Value) {
		return nil, fmt.Errorf("%w: member %q", ErrInvalidValue, member.Key)
	}

	return member.Value, nil
}

// Encode returns the object as JSON, like MarshalJSON.
func (o Object) Encode() (json.RawMessage, error) {
	return o.MarshalJSON()
}

// Get returns the value of key.
func (o Object) Get(key string) (json.RawMessage, bool) {
	for _, member := range o {
		if member.Key == key {
			return member.Value, true
		}
	}

	return nil, false
}

// Has reports whether the object contains key.
func (o Object) Has(key string) bool {
	_, found := o.Get(key)

	return found
}

// GetString returns the value of key when it is a JSON string.
func (o Object) GetString(key string) (string, bool) {
	value, found := o.Get(key)
	if !found {
		return "", false
	}

	var text string
	if err := json.Unmarshal(value, &text); err != nil {
		return "", false
	}

	return text, true
}

// With returns a copy of the object with key set to value. An existing member
// keeps its position; a new member is appended.
func (o Object) With(key string, value json.RawMessage) Object {
	updated := make(Object, 0, len(o)+1)
	replaced := false

	for _, member := range o {
		if member.Key == key {
			member.Value = value
			replaced = true
		}

		updated = append(updated, member)
	}

	if replaced {
		return updated
	}

	return append(updated, Member{
		Key:   key,
		Value: value,
	})
}

// WithFirst returns a copy of the object with key as its first member, like
// the JavaScript spread {key: value, ...object}: an existing member keeps its
// own value but moves to the front.
func (o Object) WithFirst(key string, value json.RawMessage) Object {
	result := Object{
		{
			Key:   key,
			Value: value,
		},
	}

	for _, member := range o {
		if member.Key == key {
			result[0].Value = member.Value

			continue
		}

		result = append(result, member)
	}

	return result
}

// Without returns a copy of the object without key.
func (o Object) Without(key string) Object {
	remaining := make(Object, 0, len(o))

	for _, member := range o {
		if member.Key != key {
			remaining = append(remaining, member)
		}
	}

	return remaining
}

// Merge returns the members of o overridden and extended by other, like the
// JavaScript spread {...o, ...other}.
func (o Object) Merge(other Object) Object {
	merged := append(Object{}, o...)

	for _, member := range other {
		merged = merged.With(member.Key, member.Value)
	}

	return merged
}

// Members encodes value, which must encode to a JSON object, and returns its
// members by key. Unlike Parse, the result does not keep the member order;
// callers use it to merge members into a request body that encoding/json
// sorts anyway.
func Members(value any) (map[string]json.RawMessage, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}

	var members map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &members); err != nil {
		return nil, fmt.Errorf("decode object members: %w", err)
	}

	return members, nil
}

// EncodeString encodes text as a JSON string value. Encoding a string cannot
// fail: invalid UTF-8 is replaced rather than rejected.
func EncodeString(text string) json.RawMessage {
	encoded, _ := json.Marshal(text)

	return encoded
}
