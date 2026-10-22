package codemode

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
)

// Upstream's runtime exchanges values in the run-js-v1 format: the devalue
// format, a JSON array of flattened values that refer to each other by
// index, with negative indexes for values JSON cannot express.
const (
	runSerdeUndefined        = -1
	runSerdeHole             = -2
	runSerdeNaN              = -3
	runSerdePositiveInfinity = -4
	runSerdeNegativeInfinity = -5
	runSerdeNegativeZero     = -6
	runSerdeSparse           = -7

	// maxSparseArrayLength bounds the arrays a decoded value may expand to.
	maxSparseArrayLength = 1 << 24
)

// encodeRunValue encodes a JSON document in the run-js-v1 format, as
// upstream encodes the value JSON.parse returns for it. An empty document is
// undefined.
func encodeRunValue(document json.RawMessage) (string, error) {
	if len(document) == 0 {
		return strconv.Itoa(runSerdeUndefined), nil
	}

	root, err := parseJSValue(document)
	if err != nil {
		return "", err
	}

	encoder := &runEncoder{
		primitives: make(map[string]int),
	}

	index := encoder.flatten(root)
	if index < 0 {
		return strconv.Itoa(index), nil
	}

	return "[" + strings.Join(encoder.slots, ",") + "]", nil
}

// jsKind is the type of a value parsed from JSON.
type jsKind int

const (
	jsNull jsKind = iota
	jsBoolean
	jsNumber
	jsString
	jsArray
	jsObject
)

// jsValue is a JSON value as JavaScript sees it: numbers are doubles and
// object members are in JavaScript's property order.
type jsValue struct {
	kind    jsKind
	boolean bool
	number  float64
	text    string
	items   []*jsValue
	members []jsMember
}

type jsMember struct {
	key   string
	value *jsValue
}

// parseJSValue parses a JSON document as JSON.parse does.
func parseJSValue(document json.RawMessage) (*jsValue, error) {
	decoder := json.NewDecoder(bytes.NewReader(document))
	decoder.UseNumber()

	value, err := parseJSToken(decoder)
	if err != nil {
		return nil, fmt.Errorf("parse json value: %w", err)
	}

	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("parse json value: unexpected data after the value")
	}

	return value, nil
}

func parseJSToken(decoder *json.Decoder) (*jsValue, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}

	switch typed := token.(type) {
	case json.Delim:
		if typed == '[' {
			return parseJSArray(decoder)
		}

		return parseJSObject(decoder)

	case string:
		return &jsValue{
			kind: jsString,
			text: typed,
		}, nil

	case json.Number:
		number, err := strconv.ParseFloat(typed.String(), 64)
		if err != nil && !errors.Is(err, strconv.ErrRange) {
			return nil, err
		}

		return &jsValue{
			kind:   jsNumber,
			number: number,
		}, nil

	case bool:
		return &jsValue{
			kind:    jsBoolean,
			boolean: typed,
		}, nil

	default:
		return &jsValue{
			kind: jsNull,
		}, nil
	}
}

func parseJSArray(decoder *json.Decoder) (*jsValue, error) {
	array := &jsValue{
		kind: jsArray,
	}

	for decoder.More() {
		item, err := parseJSToken(decoder)
		if err != nil {
			return nil, err
		}

		array.items = append(array.items, item)
	}

	if _, err := decoder.Token(); err != nil {
		return nil, err
	}

	return array, nil
}

// parseJSObject parses an object. A repeated key keeps its first position
// and its last value, and array index keys come first in ascending order,
// as in a JavaScript object.
func parseJSObject(decoder *json.Decoder) (*jsValue, error) {
	object := &jsValue{
		kind: jsObject,
	}

	positions := make(map[string]int)

	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, err
		}

		key, _ := token.(string)

		value, err := parseJSToken(decoder)
		if err != nil {
			return nil, err
		}

		if position, repeated := positions[key]; repeated {
			object.members[position].value = value

			continue
		}

		positions[key] = len(object.members)
		object.members = append(object.members, jsMember{
			key:   key,
			value: value,
		})
	}

	if _, err := decoder.Token(); err != nil {
		return nil, err
	}

	object.members = inPropertyOrder(object.members, jsMember.name)

	return object, nil
}

func (m jsMember) name() string {
	return m.key
}

// inPropertyOrder orders members as JavaScript enumerates the properties of
// an ordinary object: array indexes ascending, then the other keys in
// insertion order.
func inPropertyOrder[Member any](members []Member, key func(Member) string) []Member {
	var indexes, names []Member

	for _, member := range members {
		if _, ok := arrayIndex(key(member)); ok {
			indexes = append(indexes, member)
		} else {
			names = append(names, member)
		}
	}

	slices.SortStableFunc(indexes, func(left, right Member) int {
		leftIndex, _ := arrayIndex(key(left))
		rightIndex, _ := arrayIndex(key(right))

		return int(leftIndex - rightIndex)
	})

	return append(indexes, names...)
}

// arrayIndex parses a canonical array index, an integer below 2^32 - 1
// without leading zeros.
func arrayIndex(key string) (int64, bool) {
	if key == "" || (len(key) > 1 && key[0] == '0') {
		return 0, false
	}

	index, err := strconv.ParseUint(key, 10, 32)
	if err != nil || index == math.MaxUint32 {
		return 0, false
	}

	return int64(index), true
}

// runEncoder flattens values into the run-js-v1 format.
type runEncoder struct {
	slots []string

	// primitives maps a primitive to its slot: equal primitives share one
	// slot, as devalue looks values up by identity.
	primitives map[string]int
}

// flatten stores a value and returns its slot, or the negative index of a
// special value.
func (e *runEncoder) flatten(value *jsValue) int {
	if special, ok := specialNumberIndex(value); ok {
		return special
	}

	key, primitive := primitiveKey(value)
	if primitive {
		if index, found := e.primitives[key]; found {
			return index
		}
	}

	index := len(e.slots)
	e.slots = append(e.slots, "")

	if primitive {
		e.primitives[key] = index
	}

	e.slots[index] = e.stringify(value)

	return index
}

func (e *runEncoder) stringify(value *jsValue) string {
	switch value.kind {
	case jsNull:
		return "null"

	case jsBoolean:
		return strconv.FormatBool(value.boolean)

	case jsNumber:
		return formatJSNumber(value.number)

	case jsString:
		return devalueString(value.text)

	case jsArray:
		return e.stringifyArray(value)

	default:
		if hasOwnProtoKey(value) {
			return e.stringifyOwnProtoObject(value)
		}

		return e.stringifyObject(value)
	}
}

func (e *runEncoder) stringifyArray(array *jsValue) string {
	var builder strings.Builder

	builder.WriteByte('[')

	for position, item := range array.items {
		if position > 0 {
			builder.WriteByte(',')
		}

		builder.WriteString(strconv.Itoa(e.flatten(item)))
	}

	builder.WriteByte(']')

	return builder.String()
}

func (e *runEncoder) stringifyObject(object *jsValue) string {
	var builder strings.Builder

	builder.WriteByte('{')

	for position, member := range object.members {
		if position > 0 {
			builder.WriteByte(',')
		}

		builder.WriteString(devalueString(member.key))
		builder.WriteByte(':')
		builder.WriteString(strconv.Itoa(e.flatten(member.value)))
	}

	builder.WriteByte('}')

	return builder.String()
}

// stringifyOwnProtoObject encodes an object with an own "__proto__" member,
// which upstream's runtime reduces to its entries.
func (e *runEncoder) stringifyOwnProtoObject(object *jsValue) string {
	entries := &jsValue{
		kind: jsArray,
	}

	for _, member := range object.members {
		entries.items = append(entries.items, &jsValue{
			kind: jsArray,
			items: []*jsValue{
				{
					kind: jsString,
					text: member.key,
				},
				member.value,
			},
		})
	}

	reduced := &jsValue{
		kind: jsArray,
		items: []*jsValue{
			{
				kind: jsBoolean,
			},
			entries,
		},
	}

	return `["RunOwnProtoObject",` + strconv.Itoa(e.flatten(reduced)) + "]"
}

func hasOwnProtoKey(object *jsValue) bool {
	return slices.ContainsFunc(object.members, func(member jsMember) bool {
		return member.key == "__proto__"
	})
}

func specialNumberIndex(value *jsValue) (int, bool) {
	if value.kind != jsNumber {
		return 0, false
	}

	switch {
	case math.IsNaN(value.number):
		return runSerdeNaN, true
	case math.IsInf(value.number, 1):
		return runSerdePositiveInfinity, true
	case math.IsInf(value.number, -1):
		return runSerdeNegativeInfinity, true
	case value.number == 0 && math.Signbit(value.number):
		return runSerdeNegativeZero, true
	default:
		return 0, false
	}
}

func primitiveKey(value *jsValue) (string, bool) {
	switch value.kind {
	case jsNull:
		return "null", true
	case jsBoolean:
		return "boolean:" + strconv.FormatBool(value.boolean), true
	case jsNumber:
		return "number:" + formatJSNumber(value.number), true
	case jsString:
		return "string:" + value.text, true
	default:
		return "", false
	}
}

// devalueString writes a string as devalue does, which also escapes "<"
// and the line and paragraph separators.
func devalueString(text string) string {
	var builder strings.Builder

	builder.WriteByte('"')

	for _, character := range text {
		switch character {
		case '"':
			builder.WriteString(`\"`)
		case '<':
			builder.WriteString(`\u003C`)
		case '\\':
			builder.WriteString(`\\`)
		case '\n':
			builder.WriteString(`\n`)
		case '\r':
			builder.WriteString(`\r`)
		case '\t':
			builder.WriteString(`\t`)
		case '\b':
			builder.WriteString(`\b`)
		case '\f':
			builder.WriteString(`\f`)
		case '\u2028':
			builder.WriteString(`\u2028`)
		case '\u2029':
			builder.WriteString(`\u2029`)
		default:
			if character < 0x20 {
				fmt.Fprintf(&builder, `\u%04x`, character)

				continue
			}

			builder.WriteRune(character)
		}
	}

	builder.WriteByte('"')

	return builder.String()
}
