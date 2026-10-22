package codemode

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

var errMalformedRunValue = errors.New("run-js-v1 value is malformed")

// decodeRunValue decodes a run-js-v1 value to the JSON document that
// JSON.stringify writes for the value upstream's runtime revives. Undefined
// decodes to nil.
func decodeRunValue(serialized string) (json.RawMessage, error) {
	decoder, root, err := newRunDecoder(serialized)
	if err != nil {
		return nil, err
	}

	return decoder.document(root)
}

// decodeRunArguments decodes the run-js-v1 argument list of a host function
// call. Undefined arguments decode to nil.
func decodeRunArguments(serialized string) ([]json.RawMessage, error) {
	decoder, root, err := newRunDecoder(serialized)
	if err != nil {
		return nil, err
	}

	list, ok := decoder.slot(root)
	if !ok || list.kind != jsArray || isTypedEntry(list) || isSparseEntry(list) {
		return nil, errors.New("host function arguments must be encoded as an array")
	}

	arguments := make([]json.RawMessage, len(list.items))

	for position, item := range list.items {
		index, ok := slotIndex(item)
		if !ok {
			return nil, errMalformedRunValue
		}

		if index == runSerdeHole {
			continue
		}

		argument, err := decoder.document(index)
		if err != nil {
			return nil, err
		}

		arguments[position] = argument
	}

	return arguments, nil
}

// runDecoder converts run-js-v1 values to JSON documents with
// JSON.stringify semantics.
type runDecoder struct {
	values []*jsValue

	// active holds the slots being converted, to reject cycles as
	// JSON.stringify does.
	active map[int]bool
}

func newRunDecoder(serialized string) (*runDecoder, int, error) {
	root, err := parseJSValue(json.RawMessage(serialized))
	if err != nil {
		return nil, 0, fmt.Errorf("run-js-v1 value is not JSON: %w", err)
	}

	decoder := &runDecoder{
		active: make(map[int]bool),
	}

	if root.kind == jsNumber {
		index, ok := slotIndex(root)
		if !ok || index >= 0 || index == runSerdeHole || index < runSerdeNegativeZero {
			return nil, 0, errMalformedRunValue
		}

		return decoder, index, nil
	}

	if root.kind != jsArray || len(root.items) == 0 {
		return nil, 0, errMalformedRunValue
	}

	decoder.values = root.items

	return decoder, 0, nil
}

func (d *runDecoder) slot(index int) (*jsValue, bool) {
	if index < 0 || index >= len(d.values) {
		return nil, false
	}

	return d.values[index], true
}

// document converts a slot to JSON; nil is undefined.
func (d *runDecoder) document(index int) (json.RawMessage, error) {
	var builder strings.Builder

	written, err := d.write(&builder, index)
	if err != nil {
		return nil, err
	}

	if !written {
		return nil, nil
	}

	return json.RawMessage(builder.String()), nil
}

// write writes the JSON of a slot and reports false for values that
// JSON.stringify omits, such as undefined.
func (d *runDecoder) write(builder *strings.Builder, index int) (bool, error) {
	switch index {
	case runSerdeUndefined:
		return false, nil

	case runSerdeNaN, runSerdePositiveInfinity, runSerdeNegativeInfinity:
		builder.WriteString("null")

		return true, nil

	case runSerdeNegativeZero:
		builder.WriteString("0")

		return true, nil
	}

	value, ok := d.slot(index)
	if !ok {
		return false, errMalformedRunValue
	}

	if d.active[index] {
		return false, errors.New("run-js-v1 value is cyclic")
	}

	d.active[index] = true
	defer delete(d.active, index)

	return d.writeSlot(builder, value)
}

func (d *runDecoder) writeSlot(builder *strings.Builder, value *jsValue) (bool, error) {
	switch value.kind {
	case jsNull:
		builder.WriteString("null")

	case jsBoolean, jsNumber, jsString:
		writePrimitive(builder, value)

	case jsObject:
		return true, d.writeObject(builder, value)

	default:
		return d.writeArrayLike(builder, value)
	}

	return true, nil
}

func writePrimitive(builder *strings.Builder, value *jsValue) {
	switch value.kind {
	case jsBoolean:
		if value.boolean {
			builder.WriteString("true")
		} else {
			builder.WriteString("false")
		}

	case jsNumber:
		if math.IsInf(value.number, 0) {
			builder.WriteString("null")

			return
		}

		builder.WriteString(formatJSNumber(value.number))

	default:
		writeJSONString(builder, value.text)
	}
}

// runMember is an object member that refers to a slot.
type runMember struct {
	key   string
	index int
}

func (m runMember) name() string {
	return m.key
}

func (d *runDecoder) writeObject(builder *strings.Builder, object *jsValue) error {
	members := make([]runMember, 0, len(object.members))

	for _, member := range object.members {
		index, ok := slotIndex(member.value)
		if !ok || member.key == "__proto__" {
			return errMalformedRunValue
		}

		members = append(members, runMember{
			key:   member.key,
			index: index,
		})
	}

	return d.writeMembers(builder, members)
}

// writeMembers writes an object in JavaScript property order, omitting
// members whose values JSON.stringify omits.
func (d *runDecoder) writeMembers(builder *strings.Builder, members []runMember) error {
	builder.WriteByte('{')

	first := true

	for _, member := range inPropertyOrder(members, runMember.name) {
		var value strings.Builder

		written, err := d.write(&value, member.index)
		if err != nil {
			return err
		}

		if !written {
			continue
		}

		if !first {
			builder.WriteByte(',')
		}

		first = false

		writeJSONString(builder, member.key)
		builder.WriteByte(':')
		builder.WriteString(value.String())
	}

	builder.WriteByte('}')

	return nil
}

func (d *runDecoder) writeArrayLike(builder *strings.Builder, array *jsValue) (bool, error) {
	if isTypedEntry(array) {
		return d.writeTyped(builder, array.items[0].text, array.items[1:])
	}

	if isSparseEntry(array) {
		return true, d.writeSparseArray(builder, array.items)
	}

	indexes := make([]int, len(array.items))

	for position, item := range array.items {
		index, ok := slotIndex(item)
		if !ok {
			return false, errMalformedRunValue
		}

		indexes[position] = index
	}

	return true, d.writeElements(builder, indexes)
}

// writeElements writes an array; holes and values JSON.stringify omits are
// written as null.
func (d *runDecoder) writeElements(builder *strings.Builder, indexes []int) error {
	builder.WriteByte('[')

	for position, index := range indexes {
		if position > 0 {
			builder.WriteByte(',')
		}

		if index == runSerdeHole {
			builder.WriteString("null")

			continue
		}

		written, err := d.write(builder, index)
		if err != nil {
			return err
		}

		if !written {
			builder.WriteString("null")
		}
	}

	builder.WriteByte(']')

	return nil
}

// writeSparseArray writes [-7, length, index, value, ...].
func (d *runDecoder) writeSparseArray(builder *strings.Builder, items []*jsValue) error {
	length, ok := slotIndex(items[1])
	if !ok || length < 0 || length > maxSparseArrayLength || len(items)%2 != 0 {
		return errors.New("run-js-v1 sparse array is malformed or too large")
	}

	indexes := make([]int, length)
	for position := range indexes {
		indexes[position] = runSerdeHole
	}

	for position := 2; position < len(items); position += 2 {
		element, elementOK := slotIndex(items[position])
		value, valueOK := slotIndex(items[position+1])

		if !elementOK || !valueOK || element < 0 || element >= length {
			return errors.New("run-js-v1 sparse array is malformed")
		}

		indexes[element] = value
	}

	return d.writeElements(builder, indexes)
}

// writeTyped writes the JSON of a value devalue encodes as a typed entry,
// such as ["Date", "2026-01-01T00:00:00.000Z"].
func (d *runDecoder) writeTyped(builder *strings.Builder, name string, arguments []*jsValue) (bool, error) {
	switch name {
	case "Date":
		return true, writeDate(builder, arguments)

	case "Object":
		return d.writeReduced(builder, arguments)

	case "null":
		return true, d.writeNullPrototypeObject(builder, arguments)

	case "RunOwnProtoObject":
		return true, d.writeOwnProtoObject(builder, arguments)

	case "RunErrorValue":
		return true, d.writeErrorValue(builder, arguments)

	case "Map", "Set", "RegExp", "ArrayBuffer", "DataView", "URL", "URLSearchParams":
		builder.WriteString("{}")

		return true, nil

	case "BigInt", "BigInt64Array", "BigUint64Array":
		return false, errors.New("a BigInt value cannot be encoded as JSON")
	}

	if _, ok := typedArrayElementSize(name); ok {
		return true, d.writeTypedArray(builder, name, arguments)
	}

	return false, fmt.Errorf("run-js-v1 value has an unsupported type %q", name)
}

// writeDate writes a date as Date.prototype.toJSON does: its ISO string, or
// null for an invalid date.
func writeDate(builder *strings.Builder, arguments []*jsValue) error {
	if len(arguments) != 1 || arguments[0].kind != jsString {
		return errMalformedRunValue
	}

	if arguments[0].text == "" {
		builder.WriteString("null")

		return nil
	}

	writeJSONString(builder, arguments[0].text)

	return nil
}

// writeReduced writes the slot a single-argument entry refers to, such as
// the primitive of a boxed value.
func (d *runDecoder) writeReduced(builder *strings.Builder, arguments []*jsValue) (bool, error) {
	index, err := reducedIndex(arguments)
	if err != nil {
		return false, err
	}

	return d.write(builder, index)
}

func (d *runDecoder) writeNullPrototypeObject(builder *strings.Builder, arguments []*jsValue) error {
	if len(arguments)%2 != 0 {
		return errMalformedRunValue
	}

	members := make([]runMember, 0, len(arguments)/2)

	for position := 0; position < len(arguments); position += 2 {
		key := arguments[position]
		index, ok := slotIndex(arguments[position+1])

		if !ok || key.kind != jsString || key.text == "__proto__" {
			return errMalformedRunValue
		}

		members = append(members, runMember{
			key:   key.text,
			index: index,
		})
	}

	return d.writeMembers(builder, members)
}

// writeOwnProtoObject writes an object with an own "__proto__" member,
// which upstream reduces to [nullPrototype, [[key, value], ...]].
func (d *runDecoder) writeOwnProtoObject(builder *strings.Builder, arguments []*jsValue) error {
	index, err := reducedIndex(arguments)
	if err != nil {
		return err
	}

	reduced, ok := d.arraySlot(index)
	if !ok || len(reduced.items) != 2 {
		return errMalformedRunValue
	}

	listIndex, ok := slotIndex(reduced.items[1])
	if !ok {
		return errMalformedRunValue
	}

	list, ok := d.arraySlot(listIndex)
	if !ok {
		return errMalformedRunValue
	}

	members := make([]runMember, 0, len(list.items))

	for _, item := range list.items {
		member, err := d.ownProtoMember(item)
		if err != nil {
			return err
		}

		members = append(members, member)
	}

	return d.writeMembers(builder, members)
}

func (d *runDecoder) ownProtoMember(item *jsValue) (runMember, error) {
	entryIndex, ok := slotIndex(item)
	if !ok {
		return runMember{}, errMalformedRunValue
	}

	entry, ok := d.arraySlot(entryIndex)
	if !ok || len(entry.items) != 2 {
		return runMember{}, errMalformedRunValue
	}

	keyIndex, keyOK := slotIndex(entry.items[0])
	valueIndex, valueOK := slotIndex(entry.items[1])

	if !keyOK || !valueOK {
		return runMember{}, errMalformedRunValue
	}

	key, ok := d.slot(keyIndex)
	if !ok || key.kind != jsString {
		return runMember{}, errMalformedRunValue
	}

	return runMember{
		key:   key.text,
		index: valueIndex,
	}, nil
}

// writeErrorValue writes an error as JSON.stringify does: after upstream
// revives it, its name is its only own enumerable property.
func (d *runDecoder) writeErrorValue(builder *strings.Builder, arguments []*jsValue) error {
	index, err := reducedIndex(arguments)
	if err != nil {
		return err
	}

	fields, ok := d.slot(index)
	if !ok || fields.kind != jsObject {
		return errMalformedRunValue
	}

	for _, member := range fields.members {
		if member.key != "name" {
			continue
		}

		nameIndex, ok := slotIndex(member.value)
		if !ok {
			return errMalformedRunValue
		}

		name, ok := d.slot(nameIndex)
		if !ok || name.kind != jsString {
			return errMalformedRunValue
		}

		builder.WriteString(`{"name":`)
		writeJSONString(builder, name.text)
		builder.WriteByte('}')

		return nil
	}

	return errMalformedRunValue
}

func (d *runDecoder) arraySlot(index int) (*jsValue, bool) {
	value, ok := d.slot(index)

	return value, ok && value.kind == jsArray
}

// writeTypedArray writes a typed array as JSON.stringify does: an object
// keyed by element index.
func (d *runDecoder) writeTypedArray(builder *strings.Builder, name string, arguments []*jsValue) error {
	data, err := d.typedArrayBytes(name, arguments)
	if err != nil {
		return err
	}

	size, _ := typedArrayElementSize(name)

	builder.WriteByte('{')

	for element := range len(data) / size {
		if element > 0 {
			builder.WriteByte(',')
		}

		fmt.Fprintf(builder, `"%d":`, element)
		builder.WriteString(typedArrayElement(name, data[element*size:(element+1)*size]))
	}

	builder.WriteByte('}')

	return nil
}

// typedArrayBytes returns the bytes a typed array views: its whole buffer,
// or the byte offset and element length that follow the buffer.
func (d *runDecoder) typedArrayBytes(name string, arguments []*jsValue) ([]byte, error) {
	if len(arguments) != 1 && len(arguments) != 3 {
		return nil, errMalformedRunValue
	}

	bufferIndex, ok := slotIndex(arguments[0])
	if !ok {
		return nil, errMalformedRunValue
	}

	buffer, ok := d.arraySlot(bufferIndex)
	if !ok || len(buffer.items) != 2 || buffer.items[0].text != "ArrayBuffer" || buffer.items[1].kind != jsString {
		return nil, errMalformedRunValue
	}

	data, err := base64.StdEncoding.DecodeString(buffer.items[1].text)
	if err != nil {
		return nil, errMalformedRunValue
	}

	size, _ := typedArrayElementSize(name)

	if len(arguments) == 3 {
		offset, offsetOK := slotIndex(arguments[1])
		length, lengthOK := slotIndex(arguments[2])

		if !offsetOK || !lengthOK || offset < 0 || length < 0 || offset+length*size > len(data) {
			return nil, errMalformedRunValue
		}

		data = data[offset : offset+length*size]
	}

	if len(data)%size != 0 {
		return nil, errMalformedRunValue
	}

	return data, nil
}

func typedArrayElementSize(name string) (int, bool) {
	switch name {
	case "Int8Array", "Uint8Array", "Uint8ClampedArray":
		return 1, true
	case "Int16Array", "Uint16Array", "Float16Array":
		return 2, true
	case "Int32Array", "Uint32Array", "Float32Array":
		return 4, true
	case "Float64Array":
		return 8, true
	default:
		return 0, false
	}
}

// typedArrayElement formats one little-endian element as JSON.
func typedArrayElement(name string, data []byte) string {
	var number float64

	switch name {
	case "Int8Array":
		number = float64(int8(data[0]))
	case "Uint8Array", "Uint8ClampedArray":
		number = float64(data[0])
	case "Int16Array":
		number = float64(int16(binary.LittleEndian.Uint16(data)))
	case "Uint16Array":
		number = float64(binary.LittleEndian.Uint16(data))
	case "Float16Array":
		number = float16Value(binary.LittleEndian.Uint16(data))
	case "Int32Array":
		number = float64(int32(binary.LittleEndian.Uint32(data)))
	case "Uint32Array":
		number = float64(binary.LittleEndian.Uint32(data))
	case "Float32Array":
		number = float64(math.Float32frombits(binary.LittleEndian.Uint32(data)))
	default:
		number = math.Float64frombits(binary.LittleEndian.Uint64(data))
	}

	if math.IsNaN(number) || math.IsInf(number, 0) {
		return "null"
	}

	return formatJSNumber(number)
}

// float16Value converts an IEEE 754 half-precision value.
func float16Value(bits uint16) float64 {
	sign := 1.0
	if bits&0x8000 != 0 {
		sign = -1
	}

	exponent := int(bits>>10) & 0x1f
	fraction := float64(bits & 0x3ff)

	switch exponent {
	case 0:
		return sign * math.Ldexp(fraction, -24)

	case 0x1f:
		if fraction == 0 {
			return math.Inf(int(sign))
		}

		return math.NaN()

	default:
		return sign * math.Ldexp(1024+fraction, exponent-25)
	}
}

func reducedIndex(arguments []*jsValue) (int, error) {
	if len(arguments) != 1 {
		return 0, errMalformedRunValue
	}

	index, ok := slotIndex(arguments[0])
	if !ok {
		return 0, errMalformedRunValue
	}

	return index, nil
}

func isSparseEntry(array *jsValue) bool {
	if len(array.items) < 2 {
		return false
	}

	index, ok := slotIndex(array.items[0])

	return ok && index == runSerdeSparse
}

func isTypedEntry(array *jsValue) bool {
	return len(array.items) > 0 && array.items[0].kind == jsString
}

// slotIndex reads an integer that refers to a slot or a special value.
func slotIndex(value *jsValue) (int, bool) {
	if value.kind != jsNumber || value.number != math.Trunc(value.number) {
		return 0, false
	}

	if value.number < math.MinInt32 || value.number > math.MaxInt32 {
		return 0, false
	}

	return int(value.number), true
}
