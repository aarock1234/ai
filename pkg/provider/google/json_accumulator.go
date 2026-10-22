package google

import (
	"bytes"
	"encoding/json"
	"regexp"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/optional"
)

// partialArgument is one streamed function call argument. JSONPath locates
// the value, such as "$.recipe.ingredients[0].name".
type partialArgument struct {
	JSONPath    string   `json:"jsonPath"`
	StringValue *string  `json:"stringValue,omitempty"`
	NumberValue *float64 `json:"numberValue,omitempty"`
	BoolValue   *bool    `json:"boolValue,omitempty"`

	// NullValue is present, usually as JSON null, when the value is null.
	NullValue json.RawMessage `json:"nullValue,omitempty"`

	WillContinue *bool `json:"willContinue,omitempty"`
}

func (a partialArgument) continues() bool {
	return optional.Or(a.WillContinue, false)
}

var pathIndexPattern = regexp.MustCompile(`\[(\d+)\]`)

// pathSegment is an object key or an array index.
type pathSegment struct {
	key     string
	index   int
	isIndex bool
}

// openContainer is an object or array whose closing bracket has not been
// written yet.
type openContainer struct {
	segment    pathSegment
	isArray    bool
	childCount int
}

// argumentAccumulator builds function call arguments from streamed partial
// arguments. It keeps the arguments as a value and as JSON text, so the text
// deltas it returns concatenate to the same JSON that finalize encodes.
type argumentAccumulator struct {
	arguments  *argumentValue
	text       strings.Builder
	open       []openContainer
	stringOpen bool
}

func newArgumentAccumulator() *argumentAccumulator {
	return &argumentAccumulator{
		arguments: newObjectValue(),
	}
}

// process applies partial arguments and returns the JSON text they add.
func (a *argumentAccumulator) process(arguments []partialArgument) string {
	var delta strings.Builder

	for _, argument := range arguments {
		rawPath := strings.TrimPrefix(argument.JSONPath, "$.")
		if rawPath == "" {
			continue
		}

		segments := parsePath(rawPath)
		existing := a.arguments.lookup(segments)

		if argument.StringValue != nil && existing != nil {
			existing.appendText(*argument.StringValue)
			delta.WriteString(escapeStringContent(*argument.StringValue))

			continue
		}

		value, ok := resolvePartialValue(argument)
		if !ok {
			continue
		}

		a.arguments.store(segments, value)
		delta.WriteString(a.navigateTo(segments, argument, value.encode()))
	}

	a.text.WriteString(delta.String())

	return delta.String()
}

// finalize returns the complete arguments and the text that closes the
// deltas returned so far.
func (a *argumentAccumulator) finalize() (string, string) {
	final := a.arguments.encode()
	written := a.text.String()

	if !strings.HasPrefix(final, written) {
		return final, ""
	}

	return final, final[len(written):]
}

// navigateTo writes the text that moves from the open containers to the
// target leaf, followed by the leaf value.
func (a *argumentAccumulator) navigateTo(target []pathSegment, argument partialArgument, valueJSON string) string {
	var fragment strings.Builder

	if a.stringOpen {
		fragment.WriteString(`"`)
		a.stringOpen = false
	}

	fragment.WriteString(a.ensureRoot())

	containerPath := target[:len(target)-1]
	leaf := target[len(target)-1]

	fragment.WriteString(a.closeDownTo(a.commonDepth(containerPath)))
	fragment.WriteString(a.openDownTo(containerPath, leaf))
	fragment.WriteString(a.writeLeaf(leaf, argument, valueJSON))

	return fragment.String()
}

func (a *argumentAccumulator) ensureRoot() string {
	if len(a.open) > 0 {
		return ""
	}

	a.open = append(a.open, openContainer{})

	return "{"
}

// commonDepth returns the number of open containers to keep when moving to
// containerPath. The root is always kept.
func (a *argumentAccumulator) commonDepth(containerPath []pathSegment) int {
	limit := min(len(a.open)-1, len(containerPath))

	common := 0
	for common < limit && a.open[common+1].segment == containerPath[common] {
		common++
	}

	return common + 1
}

func (a *argumentAccumulator) closeDownTo(depth int) string {
	var fragment strings.Builder

	for len(a.open) > depth {
		closing := a.open[len(a.open)-1]
		a.open = a.open[:len(a.open)-1]

		if closing.isArray {
			fragment.WriteString("]")
		} else {
			fragment.WriteString("}")
		}
	}

	return fragment.String()
}

// openDownTo opens the containers of containerPath that are not open yet.
// The next segment decides whether each container is an object or an array.
func (a *argumentAccumulator) openDownTo(containerPath []pathSegment, leaf pathSegment) string {
	var fragment strings.Builder

	for position := len(a.open) - 1; position < len(containerPath); position++ {
		segment := containerPath[position]
		fragment.WriteString(a.beginChild(segment))

		next := leaf
		if position+1 < len(containerPath) {
			next = containerPath[position+1]
		}

		if next.isIndex {
			fragment.WriteString("[")
		} else {
			fragment.WriteString("{")
		}

		a.open = append(a.open, openContainer{
			segment: segment,
			isArray: next.isIndex,
		})
	}

	return fragment.String()
}

func (a *argumentAccumulator) writeLeaf(leaf pathSegment, argument partialArgument, valueJSON string) string {
	fragment := a.beginChild(leaf)

	if argument.StringValue != nil && argument.continues() {
		a.stringOpen = true

		return fragment + strings.TrimSuffix(valueJSON, `"`)
	}

	return fragment + valueJSON
}

// beginChild writes the separator and key that precede a new child of the
// innermost open container.
func (a *argumentAccumulator) beginChild(segment pathSegment) string {
	container := &a.open[len(a.open)-1]

	prefix := ""
	if container.childCount > 0 {
		prefix = ","
	}
	container.childCount++

	if segment.isIndex {
		return prefix
	}

	return prefix + encodeJSONString(segment.key) + ":"
}

// parsePath splits a path such as "recipe.ingredients[0].name" into
// segments.
func parsePath(rawPath string) []pathSegment {
	var segments []pathSegment

	for _, piece := range strings.Split(rawPath, ".") {
		bracket := strings.Index(piece, "[")
		if bracket == -1 {
			segments = append(segments, pathSegment{
				key: piece,
			})

			continue
		}

		if bracket > 0 {
			segments = append(segments, pathSegment{
				key: piece[:bracket],
			})
		}

		for _, match := range pathIndexPattern.FindAllStringSubmatch(piece, -1) {
			index, err := strconv.Atoi(match[1])
			if err != nil {
				continue
			}

			segments = append(segments, pathSegment{
				index:   index,
				isIndex: true,
			})
		}
	}

	return segments
}

// resolvePartialValue returns the first value set on the argument.
func resolvePartialValue(argument partialArgument) (*argumentValue, bool) {
	switch {
	case argument.StringValue != nil:
		return newStringValue(*argument.StringValue), true
	case argument.NumberValue != nil:
		// The number was decoded from JSON, so it is finite and encodes.
		encoded, _ := json.Marshal(*argument.NumberValue)

		return newScalarValue(string(encoded)), true
	case argument.BoolValue != nil:
		return newScalarValue(strconv.FormatBool(*argument.BoolValue)), true
	case argument.NullValue != nil:
		return newScalarValue("null"), true
	default:
		return nil, false
	}
}

type valueKind int

const (
	objectKind valueKind = iota
	arrayKind
	stringKind
	scalarKind
)

// argumentValue is a JSON value that keeps object keys in insertion order,
// matching the order of the streamed text.
type argumentValue struct {
	kind valueKind

	keys   []string
	fields map[string]*argumentValue
	items  []*argumentValue

	// text is the content of a string; scalar is the JSON of a number,
	// boolean or null.
	text   string
	scalar string
}

func newObjectValue() *argumentValue {
	return &argumentValue{
		kind:   objectKind,
		fields: make(map[string]*argumentValue),
	}
}

func newArrayValue() *argumentValue {
	return &argumentValue{
		kind: arrayKind,
	}
}

func newStringValue(text string) *argumentValue {
	return &argumentValue{
		kind: stringKind,
		text: text,
	}
}

func newScalarValue(scalar string) *argumentValue {
	return &argumentValue{
		kind:   scalarKind,
		scalar: scalar,
	}
}

// appendText continues a string value. A non-string value becomes a string
// of its JSON followed by the text.
func (v *argumentValue) appendText(text string) {
	if v.kind != stringKind {
		v.text = v.encode()
		v.kind = stringKind
	}

	v.text += text
}

// lookup returns the value at segments, or nil.
func (v *argumentValue) lookup(segments []pathSegment) *argumentValue {
	current := v
	for _, segment := range segments {
		current = current.child(segment)
		if current == nil {
			return nil
		}
	}

	return current
}

// store sets the value at segments, creating containers on the way.
func (v *argumentValue) store(segments []pathSegment, value *argumentValue) {
	current := v

	for position, segment := range segments[:len(segments)-1] {
		next := current.child(segment)
		if !next.isContainer() {
			next = newObjectValue()
			if segments[position+1].isIndex {
				next = newArrayValue()
			}

			current.setChild(segment, next)
		}

		current = next
	}

	current.setChild(segments[len(segments)-1], value)
}

func (v *argumentValue) isContainer() bool {
	return v != nil && (v.kind == objectKind || v.kind == arrayKind)
}

func (v *argumentValue) child(segment pathSegment) *argumentValue {
	switch v.kind {
	case objectKind:
		return v.fields[segment.objectKey()]
	case arrayKind:
		if !segment.isIndex || segment.index >= len(v.items) {
			return nil
		}

		return v.items[segment.index]
	default:
		return nil
	}
}

func (v *argumentValue) setChild(segment pathSegment, value *argumentValue) {
	switch v.kind {
	case objectKind:
		key := segment.objectKey()
		if _, exists := v.fields[key]; !exists {
			v.keys = append(v.keys, key)
		}

		v.fields[key] = value

	case arrayKind:
		if !segment.isIndex {
			return
		}

		for len(v.items) <= segment.index {
			v.items = append(v.items, nil)
		}

		v.items[segment.index] = value
	}
}

func (s pathSegment) objectKey() string {
	if s.isIndex {
		return strconv.Itoa(s.index)
	}

	return s.key
}

// encode returns the value as compact JSON. Missing array items encode as
// null.
func (v *argumentValue) encode() string {
	if v == nil {
		return "null"
	}

	switch v.kind {
	case objectKind:
		members := make([]string, 0, len(v.keys))
		for _, key := range v.keys {
			members = append(members, encodeJSONString(key)+":"+v.fields[key].encode())
		}

		return "{" + strings.Join(members, ",") + "}"

	case arrayKind:
		items := make([]string, 0, len(v.items))
		for _, item := range v.items {
			items = append(items, item.encode())
		}

		return "[" + strings.Join(items, ",") + "]"

	case stringKind:
		return encodeJSONString(v.text)

	default:
		return v.scalar
	}
}

// encodeJSONString encodes text as a JSON string without HTML escaping.
func encodeJSONString(text string) string {
	var buffer bytes.Buffer

	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)

	// Encoding a string cannot fail.
	_ = encoder.Encode(text)

	return strings.TrimSuffix(buffer.String(), "\n")
}

// escapeStringContent returns text escaped for use inside a JSON string.
func escapeStringContent(text string) string {
	encoded := encodeJSONString(text)

	return encoded[1 : len(encoded)-1]
}
