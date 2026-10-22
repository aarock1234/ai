package codemode

import (
	"bytes"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"unicode/utf8"
)

const (
	maxSchemaDepth            = 8
	maxCompactObjectTypeWidth = 120
)

// jsonObject is a decoded JSON object that keeps its member order, which
// the rendered declarations and examples follow. Its values are nil, bool,
// json.Number, string, []any or *jsonObject.
type jsonObject struct {
	keys   []string
	values map[string]any
}

func newJSONObject() *jsonObject {
	return &jsonObject{
		values: make(map[string]any),
	}
}

func (o *jsonObject) get(key string) (any, bool) {
	value, ok := o.values[key]

	return value, ok
}

func (o *jsonObject) set(key string, value any) {
	if _, exists := o.values[key]; !exists {
		o.keys = append(o.keys, key)
	}

	o.values[key] = value
}

// with returns a copy of the object with one member replaced.
func (o *jsonObject) with(key string, value any) *jsonObject {
	copied := &jsonObject{
		keys:   slices.Clone(o.keys),
		values: make(map[string]any, len(o.values)),
	}

	for _, existing := range o.keys {
		copied.values[existing] = o.values[existing]
	}

	copied.set(key, value)

	return copied
}

// MarshalJSON implements json.Marshaler, keeping the member order.
func (o *jsonObject) MarshalJSON() ([]byte, error) {
	var buffer bytes.Buffer

	buffer.WriteByte('{')

	for index, key := range o.keys {
		if index > 0 {
			buffer.WriteByte(',')
		}

		buffer.WriteString(encodeSchemaValue(key))
		buffer.WriteByte(':')
		buffer.WriteString(encodeSchemaValue(o.values[key]))
	}

	buffer.WriteByte('}')

	return buffer.Bytes(), nil
}

// decodeSchema decodes a JSON Schema object, keeping member order.
func decodeSchema(raw json.RawMessage) (*jsonObject, bool) {
	value, ok := decodeSchemaValue(raw)
	if !ok {
		return nil, false
	}

	object, ok := value.(*jsonObject)

	return object, ok
}

// decodeSchemaValue decodes any JSON value, keeping object member order.
func decodeSchemaValue(raw []byte) (any, bool) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, false
	}

	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()

	value, err := decodeOrderedValue(decoder)
	if err != nil {
		return nil, false
	}

	return value, true
}

func decodeOrderedValue(decoder *json.Decoder) (any, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}

	delimiter, isDelimiter := token.(json.Delim)
	if !isDelimiter {
		return token, nil
	}

	switch delimiter {
	case '{':
		object := newJSONObject()

		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return nil, err
			}

			key, ok := keyToken.(string)
			if !ok {
				return nil, errors.New("object key is not a string")
			}

			value, err := decodeOrderedValue(decoder)
			if err != nil {
				return nil, err
			}

			object.set(key, value)
		}

		if _, err := decoder.Token(); err != nil {
			return nil, err
		}

		return object, nil

	case '[':
		items := []any{}

		for decoder.More() {
			item, err := decodeOrderedValue(decoder)
			if err != nil {
				return nil, err
			}

			items = append(items, item)
		}

		if _, err := decoder.Token(); err != nil {
			return nil, err
		}

		return items, nil

	default:
		return nil, errors.New("unexpected JSON delimiter")
	}
}

// encodeSchemaValue encodes a decoded value as JSON.stringify would,
// without escaping HTML characters.
func encodeSchemaValue(value any) string {
	var buffer bytes.Buffer

	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)

	if err := encoder.Encode(value); err != nil {
		return "null"
	}

	return strings.TrimSuffix(buffer.String(), "\n")
}

// schemaContext tracks a traversal of one schema. References are expanded
// at most once per traversal, which also stops recursive schemas.
type schemaContext struct {
	root     *jsonObject
	seenRefs map[string]bool
	depth    int
}

func newSchemaContext(root *jsonObject) schemaContext {
	return schemaContext{
		root:     root,
		seenRefs: make(map[string]bool),
	}
}

func (c schemaContext) next() schemaContext {
	c.depth++

	return c
}

// schemaToType renders a JSON Schema as a TypeScript type.
func schemaToType(schema *jsonObject) string {
	return renderSchemaType(schema, newSchemaContext(schema))
}

func renderSchemaType(schema *jsonObject, ctx schemaContext) string {
	if ctx.depth > maxSchemaDepth {
		return "unknown"
	}

	if ref, ok := stringMember(schema, "$ref"); ok {
		resolved := resolveRef(ref, ctx)
		if resolved == nil {
			return "unknown"
		}

		return renderSchemaType(resolved, ctx.next())
	}

	if constant, ok := schema.get("const"); ok {
		return literalType(constant)
	}

	if values, ok := arrayMember(schema, "enum"); ok {
		literals := make([]string, len(values))
		for index, value := range values {
			literals[index] = literalType(value)
		}

		return union(literals)
	}

	if parts, ok := schemaArrayMember(schema, "oneOf"); ok {
		return union(renderSchemaTypes(parts, ctx.next()))
	}

	if parts, ok := schemaArrayMember(schema, "anyOf"); ok {
		return union(renderSchemaTypes(parts, ctx.next()))
	}

	if parts, ok := schemaArrayMember(schema, "allOf"); ok {
		return strings.Join(renderSchemaTypes(parts, ctx.next()), " & ")
	}

	schemaType, _ := schema.get("type")

	if types, ok := schemaType.([]any); ok {
		rendered := make([]string, len(types))
		for index, part := range types {
			rendered[index] = renderSchemaType(schema.with("type", part), ctx)
		}

		return union(rendered)
	}

	if schemaType == "object" || hasObjectMember(schema, "properties") {
		return renderObjectType(schema, ctx)
	}

	if _, hasItems := schema.get("items"); schemaType == "array" || hasItems {
		return renderArrayType(schema, ctx)
	}

	switch schemaType {
	case "string":
		return "string"

	case "number", "integer":
		return "number"

	case "boolean":
		return "boolean"

	case "null":
		return "null"

	default:
		return "unknown"
	}
}

func renderSchemaTypes(schemas []*jsonObject, ctx schemaContext) []string {
	rendered := make([]string, len(schemas))
	for index, schema := range schemas {
		rendered[index] = renderSchemaType(schema, ctx)
	}

	return rendered
}

func renderObjectType(schema *jsonObject, ctx schemaContext) string {
	properties := objectProperties(schema)
	required := requiredProperties(schema)
	additional, _ := schema.get("additionalProperties")

	if len(properties.keys) == 0 {
		if additionalSchema, ok := additional.(*jsonObject); ok {
			return "Record<string, " + renderSchemaType(additionalSchema, ctx.next()) + ">"
		}

		if additional == true {
			return "Record<string, unknown>"
		}

		return "{}"
	}

	var lines []string

	for _, name := range properties.keys {
		property := properties.values[name].(*jsonObject)

		if description, ok := stringMember(property, "description"); ok {
			lines = append(lines, "  /** "+toComment(description)+" */")
		}

		optional := "?"
		if required[name] {
			optional = ""
		}

		lines = append(lines, "  "+formatObjectKey(name)+optional+": "+
			indentType(renderSchemaType(property, ctx.next()), 2)+";")
	}

	if additionalSchema, ok := additional.(*jsonObject); ok {
		lines = append(lines, "  [key: string]: "+renderSchemaType(additionalSchema, ctx.next())+";")
	} else if additional == true {
		lines = append(lines, "  [key: string]: unknown;")
	}

	if compact, ok := compactObjectType(lines); ok {
		return compact
	}

	return "{\n" + strings.Join(lines, "\n") + "\n}"
}

func renderArrayType(schema *jsonObject, ctx schemaContext) string {
	items, _ := schema.get("items")

	switch typed := items.(type) {
	case []any:
		rendered := make([]string, len(typed))
		for index, item := range typed {
			rendered[index] = "unknown"
			if itemSchema, ok := item.(*jsonObject); ok {
				rendered[index] = renderSchemaType(itemSchema, ctx.next())
			}
		}

		return "[" + strings.Join(rendered, ", ") + "]"

	case *jsonObject:
		itemType := renderSchemaType(typed, ctx.next())
		if needsParentheses(itemType) || strings.Contains(itemType, "\n") {
			return "Array<" + itemType + ">"
		}

		return itemType + "[]"

	default:
		return "unknown[]"
	}
}

// sampleFromSchema returns an example value that matches a JSON Schema.
func sampleFromSchema(schema *jsonObject) any {
	return sampleSchema(schema, newSchemaContext(schema))
}

func sampleSchema(schema *jsonObject, ctx schemaContext) any {
	if ctx.depth > maxSchemaDepth {
		return nil
	}

	if defaultValue, ok := schema.get("default"); ok {
		return defaultValue
	}

	if ref, ok := stringMember(schema, "$ref"); ok {
		resolved := resolveRef(ref, ctx)
		if resolved == nil {
			return nil
		}

		return sampleSchema(resolved, ctx.next())
	}

	if constant, ok := schema.get("const"); ok {
		return constant
	}

	if values, ok := arrayMember(schema, "enum"); ok {
		if len(values) == 0 {
			return nil
		}

		return values[0]
	}

	if parts, ok := schemaArrayMember(schema, "oneOf"); ok {
		return sampleSchema(firstSchema(parts), ctx.next())
	}

	if parts, ok := schemaArrayMember(schema, "anyOf"); ok {
		return sampleSchema(firstSchema(parts), ctx.next())
	}

	schemaType, hasType := schema.get("type")

	if types, ok := schemaType.([]any); ok {
		return sampleSchema(schema.with("type", preferredSampleType(types)), ctx)
	}

	if schemaType == "object" || hasObjectMember(schema, "properties") {
		properties := objectProperties(schema)
		sample := newJSONObject()

		for _, name := range properties.keys {
			sample.set(name, sampleSchema(properties.values[name].(*jsonObject), ctx.next()))
		}

		return sample
	}

	if items, hasItems := schema.get("items"); schemaType == "array" || hasItems {
		if tuple, ok := items.([]any); ok && len(tuple) > 0 {
			items = tuple[0]
		}

		itemSchema, ok := items.(*jsonObject)
		if !ok {
			return []any{
				nil,
			}
		}

		return []any{
			sampleSchema(itemSchema, ctx.next()),
		}
	}

	switch schemaType {
	case "number", "integer":
		return json.Number("1")

	case "boolean":
		return true

	case "null":
		return nil
	}

	if schemaType == "string" || !hasType {
		return sampleString(schema)
	}

	return nil
}

func sampleString(schema *jsonObject) any {
	if examples, ok := arrayMember(schema, "examples"); ok && len(examples) > 0 {
		return examples[0]
	}

	format, _ := stringMember(schema, "format")

	switch format {
	case "uri", "url":
		return "https://example.com"

	case "date-time":
		return "2026-01-01T00:00:00.000Z"

	case "date":
		return "2026-01-01"

	default:
		return "string"
	}
}

// preferredSampleType picks the first non-null type of a type union.
func preferredSampleType(types []any) any {
	for _, part := range types {
		if part != "null" {
			return part
		}
	}

	if len(types) == 0 {
		return nil
	}

	return types[0]
}

func firstSchema(schemas []*jsonObject) *jsonObject {
	if len(schemas) == 0 {
		return newJSONObject()
	}

	return schemas[0]
}

// resolveRef resolves a local JSON pointer reference once per traversal.
func resolveRef(ref string, ctx schemaContext) *jsonObject {
	pointer, isLocal := strings.CutPrefix(ref, "#/")
	if !isLocal || ctx.seenRefs[ref] {
		return nil
	}

	ctx.seenRefs[ref] = true

	var current any = ctx.root

	for part := range strings.SplitSeq(pointer, "/") {
		object, ok := current.(*jsonObject)
		if !ok {
			return nil
		}

		part = strings.ReplaceAll(part, "~1", "/")
		part = strings.ReplaceAll(part, "~0", "~")

		current, _ = object.get(part)
	}

	resolved, _ := current.(*jsonObject)

	return resolved
}

// objectProperties returns the properties whose schemas are objects.
func objectProperties(schema *jsonObject) *jsonObject {
	filtered := newJSONObject()

	properties, ok := objectMember(schema, "properties")
	if !ok {
		return filtered
	}

	for _, name := range properties.keys {
		if property, ok := properties.values[name].(*jsonObject); ok {
			filtered.set(name, property)
		}
	}

	return filtered
}

func requiredProperties(schema *jsonObject) map[string]bool {
	required := make(map[string]bool)

	names, _ := arrayMember(schema, "required")
	for _, name := range names {
		if text, ok := name.(string); ok {
			required[text] = true
		}
	}

	return required
}

func literalType(value any) string {
	switch typed := value.(type) {
	case string:
		return encodeSchemaValue(typed)

	case json.Number:
		return typed.String()

	case bool:
		if typed {
			return "true"
		}

		return "false"

	case nil:
		return "null"

	default:
		return "unknown"
	}
}

// union joins distinct types in their first-seen order.
func union(types []string) string {
	var unique []string

	for _, candidate := range types {
		if !slices.Contains(unique, candidate) {
			unique = append(unique, candidate)
		}
	}

	if len(unique) == 0 {
		return "unknown"
	}

	return strings.Join(unique, " | ")
}

func indentType(rendered string, spaces int) string {
	return strings.ReplaceAll(rendered, "\n", "\n"+strings.Repeat(" ", spaces))
}

func needsParentheses(rendered string) bool {
	return strings.Contains(rendered, " | ") || strings.Contains(rendered, " & ")
}

// compactObjectType renders short object types without comments on one
// line.
func compactObjectType(lines []string) (string, bool) {
	fields := make([]string, len(lines))

	for index, line := range lines {
		if strings.Contains(line, "/**") {
			return "", false
		}

		fields[index] = strings.TrimSpace(line)
		if strings.Contains(fields[index], "\n") {
			return "", false
		}
	}

	compact := "{ " + strings.Join(fields, " ") + " }"
	if utf8.RuneCountInString(compact) > maxCompactObjectTypeWidth {
		return "", false
	}

	return compact, true
}

func stringMember(schema *jsonObject, key string) (string, bool) {
	value, _ := schema.get(key)
	text, ok := value.(string)

	return text, ok
}

func arrayMember(schema *jsonObject, key string) ([]any, bool) {
	value, _ := schema.get(key)
	items, ok := value.([]any)

	return items, ok
}

func objectMember(schema *jsonObject, key string) (*jsonObject, bool) {
	value, _ := schema.get(key)
	object, ok := value.(*jsonObject)

	return object, ok
}

func hasObjectMember(schema *jsonObject, key string) bool {
	_, ok := objectMember(schema, key)

	return ok
}

// schemaArrayMember returns an array member whose elements are all schema
// objects.
func schemaArrayMember(schema *jsonObject, key string) ([]*jsonObject, bool) {
	items, ok := arrayMember(schema, key)
	if !ok {
		return nil, false
	}

	schemas := make([]*jsonObject, len(items))

	for index, item := range items {
		itemSchema, isSchema := item.(*jsonObject)
		if !isSchema {
			return nil, false
		}

		schemas[index] = itemSchema
	}

	return schemas, true
}
