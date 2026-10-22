// Package schema generates JSON Schemas from Go types and validates JSON
// values against schemas.
package schema

import (
	"encoding"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"
)

// ErrUnsupportedType reports a Go type whose JSON form cannot be inferred,
// such as a type with custom marshaling or a recursive type. Supply an
// explicit schema for these.
var ErrUnsupportedType = errors.New("cannot infer JSON schema")

var (
	timeType       = reflect.TypeFor[time.Time]()
	jsonNumberType = reflect.TypeFor[json.Number]()
	rawMessageType = reflect.TypeFor[json.RawMessage]()
)

// Generate returns the JSON Schema of T. Struct fields follow their json tags;
// fields tagged omitempty or omitzero are optional, pointers are nullable, and
// objects reject additional properties. The jsonschema tag adds keywords:
//
//	`jsonschema:"description=The city name"`
//	`jsonschema:"enum=celsius|fahrenheit"`
func Generate[T any]() (json.RawMessage, error) {
	return GenerateWithOptions[T](GenerateOptions{})
}

// GenerateOptions configure GenerateWithOptions.
type GenerateOptions struct {
	// NullableCollections makes the schemas of slices, maps and byte slices
	// accept null, because encoding/json writes nil ones as null. Use it for
	// values that Go encodes, such as tool outputs.
	NullableCollections bool
}

// GenerateWithOptions returns the JSON Schema of T like Generate, adjusted
// by options.
func GenerateWithOptions[T any](options GenerateOptions) (json.RawMessage, error) {
	generator := generator{
		visiting:            make(map[reflect.Type]bool),
		nullableCollections: options.NullableCollections,
	}

	root, err := generator.generate(reflect.TypeFor[T]())
	if err != nil {
		return nil, err
	}

	encoded, err := json.Marshal(root)
	if err != nil {
		return nil, fmt.Errorf("encode schema: %w", err)
	}

	return encoded, nil
}

type generator struct {
	visiting            map[reflect.Type]bool
	nullableCollections bool
}

func (g *generator) generate(t reflect.Type) (*node, error) {
	switch t {
	case timeType:
		return &node{
			Type:   "string",
			Format: "date-time",
		}, nil

	case jsonNumberType:
		return &node{
			Type: "number",
		}, nil

	case rawMessageType:
		return &node{}, nil
	}

	if hasCustomEncoding(t) {
		return nil, fmt.Errorf("%w: %s has custom JSON encoding", ErrUnsupportedType, t)
	}

	if g.visiting[t] {
		return nil, fmt.Errorf("%w: %s is recursive", ErrUnsupportedType, t)
	}

	g.visiting[t] = true
	defer delete(g.visiting, t)

	return g.generateKind(t)
}

func (g *generator) generateKind(t reflect.Type) (*node, error) {
	switch t.Kind() {
	case reflect.String:
		return &node{
			Type: "string",
		}, nil

	case reflect.Bool:
		return &node{
			Type: "boolean",
		}, nil

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &node{
			Type: "integer",
		}, nil

	case reflect.Float32, reflect.Float64:
		return &node{
			Type: "number",
		}, nil

	case reflect.Pointer:
		return g.generateNullable(t.Elem())

	case reflect.Slice:
		return g.generateSlice(t)

	case reflect.Array:
		return g.generateArray(t)

	case reflect.Map:
		return g.generateMap(t)

	case reflect.Struct:
		return g.generateStruct(t)

	default:
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedType, t)
	}
}

func (g *generator) generateNullable(element reflect.Type) (*node, error) {
	inner, err := g.generate(element)
	if err != nil {
		return nil, err
	}

	return &node{
		AnyOf: []*node{
			inner,
			{
				Type: "null",
			},
		},
	}, nil
}

func (g *generator) generateSlice(t reflect.Type) (*node, error) {
	// encoding/json writes []byte as a base64 string.
	if t.Elem().Kind() == reflect.Uint8 {
		return &node{
			Type:            "string",
			ContentEncoding: "base64",
			nullable:        g.nullableCollections,
		}, nil
	}

	items, err := g.generate(t.Elem())
	if err != nil {
		return nil, err
	}

	return &node{
		Type:     "array",
		Items:    items,
		nullable: g.nullableCollections,
	}, nil
}

func (g *generator) generateArray(t reflect.Type) (*node, error) {
	items, err := g.generate(t.Elem())
	if err != nil {
		return nil, err
	}

	return &node{
		Type:     "array",
		Items:    items,
		MinItems: new(t.Len()),
		MaxItems: new(t.Len()),
	}, nil
}

func (g *generator) generateMap(t reflect.Type) (*node, error) {
	if t.Key().Kind() != reflect.String || hasCustomEncoding(t.Key()) {
		return nil, fmt.Errorf("%w: map key %s is not a plain string", ErrUnsupportedType, t.Key())
	}

	values, err := g.generate(t.Elem())
	if err != nil {
		return nil, err
	}

	return &node{
		Type: "object",
		AdditionalProperties: &additionalProperties{
			schema: values,
		},
		nullable: g.nullableCollections,
	}, nil
}

func (g *generator) generateStruct(t reflect.Type) (*node, error) {
	properties := make(map[string]*node)
	var required []string

	for index := range t.NumField() {
		field, ok, err := parseField(t.Field(index))
		if err != nil {
			return nil, fmt.Errorf("field %s.%s: %w", t, t.Field(index).Name, err)
		}

		if !ok {
			continue
		}

		if _, duplicate := properties[field.name]; duplicate {
			return nil, fmt.Errorf("%w: %s has duplicate JSON field %q", ErrUnsupportedType, t, field.name)
		}

		fieldSchema, err := g.generate(field.goType)
		if err != nil {
			return nil, fmt.Errorf("field %s.%s: %w", t, t.Field(index).Name, err)
		}

		fieldSchema.Description = field.description
		fieldSchema.Enum = field.enum
		properties[field.name] = fieldSchema

		if !field.optional {
			required = append(required, field.name)
		}
	}

	return &node{
		Type:       "object",
		Properties: &properties,
		Required:   required,
		AdditionalProperties: &additionalProperties{
			allowed: new(false),
		},
	}, nil
}

func hasCustomEncoding(t reflect.Type) bool {
	encoders := []reflect.Type{
		reflect.TypeFor[json.Marshaler](),
		reflect.TypeFor[json.Unmarshaler](),
		reflect.TypeFor[encoding.TextMarshaler](),
		reflect.TypeFor[encoding.TextUnmarshaler](),
	}

	for _, encoder := range encoders {
		if t.Implements(encoder) {
			return true
		}

		if t.Kind() != reflect.Pointer && reflect.PointerTo(t).Implements(encoder) {
			return true
		}
	}

	return false
}

// structField is a struct field as it appears in JSON.
type structField struct {
	name        string
	goType      reflect.Type
	optional    bool
	description *string
	enum        []string
}

// parseField reads a field's json and jsonschema tags. It reports false for
// fields that are not encoded.
func parseField(field reflect.StructField) (structField, bool, error) {
	// As in encoding/json, the tag "-" skips the field, while "-," names it
	// "-".
	tag := field.Tag.Get("json")
	if tag == "-" || !field.IsExported() {
		return structField{}, false, nil
	}

	name, options, _ := strings.Cut(tag, ",")

	if field.Anonymous {
		return structField{}, false, fmt.Errorf("%w: embedded fields need an explicit schema", ErrUnsupportedType)
	}

	if name == "" {
		name = field.Name
	}

	parsed := structField{
		name:   name,
		goType: field.Type,
	}

	for option := range strings.SplitSeq(options, ",") {
		switch option {
		case "":
			// A tag without options yields one empty option.

		case "omitempty", "omitzero":
			parsed.optional = true

		default:
			return structField{}, false, fmt.Errorf("%w: json option %q", ErrUnsupportedType, option)
		}
	}

	if err := parsed.applySchemaTag(field.Tag.Get("jsonschema")); err != nil {
		return structField{}, false, err
	}

	return parsed, true, nil
}

func (f *structField) applySchemaTag(tag string) error {
	for keyword := range strings.SplitSeq(tag, ",") {
		keyword = strings.TrimSpace(keyword)

		if description, ok := strings.CutPrefix(keyword, "description="); ok {
			f.description = new(description)

			continue
		}

		if values, ok := strings.CutPrefix(keyword, "enum="); ok {
			if f.goType.Kind() != reflect.String {
				return fmt.Errorf("%w: enum requires a string field", ErrUnsupportedType)
			}

			f.enum = strings.Split(values, "|")

			continue
		}

		if keyword != "" {
			return fmt.Errorf("%w: jsonschema keyword %q", ErrUnsupportedType, keyword)
		}
	}

	return nil
}
