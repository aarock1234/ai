// Package schema generates JSON Schemas from Go types and validates JSON
// values against schemas.
package schema

import (
	"encoding"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
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

// Writer is who writes the values that a schema describes. Models and
// programs leave out optional values differently, so the schema of a field
// tagged omitempty or omitzero depends on it.
type Writer int

const (
	// ModelWriter describes values that a model writes, such as structured
	// outputs and tool inputs. The schema is valid for strict structured
	// output: every property is required, and fields tagged omitempty or
	// omitzero accept null instead of being left out. Decoding null leaves
	// the Go zero value, as leaving the field out would.
	ModelWriter Writer = iota

	// ProgramWriter describes values that encoding/json writes, such as tool
	// outputs. Fields tagged omitempty or omitzero may be left out, and nil
	// slices, maps and byte slices are null.
	ProgramWriter
)

// Generate returns the JSON Schema of values of T that a model writes. Struct
// fields follow their json tags, pointers are nullable, and objects reject
// additional properties. The jsonschema tag adds keywords separated by
// commas. A description comes last and runs to the end of the tag, so it may
// contain commas:
//
//	`jsonschema:"enum=celsius|fahrenheit"`
//	`jsonschema:"minLength=1,maxLength=256,description=The title, in title case"`
//	`jsonschema:"minimum=0,maximum=100"`
//	`jsonschema:"minItems=1,maxItems=25"`
func Generate[T any]() (json.RawMessage, error) {
	return GenerateWithOptions[T](GenerateOptions{})
}

// GenerateOptions configure GenerateWithOptions.
type GenerateOptions struct {
	// Writer is who writes the values. The zero value is ModelWriter.
	Writer Writer
}

// GenerateWithOptions returns the JSON Schema of T like Generate, adjusted
// by options.
func GenerateWithOptions[T any](options GenerateOptions) (json.RawMessage, error) {
	generator := generator{
		visiting: make(map[reflect.Type]bool),
		writer:   options.Writer,
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
	visiting map[reflect.Type]bool
	writer   Writer
}

// nullableCollections reports whether nil slices, maps and byte slices may
// be written as null, which encoding/json does.
func (g *generator) nullableCollections() bool {
	return g.writer == ProgramWriter
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
		inner, err := g.generate(t.Elem())
		if err != nil {
			return nil, err
		}

		return nullable(inner), nil

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

// nullable returns a schema that accepts null or a value of inner.
func nullable(inner *node) *node {
	return &node{
		AnyOf: []*node{
			inner,
			{
				Type: "null",
			},
		},
	}
}

func (g *generator) generateSlice(t reflect.Type) (*node, error) {
	// encoding/json writes []byte as a base64 string.
	if t.Elem().Kind() == reflect.Uint8 {
		return &node{
			Type:            "string",
			ContentEncoding: "base64",
			nullable:        g.nullableCollections(),
		}, nil
	}

	items, err := g.generate(t.Elem())
	if err != nil {
		return nil, err
	}

	return &node{
		Type:     "array",
		Items:    items,
		nullable: g.nullableCollections(),
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
		nullable: g.nullableCollections(),
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

		fieldSchema, err := g.generateField(field)
		if err != nil {
			return nil, fmt.Errorf("field %s.%s: %w", t, t.Field(index).Name, err)
		}

		properties[field.name] = fieldSchema

		if g.requires(field) {
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

// generateField returns the schema of a struct field. The tag's keywords
// constrain the value, so they apply beneath any null alternative, while the
// description covers the whole field.
func (g *generator) generateField(field structField) (*node, error) {
	valueType, isPointer := pointedType(field.goType)

	value, err := g.generate(valueType)
	if err != nil {
		return nil, err
	}

	for _, keyword := range field.keywords {
		if err := keyword.apply(value); err != nil {
			return nil, err
		}
	}

	fieldSchema := value
	if isPointer || g.writesNull(field) {
		fieldSchema = nullable(value)
	}

	fieldSchema.Description = field.description

	return fieldSchema, nil
}

// requires reports whether a field's property must be present. A program
// leaves optional fields out, while a model writes null for them.
func (g *generator) requires(field structField) bool {
	return g.writer == ModelWriter || !field.optional
}

// writesNull reports whether a field is written as null when it is empty,
// instead of being left out.
func (g *generator) writesNull(field structField) bool {
	return g.writer == ModelWriter && field.optional
}

// pointedType returns the type a chain of pointers points to, and whether t
// is a pointer at all.
func pointedType(t reflect.Type) (reflect.Type, bool) {
	isPointer := t.Kind() == reflect.Pointer

	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	return t, isPointer
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
