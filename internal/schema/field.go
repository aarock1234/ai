package schema

import (
	"fmt"
	"reflect"
	"strings"
)

// structField is a struct field as it appears in JSON.
type structField struct {
	name        string
	goType      reflect.Type
	optional    bool
	description *string
	keywords    []tagKeyword
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

	if err := parsed.readSchemaTag(field.Tag.Get("jsonschema")); err != nil {
		return structField{}, false, err
	}

	return parsed, true, nil
}

// readSchemaTag reads the comma-separated "name=value" keywords of a
// jsonschema tag. The description runs to the end of the tag, so it may
// contain commas.
func (f *structField) readSchemaTag(tag string) error {
	remaining := strings.TrimSpace(tag)

	for remaining != "" {
		if description, ok := strings.CutPrefix(remaining, "description="); ok {
			f.description = new(description)

			return nil
		}

		var text string
		text, remaining, _ = strings.Cut(remaining, ",")
		remaining = strings.TrimSpace(remaining)

		keyword, err := parseTagKeyword(strings.TrimSpace(text))
		if err != nil {
			return err
		}

		if keyword.name != "" {
			f.keywords = append(f.keywords, keyword)
		}
	}

	return nil
}
