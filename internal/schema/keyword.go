package schema

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// keyword is a jsonschema tag keyword that restricts a field's value: the
// JSON types it applies to and how its text is written into the schema.
type keyword struct {
	types []string
	write func(value *node, text string) error
}

// keywords are the supported jsonschema tag keywords, by name. The
// description is not listed because it describes the field, not its value.
var keywords = map[string]keyword{
	"enum": {
		types: []string{
			"string",
		},
		write: func(value *node, text string) error {
			value.Enum = strings.Split(text, "|")

			return nil
		},
	},
	"minLength": countKeyword("string", func(value *node, count int) {
		value.MinLength = &count
	}),
	"maxLength": countKeyword("string", func(value *node, count int) {
		value.MaxLength = &count
	}),
	"minItems": countKeyword("array", func(value *node, count int) {
		value.MinItems = &count
	}),
	"maxItems": countKeyword("array", func(value *node, count int) {
		value.MaxItems = &count
	}),
	"minimum": boundKeyword(func(value *node, bound float64) {
		value.Minimum = &bound
	}),
	"maximum": boundKeyword(func(value *node, bound float64) {
		value.Maximum = &bound
	}),
}

// countKeyword is a keyword whose text is a non-negative integer, such as a
// length or an item count.
func countKeyword(valueType string, set func(value *node, count int)) keyword {
	return keyword{
		types: []string{
			valueType,
		},
		write: func(value *node, text string) error {
			count, err := strconv.Atoi(text)
			if err != nil || count < 0 {
				return fmt.Errorf("%q is not a non-negative integer", text)
			}

			set(value, count)

			return nil
		},
	}
}

// boundKeyword is a keyword whose text is a numeric bound.
func boundKeyword(set func(value *node, bound float64)) keyword {
	return keyword{
		types: []string{
			"integer",
			"number",
		},
		write: func(value *node, text string) error {
			bound, err := strconv.ParseFloat(text, 64)
			if err != nil {
				return fmt.Errorf("%q is not a number", text)
			}

			set(value, bound)

			return nil
		},
	}
}

// tagKeyword is one "name=value" keyword of a jsonschema tag.
type tagKeyword struct {
	name string
	text string
}

// parseTagKeyword reads one keyword of a jsonschema tag. An empty keyword,
// as a trailing comma leaves, has an empty name.
func parseTagKeyword(text string) (tagKeyword, error) {
	if text == "" {
		return tagKeyword{}, nil
	}

	name, value, _ := strings.Cut(text, "=")

	if _, known := keywords[name]; !known {
		return tagKeyword{}, fmt.Errorf("%w: jsonschema keyword %q", ErrUnsupportedType, name)
	}

	return tagKeyword{
		name: name,
		text: value,
	}, nil
}

// apply writes the keyword into the schema of the value it restricts.
func (k tagKeyword) apply(value *node) error {
	definition := keywords[k.name]

	if !slices.Contains(definition.types, value.Type) {
		return fmt.Errorf(
			"%w: jsonschema %s applies to %s values, not %q",
			ErrUnsupportedType,
			k.name,
			strings.Join(definition.types, " and "),
			value.Type,
		)
	}

	if err := definition.write(value, k.text); err != nil {
		return fmt.Errorf("%w: jsonschema %s: %w", ErrUnsupportedType, k.name, err)
	}

	return nil
}
