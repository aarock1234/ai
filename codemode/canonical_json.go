package codemode

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"

	"golang.org/x/text/collate"
	"golang.org/x/text/language"
)

// keyOrder orders the keys of a canonical JSON object.
type keyOrder func(left, right string) int

// codeModeCanonicalJSON encodes a value like upstream's code mode
// continuation signature does: object keys sorted with
// String.prototype.localeCompare at every level, strings and numbers as
// JSON.stringify writes them. A continuation stored by a JSON database
// that reorders keys still verifies, in either implementation.
func codeModeCanonicalJSON(value any) ([]byte, error) {
	collator := collate.New(language.Und)

	return canonicalJSON(value, collator.CompareString)
}

// runCanonicalJSON encodes a value like upstream's run continuation codec:
// object keys sorted by UTF-16 code units, as Array.prototype.toSorted
// orders them.
func runCanonicalJSON(value any) ([]byte, error) {
	return canonicalJSON(value, compareUTF16)
}

func canonicalJSON(value any, order keyOrder) ([]byte, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}

	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()

	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		return nil, err
	}

	var builder strings.Builder
	if err := writeCanonicalJSON(&builder, decoded, order); err != nil {
		return nil, err
	}

	return []byte(builder.String()), nil
}

func writeCanonicalJSON(builder *strings.Builder, value any, order keyOrder) error {
	switch typed := value.(type) {
	case map[string]any:
		builder.WriteByte('{')

		for index, key := range slices.SortedStableFunc(maps.Keys(typed), order) {
			if index > 0 {
				builder.WriteByte(',')
			}

			writeJSONString(builder, key)
			builder.WriteByte(':')

			if err := writeCanonicalJSON(builder, typed[key], order); err != nil {
				return err
			}
		}

		builder.WriteByte('}')

		return nil

	case []any:
		builder.WriteByte('[')

		for index, element := range typed {
			if index > 0 {
				builder.WriteByte(',')
			}

			if err := writeCanonicalJSON(builder, element, order); err != nil {
				return err
			}
		}

		builder.WriteByte(']')

		return nil

	case string:
		writeJSONString(builder, typed)

		return nil

	case json.Number:
		number, err := typed.Float64()
		if err != nil || math.IsInf(number, 0) {
			return fmt.Errorf("continuation contains a non-JSON number %s", typed)
		}

		builder.WriteString(formatJSNumber(number))

		return nil

	case bool:
		builder.WriteString(strconv.FormatBool(typed))

		return nil

	case nil:
		builder.WriteString("null")

		return nil

	default:
		return fmt.Errorf("continuation contains an unsupported value of type %T", typed)
	}
}

// writeJSONString writes a string as JSON.stringify does: only quotes,
// backslashes and control characters are escaped.
func writeJSONString(builder *strings.Builder, text string) {
	builder.WriteByte('"')

	for _, character := range text {
		switch character {
		case '"':
			builder.WriteString(`\"`)
		case '\\':
			builder.WriteString(`\\`)
		case '\b':
			builder.WriteString(`\b`)
		case '\f':
			builder.WriteString(`\f`)
		case '\n':
			builder.WriteString(`\n`)
		case '\r':
			builder.WriteString(`\r`)
		case '\t':
			builder.WriteString(`\t`)
		default:
			if character < 0x20 {
				fmt.Fprintf(builder, `\u%04x`, character)

				continue
			}

			builder.WriteRune(character)
		}
	}

	builder.WriteByte('"')
}

// formatJSNumber formats a finite number as JavaScript's Number to String
// conversion does.
func formatJSNumber(number float64) string {
	if number == 0 {
		return "0"
	}

	format := byte('f')
	if magnitude := math.Abs(number); magnitude < 1e-6 || magnitude >= 1e21 {
		format = 'e'
	}

	formatted := strconv.FormatFloat(number, format, -1, 64)

	// Go writes two-digit exponents, such as 1e-07; JavaScript writes 1e-7.
	mantissa, exponent, found := strings.Cut(formatted, "e")
	if !found {
		return formatted
	}

	sign := exponent[:1]
	digits := strings.TrimLeft(exponent[1:], "0")

	return mantissa + "e" + sign + digits
}

// compareUTF16 orders strings by their UTF-16 code units, as JavaScript's
// default sort does.
func compareUTF16(left, right string) int {
	for left != "" && right != "" {
		leftRune, leftSize := utf8.DecodeRuneInString(left)
		rightRune, rightSize := utf8.DecodeRuneInString(right)

		if leftRune != rightRune {
			return compareCodeUnits(leftRune, rightRune)
		}

		left = left[leftSize:]
		right = right[rightSize:]
	}

	return len(left) - len(right)
}

// compareCodeUnits compares the first UTF-16 code units of two different
// runes. Runes outside the Basic Multilingual Plane start with a high
// surrogate, which sorts before U+E000 through U+FFFF.
func compareCodeUnits(left, right rune) int {
	leftUnit := firstCodeUnit(left)
	rightUnit := firstCodeUnit(right)

	if leftUnit != rightUnit {
		return int(leftUnit) - int(rightUnit)
	}

	return int(left) - int(right)
}

func firstCodeUnit(character rune) rune {
	if high, _ := utf16.EncodeRune(character); high != utf8.RuneError {
		return high
	}

	return character
}
