package mcp

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

// FingerprintAppResource returns a stable SHA-256 digest, base64url without
// padding, over the resource's HTML, content security policy and
// permissions. Store a baseline on first load and compare later reads with
// DetectAppResourceDrift.
func FingerprintAppResource(resource AppResource) (string, error) {
	var csp, permissions any
	if resource.Meta != nil {
		if resource.Meta.CSP != nil {
			csp = resource.Meta.CSP
		}

		if resource.Meta.Permissions != nil {
			permissions = resource.Meta.Permissions
		}
	}

	canonical, err := canonicalJSON(map[string]any{
		"html":        resource.HTML,
		"csp":         csp,
		"permissions": permissions,
	})
	if err != nil {
		return "", fmt.Errorf("fingerprint mcp app resource: %w", err)
	}

	digest := sha256.Sum256([]byte(canonical))

	return base64.RawURLEncoding.EncodeToString(digest[:]), nil
}

// DetectAppResourceDrift reports whether a fingerprint from
// FingerprintAppResource differs from its baseline.
func DetectAppResourceDrift(current string, baseline string) bool {
	return current != baseline
}

// canonicalJSON encodes value as JSON with sorted object keys, formatted as
// JavaScript's JSON.stringify formats it, so that fingerprints match
// upstream's.
func canonicalJSON(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", err
	}

	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()

	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		return "", err
	}

	var builder strings.Builder
	if err := writeCanonical(&builder, decoded); err != nil {
		return "", err
	}

	return builder.String(), nil
}

func writeCanonical(builder *strings.Builder, value any) error {
	switch typed := value.(type) {
	case nil:
		builder.WriteString("null")
	case bool:
		builder.WriteString(strconv.FormatBool(typed))
	case json.Number:
		return writeCanonicalNumber(builder, typed)
	case string:
		writeCanonicalString(builder, typed)
	case []any:
		builder.WriteByte('[')
		for index, item := range typed {
			if index > 0 {
				builder.WriteByte(',')
			}

			if err := writeCanonical(builder, item); err != nil {
				return err
			}
		}
		builder.WriteByte(']')
	case map[string]any:
		return writeCanonicalObject(builder, typed)
	default:
		return fmt.Errorf("unexpected json value %T", value)
	}

	return nil
}

func writeCanonicalObject(builder *strings.Builder, object map[string]any) error {
	keys := make([]string, 0, len(object))
	for key := range object {
		keys = append(keys, key)
	}

	slices.Sort(keys)

	builder.WriteByte('{')
	for index, key := range keys {
		if index > 0 {
			builder.WriteByte(',')
		}

		writeCanonicalString(builder, key)
		builder.WriteByte(':')

		if err := writeCanonical(builder, object[key]); err != nil {
			return err
		}
	}
	builder.WriteByte('}')

	return nil
}

// writeCanonicalNumber formats a number as JavaScript does: integers below
// 1e21 without an exponent and other values in shortest form.
func writeCanonicalNumber(builder *strings.Builder, number json.Number) error {
	value, err := number.Float64()
	if err != nil {
		return err
	}

	if value == math.Trunc(value) && math.Abs(value) < 1e21 {
		builder.WriteString(strconv.FormatFloat(value, 'f', -1, 64))

		return nil
	}

	builder.WriteString(strconv.FormatFloat(value, 'g', -1, 64))

	return nil
}

// writeCanonicalString quotes text as JSON.stringify does: only quotes,
// backslashes and control characters are escaped.
func writeCanonicalString(builder *strings.Builder, text string) {
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
