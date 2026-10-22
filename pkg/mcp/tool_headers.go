package mcp

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// headerValueType is the JSON Schema type of a tool argument mirrored into
// a header.
type headerValueType string

const (
	headerValueBoolean headerValueType = "boolean"
	headerValueInteger headerValueType = "integer"
	headerValueString  headerValueType = "string"
)

// toolHeaderBinding mirrors the tool argument at path into the header
// Mcp-Param-<headerName>.
type toolHeaderBinding struct {
	headerName string
	path       []string
	valueType  headerValueType
}

const toolHeaderKeyword = "x-mcp-header"

var (
	httpTokenPattern      = regexp.MustCompile("^[!#$%&'*+\\-.^_`|~0-9A-Za-z]+$")
	base64SentinelPattern = regexp.MustCompile(`^=\?base64\?.*\?=$`)
)

// encodeHeaderValue returns value unchanged when it is printable ASCII
// without surrounding whitespace, and as =?base64?...?= otherwise.
func encodeHeaderValue(value string) string {
	if isPlainHeaderValue(value) {
		return value
	}

	return "=?base64?" + base64.StdEncoding.EncodeToString([]byte(value)) + "?="
}

func isPlainHeaderValue(value string) bool {
	for _, character := range value {
		if character != '\t' && (character < 0x20 || character > 0x7e) {
			return false
		}
	}

	return strings.TrimSpace(value) == value && !base64SentinelPattern.MatchString(value)
}

// toolHeaderBindings finds the properties of an input schema annotated with
// x-mcp-header. Annotations must sit on statically reachable boolean,
// integer or string properties and name unique HTTP tokens.
func toolHeaderBindings(inputSchema json.RawMessage) ([]toolHeaderBinding, error) {
	decoder := json.NewDecoder(bytes.NewReader(inputSchema))
	decoder.UseNumber()

	var schemaValue any
	if err := decoder.Decode(&schemaValue); err != nil {
		return nil, fmt.Errorf("decode inputSchema: %w", err)
	}

	if _, ok := schemaValue.(map[string]any); !ok {
		return nil, errors.New("inputSchema must be a JSON Schema object")
	}

	visitor := headerBindingVisitor{
		headerNames: make(map[string]bool),
	}

	visitor.visit(schemaValue, nil, true)

	if visitor.err != nil {
		return nil, visitor.err
	}

	return visitor.bindings, nil
}

// headerBindingVisitor walks a schema collecting x-mcp-header bindings. It
// stops at the first invalid annotation.
type headerBindingVisitor struct {
	bindings    []toolHeaderBinding
	headerNames map[string]bool
	err         error
}

func (v *headerBindingVisitor) visit(value any, path []string, staticallyReachable bool) {
	record, ok := value.(map[string]any)
	if v.err != nil || !ok {
		return
	}

	if headerName, annotated := record[toolHeaderKeyword]; annotated {
		v.bind(record, headerName, path, staticallyReachable)
	}

	for _, key := range sortedKeys(record) {
		child := record[key]

		switch key {
		case toolHeaderKeyword:
			continue
		case "properties":
			v.visitProperties(child, path, staticallyReachable)
		default:
			v.visit(child, path, false)
		}
	}
}

func (v *headerBindingVisitor) visitProperties(value any, path []string, staticallyReachable bool) {
	properties, ok := value.(map[string]any)
	if !ok {
		return
	}

	for _, name := range sortedKeys(properties) {
		v.visit(properties[name], append(slices.Clone(path), name), staticallyReachable)
	}
}

func (v *headerBindingVisitor) bind(record map[string]any, headerNameValue any, path []string, staticallyReachable bool) {
	if !staticallyReachable || len(path) == 0 {
		v.err = errors.New("x-mcp-header is not on a statically reachable property")

		return
	}

	headerName, ok := headerNameValue.(string)
	if !ok || !httpTokenPattern.MatchString(headerName) {
		v.err = errors.New("x-mcp-header must be a non-empty HTTP token")

		return
	}

	normalizedName := strings.ToLower(headerName)
	if v.headerNames[normalizedName] {
		v.err = fmt.Errorf("x-mcp-header value %q is not unique", headerName)

		return
	}

	valueType, ok := record["type"].(string)
	if !ok || !isHeaderValueType(headerValueType(valueType)) {
		v.err = errors.New("x-mcp-header can only annotate boolean, integer, or string properties")

		return
	}

	v.headerNames[normalizedName] = true
	v.bindings = append(v.bindings, toolHeaderBinding{
		headerName: headerName,
		path:       path,
		valueType:  headerValueType(valueType),
	})
}

func isHeaderValueType(valueType headerValueType) bool {
	return valueType == headerValueBoolean || valueType == headerValueInteger || valueType == headerValueString
}

func sortedKeys(record map[string]any) []string {
	keys := make([]string, 0, len(record))
	for key := range record {
		keys = append(keys, key)
	}

	slices.Sort(keys)

	return keys
}

// createToolHeaders returns the Mcp-Param headers for the arguments of a
// call. Absent arguments are skipped; arguments of the wrong type fail.
func createToolHeaders(bindings []toolHeaderBinding, arguments map[string]any) (http.Header, error) {
	headers := make(http.Header)

	for _, binding := range bindings {
		value := valueAtPath(arguments, binding.path)
		if value == nil {
			continue
		}

		text, ok := headerText(value, binding.valueType)
		if !ok {
			return nil, fmt.Errorf("tool argument %q does not match its x-mcp-header type", strings.Join(binding.path, "."))
		}

		headers.Set("Mcp-Param-"+binding.headerName, encodeHeaderValue(text))
	}

	return headers, nil
}

func valueAtPath(arguments map[string]any, path []string) any {
	var current any = arguments

	for _, segment := range path {
		record, ok := current.(map[string]any)
		if !ok {
			return nil
		}

		current = record[segment]
	}

	return current
}

// headerText formats an argument of the bound type as header text.
func headerText(value any, valueType headerValueType) (string, bool) {
	switch valueType {
	case headerValueString:
		text, ok := value.(string)

		return text, ok
	case headerValueBoolean:
		flag, ok := value.(bool)

		return strconv.FormatBool(flag), ok
	case headerValueInteger:
		number, ok := value.(json.Number)
		if !ok {
			return "", false
		}

		integer, err := parseInteger(number.String())
		if err != nil || integer > maxSafeInteger || integer < -maxSafeInteger {
			return "", false
		}

		return strconv.FormatInt(integer, 10), true
	default:
		return "", false
	}
}
