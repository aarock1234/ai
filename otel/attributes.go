package otel

import (
	"encoding/json"
	"math"
	"net/http"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/ai"
)

// spanAttributes collects the attributes of one span. Inputs and outputs
// are only resolved when the call records them, so that unrecorded prompts
// and responses are never serialized. Missing optional values are skipped,
// and non-finite numbers are dropped because OpenTelemetry cannot export
// them.
type spanAttributes struct {
	metadata ai.TelemetryMetadata
	values   []attribute.KeyValue
}

func newSpanAttributes(metadata ai.TelemetryMetadata) *spanAttributes {
	return &spanAttributes{
		metadata: metadata,
	}
}

// list returns the collected attributes.
func (a *spanAttributes) list() []attribute.KeyValue {
	return a.values
}

// add appends attributes, dropping empty and non-finite values.
func (a *spanAttributes) add(values ...attribute.KeyValue) {
	a.values = append(a.values, sanitizeAttributes(values)...)
}

// setString records a string, skipping the empty string.
func (a *spanAttributes) setString(key, value string) {
	if value == "" {
		return
	}

	a.add(attribute.String(key, value))
}

// setBool records a boolean.
func (a *spanAttributes) setBool(key string, value bool) {
	a.add(attribute.Bool(key, value))
}

// setInt records an optional integer.
func (a *spanAttributes) setInt(key string, value *int) {
	if value == nil {
		return
	}

	a.add(attribute.Int(key, *value))
}

// setFloat records an optional number.
func (a *spanAttributes) setFloat(key string, value *float64) {
	if value == nil {
		return
	}

	a.add(attribute.Float64(key, *value))
}

// setSeconds records a duration in seconds.
func (a *spanAttributes) setSeconds(key string, duration time.Duration) {
	a.add(attribute.Float64(key, duration.Seconds()))
}

// setStrings records a list of strings, skipping a nil list.
func (a *spanAttributes) setStrings(key string, values []string) {
	if values == nil {
		return
	}

	a.add(attribute.StringSlice(key, values))
}

// setJSON records the JSON encoding of a value that is not an input or an
// output, skipping values that cannot be encoded.
func (a *spanAttributes) setJSON(key string, value any) {
	a.add(attribute.KeyValue{
		Key:   attribute.Key(key),
		Value: jsonValue(value),
	})
}

// setInput records an input value when the call records inputs. An empty
// resolved value is skipped.
func (a *spanAttributes) setInput(key string, resolve func() attribute.Value) {
	if !a.metadata.RecordInputs {
		return
	}

	a.add(attribute.KeyValue{
		Key:   attribute.Key(key),
		Value: resolve(),
	})
}

// setOutput records an output value when the call records outputs. An
// empty resolved value is skipped.
func (a *spanAttributes) setOutput(key string, resolve func() attribute.Value) {
	if !a.metadata.RecordOutputs {
		return
	}

	a.add(attribute.KeyValue{
		Key:   attribute.Key(key),
		Value: resolve(),
	})
}

// sanitizeAttributes drops attributes that OpenTelemetry cannot export:
// empty values and numbers that are not finite.
func sanitizeAttributes(values []attribute.KeyValue) []attribute.KeyValue {
	sanitized := make([]attribute.KeyValue, 0, len(values))

	for _, value := range values {
		if isExportable(value.Value) {
			sanitized = append(sanitized, value)
		}
	}

	return sanitized
}

func isExportable(value attribute.Value) bool {
	switch value.Type() {
	case attribute.EMPTY:
		return false

	case attribute.FLOAT64:
		return isFinite(value.AsFloat64())

	case attribute.FLOAT64SLICE:
		for _, number := range value.AsFloat64Slice() {
			if !isFinite(number) {
				return false
			}
		}

		return true

	default:
		return true
	}
}

func isFinite(number float64) bool {
	return !math.IsInf(number, 0) && !math.IsNaN(number)
}

// telemetryJSON encodes prompts and tool outputs in the shape upstream
// telemetry reports them, with file contents flattened to strings.
var telemetryJSON = wirejson.Encoder{
	FileData: wirejson.FlattenedFileData,
}

// jsonValue returns the JSON encoding of value as a string attribute, or an
// empty value when it cannot be encoded.
func jsonValue(value any) attribute.Value {
	encoded, err := json.Marshal(value)
	if err != nil {
		return attribute.Value{}
	}

	return attribute.StringValue(string(encoded))
}

// jsonListValue returns the JSON encoding of each value as a string slice
// attribute, or an empty value when one cannot be encoded.
func jsonListValue[Value any](values []Value) attribute.Value {
	encoded := make([]string, 0, len(values))

	for _, value := range values {
		item, err := json.Marshal(value)
		if err != nil {
			return attribute.Value{}
		}

		encoded = append(encoded, string(item))
	}

	return attribute.StringSliceValue(encoded)
}

// headerAttributes returns one ai.request.headers.* attribute per request
// header. Multiple values of a header are joined with commas.
func headerAttributes(headers http.Header) []attribute.KeyValue {
	attributes := make([]attribute.KeyValue, 0, len(headers))

	for key, values := range headers {
		attributes = append(attributes, attribute.String(
			"ai.request.headers."+key,
			strings.Join(values, ", "),
		))
	}

	return attributes
}
