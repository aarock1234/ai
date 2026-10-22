package devtools

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// The SDK's model and content types have no JSON encoding, so DevTools
// converts captured values into typed records in the JSON shape of the
// upstream SDK, which the viewer reads. Model messages, content and stream
// parts use the shared upstream encoder, with file contents in the tagged
// form of the provider specification and binary data as base64 strings, as
// upstream serializeForDevTools records media. Unset optional properties
// are omitted, like undefined properties.

// recordJSON encodes model values in the provider specification shape.
var recordJSON = wirejson.Encoder{
	FileData: wirejson.TaggedFileData,
}

// serialize encodes a record as JSON text.
func serialize(record any) (string, error) {
	encoded, err := json.Marshal(record)
	if err != nil {
		return "", fmt.Errorf("serialize devtools value: %w", err)
	}

	return string(encoded), nil
}

// serializeProviderOptions encodes provider options, returning nil when
// there are none.
func serializeProviderOptions(options model.ProviderOptions) (*string, error) {
	if options == nil {
		return nil, nil
	}

	encoded, err := serialize(options)
	if err != nil {
		return nil, err
	}

	return &encoded, nil
}

// rawJSON returns a JSON document as text, or nil when it is empty.
func rawJSON(document json.RawMessage) *string {
	if len(document) == 0 {
		return nil
	}

	return new(string(document))
}

// responseRecord is the recorded response of a model call.
type responseRecord struct {
	wirejson.ResponseMetadata

	Headers map[string]string `json:"headers,omitempty"`
	Body    json.RawMessage   `json:"body,omitempty"`
}

func newResponseRecord(response model.ResponseInfo) responseRecord {
	record := responseRecord{
		ResponseMetadata: wirejson.EncodeResponseMetadata(response.ResponseMetadata),
		Headers:          headerRecord(response.Headers),
	}

	if len(response.Body) > 0 {
		record.Body = wirejson.RawValue(response.Body)
	}

	return record
}

// headerRecord flattens headers into an object with lower case names and
// comma-separated values, like the upstream response headers.
func headerRecord(headers http.Header) map[string]string {
	if len(headers) == 0 {
		return nil
	}

	flattened := make(map[string]string, len(headers))

	for name, values := range headers {
		flattened[strings.ToLower(name)] = strings.Join(values, ", ")
	}

	return flattened
}

func toolChoiceRecord(choice *model.ToolChoice) *wirejson.ToolChoice {
	if choice == nil {
		return nil
	}

	return new(wirejson.EncodeToolChoice(*choice))
}

func responseFormatRecord(format *model.ResponseFormat) *wirejson.ResponseFormat {
	if format == nil {
		return nil
	}

	return new(wirejson.EncodeResponseFormat(*format))
}
