// Package provideropts applies provider-specific request options to
// normalized modality requests.
package provideropts

import (
	"encoding/json"
	"fmt"
	"mime/multipart"
	"strconv"

	"github.com/aarock1234/ai/pkg/model"
)

// JSONBody marshals a JSON request body and merges provider-specific
// fields from ProviderOptions. It rejects keys that collide with
// normalized request fields.
func JSONBody(base any, provider string, opts model.ProviderOptions, reservedFields ...string) ([]byte, error) {
	body, err := json.Marshal(base)
	if err != nil {
		return nil, fmt.Errorf("marshalling request body: %w", err)
	}

	optionSet := optionSetForProvider(provider, opts)
	if len(optionSet) == 0 {
		return body, nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("decoding request body: %w", err)
	}

	reserved := reservedFieldSet(reservedFields)
	for key, value := range optionSet {
		if _, ok := reserved[key]; ok {
			return nil, fmt.Errorf("%s provider option %q conflicts with a normalized request field", provider, key)
		}
		if !json.Valid(value) {
			return nil, fmt.Errorf("%s provider option %q must be valid JSON", provider, key)
		}

		payload[key] = append(json.RawMessage(nil), value...)
	}

	merged, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshalling merged request body: %w", err)
	}

	return merged, nil
}

// WriteMultipartFields writes provider-specific multipart form fields
// from ProviderOptions. It rejects keys that collide with normalized
// request fields.
func WriteMultipartFields(writer *multipart.Writer, provider string, opts model.ProviderOptions, reservedFields ...string) error {
	optionSet := optionSetForProvider(provider, opts)
	if len(optionSet) == 0 {
		return nil
	}

	reserved := reservedFieldSet(reservedFields)
	for key, value := range optionSet {
		if _, ok := reserved[key]; ok {
			return fmt.Errorf("%s provider option %q conflicts with a normalized request field", provider, key)
		}
		if !json.Valid(value) {
			return fmt.Errorf("%s provider option %q must be valid JSON", provider, key)
		}
		if err := writeMultipartValue(writer, key, value); err != nil {
			return fmt.Errorf("writing %s provider option %q: %w", provider, key, err)
		}
	}

	return nil
}

func optionSetForProvider(provider string, opts model.ProviderOptions) model.ProviderOptionSet {
	if len(opts) == 0 {
		return nil
	}

	optionSet, ok := opts[provider]
	if !ok {
		return nil
	}

	return optionSet.Clone()
}

func reservedFieldSet(fields []string) map[string]struct{} {
	if len(fields) == 0 {
		return nil
	}

	reserved := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		reserved[field] = struct{}{}
	}

	return reserved
}

func writeMultipartValue(writer *multipart.Writer, key string, raw json.RawMessage) error {
	var decoded any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return fmt.Errorf("decoding field value: %w", err)
	}

	return writeMultipartField(writer, key, decoded)
}

func writeMultipartField(writer *multipart.Writer, key string, value any) error {
	switch v := value.(type) {
	case nil:
		return fmt.Errorf("null values are not supported")
	case string:
		return writer.WriteField(key, v)
	case bool:
		return writer.WriteField(key, strconv.FormatBool(v))
	case float64:
		return writer.WriteField(key, strconv.FormatFloat(v, 'f', -1, 64))
	case []any:
		for _, item := range v {
			if err := writeMultipartField(writer, key, item); err != nil {
				return err
			}
		}

		return nil
	case map[string]any:
		encoded, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("encoding object value: %w", err)
		}

		return writer.WriteField(key, string(encoded))
	default:
		return fmt.Errorf("unsupported value type %T", value)
	}
}
