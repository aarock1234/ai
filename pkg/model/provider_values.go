package model

import (
	"encoding/json"
	"fmt"
)

// ProviderOptions carries provider-specific input settings. Each key is a
// provider namespace, such as "anthropic", and each value is that provider's
// options struct, or any value that encodes to the same JSON object.
type ProviderOptions map[string]any

// ProviderMetadata carries provider-specific output data, keyed the same way
// as ProviderOptions. When output is sent back to the model in a later call,
// its metadata becomes that part's provider options.
type ProviderMetadata map[string]any

// Options returns the metadata as provider options for replaying content.
func (m ProviderMetadata) Options() ProviderOptions {
	return ProviderOptions(m)
}

// ProviderValue returns the value stored under provider as a T. A stored T or
// *T is returned directly; any other value is converted through JSON, which
// accepts maps and raw JSON from serialized messages. It reports false when
// the namespace is absent.
func ProviderValue[T any, Values ~map[string]any](values Values, provider string) (T, bool, error) {
	var zero T

	stored, ok := values[provider]
	if !ok || stored == nil {
		return zero, false, nil
	}

	switch value := stored.(type) {
	case *T:
		if value != nil {
			return *value, true, nil
		}

		return zero, false, nil

	case T:
		return value, true, nil
	}

	encoded, err := json.Marshal(stored)
	if err != nil {
		return zero, true, fmt.Errorf("%w: encode %s provider values: %w", ErrInvalidArgument, provider, err)
	}

	var converted T
	if err := json.Unmarshal(encoded, &converted); err != nil {
		return zero, true, fmt.Errorf("%w: decode %s provider values: %w", ErrInvalidArgument, provider, err)
	}

	return converted, true, nil
}
