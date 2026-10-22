package googlevertex

import "github.com/aarock1234/ai/pkg/model"

// readOptions returns the options stored under the first of
// optionNamespaces that holds them.
func readOptions[T any](options model.ProviderOptions) (T, bool, error) {
	for _, namespace := range optionNamespaces {
		value, ok, err := model.ProviderValue[T](options, namespace)
		if err != nil || ok {
			return value, ok, err
		}
	}

	var zero T

	return zero, false, nil
}

// wrapMetadata stores value under both metadata namespaces.
func wrapMetadata(value any) model.ProviderMetadata {
	return model.ProviderMetadata{
		Namespace:       value,
		LegacyNamespace: value,
	}
}
