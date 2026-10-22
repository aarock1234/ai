package acp

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// gatewayValues are the values profile value sources resolve to.
type gatewayValues struct {
	apiKey           string
	baseURL          string
	clientAppName    string
	clientAppVersion string
}

// gatewayLaunchEnvironment resolves the AI Gateway profile into the launch
// environment of the implementation. Values that are not strings are
// JSON-encoded. It returns an empty environment without AI Gateway routing.
func gatewayLaunchEnvironment(resolved resolvedProviderAuthentication, clientApp ClientApp) (map[string]string, error) {
	if !resolved.isGateway() {
		return map[string]string{}, nil
	}

	apiKey, ok := resolved.env[gatewayAPIKeyVariable]
	if !ok {
		return nil, fmt.Errorf("acp resolved environment value %s is unavailable", gatewayAPIKeyVariable)
	}

	baseURL, ok := resolved.env[gatewayBaseURLVariable]
	if !ok {
		return nil, fmt.Errorf("acp resolved environment value %s is unavailable", gatewayBaseURLVariable)
	}

	gateway := gatewayValues{
		apiKey:           apiKey,
		baseURL:          baseURL,
		clientAppName:    clientApp.Name,
		clientAppVersion: clientApp.Version,
	}

	environment := make(map[string]string, len(resolved.provider.Env))

	for name, value := range resolved.provider.Env {
		resolvedValue, err := resolveProfileValue(value, gateway)
		if err != nil {
			return nil, err
		}

		launchValue, err := launchEnvironmentValue(resolvedValue)
		if err != nil {
			return nil, fmt.Errorf("encode acp profile value %s: %w", name, err)
		}

		environment[name] = launchValue
	}

	return environment, nil
}

// launchEnvironmentValue keeps strings and JSON-encodes every other value.
func launchEnvironmentValue(value any) (string, error) {
	if text, ok := value.(string); ok {
		return text, nil
	}

	encoded, err := marshalJSON(value)
	if err != nil {
		return "", err
	}

	return string(encoded), nil
}

// resolveProfileValue replaces the value sources of a profile value.
func resolveProfileValue(value any, gateway gatewayValues) (any, error) {
	switch typed := value.(type) {
	case ValueSource:
		return resolveValueSource(typed, gateway)
	case *ValueSource:
		if typed == nil {
			return nil, nil
		}

		return resolveValueSource(*typed, gateway)
	case []any:
		resolved := make([]any, 0, len(typed))

		for _, item := range typed {
			resolvedItem, err := resolveProfileValue(item, gateway)
			if err != nil {
				return nil, err
			}

			resolved = append(resolved, resolvedItem)
		}

		return resolved, nil
	case map[string]any:
		if source, ok := valueSourceOf(typed); ok {
			return resolveValueSource(source, gateway)
		}

		resolved := make(map[string]any, len(typed))

		for key, item := range typed {
			resolvedItem, err := resolveProfileValue(item, gateway)
			if err != nil {
				return nil, err
			}

			resolved[key] = resolvedItem
		}

		return resolved, nil
	default:
		return value, nil
	}
}

// valueSourceOf decodes a value source written as a map, such as one read
// back from JSON.
func valueSourceOf(value map[string]any) (ValueSource, bool) {
	if _, ok := value["$source"].(string); !ok {
		return ValueSource{}, false
	}

	encoded, err := json.Marshal(value)
	if err != nil {
		return ValueSource{}, false
	}

	var source ValueSource
	if err := json.Unmarshal(encoded, &source); err != nil {
		return ValueSource{}, false
	}

	return source, true
}

// resolveValueSource resolves one value source with its prefix and suffix.
func resolveValueSource(source ValueSource, gateway gatewayValues) (string, error) {
	sourceValue, err := sourceValueOf(source.Source, gateway)
	if err != nil {
		return "", err
	}

	resolved := source.Prefix + sourceValue

	if source.EnsureSuffix == "" {
		return resolved + source.Suffix, nil
	}

	normalized := strings.TrimRight(resolved, "/")
	if strings.HasSuffix(normalized, source.EnsureSuffix) {
		return normalized, nil
	}

	return normalized + source.EnsureSuffix, nil
}

// sourceValueOf returns the value a source names.
func sourceValueOf(name ValueSourceName, gateway gatewayValues) (string, error) {
	switch name {
	case SourceGatewayAPIKey:
		return gateway.apiKey, nil
	case SourceGatewayBaseURL:
		return gateway.baseURL, nil
	case SourceGatewayAuthorization:
		return "Bearer " + gateway.apiKey, nil
	case SourceClientApp:
		return gateway.clientAppName + "/" + gateway.clientAppVersion, nil
	case SourceClientAppName:
		return gateway.clientAppName, nil
	case SourceClientAppVersion:
		return gateway.clientAppVersion, nil
	default:
		return "", fmt.Errorf("acp profile value source %q is unknown", name)
	}
}

// containsProviderCredential reports a profile value that resolves to the
// gateway credential.
func containsProviderCredential(value any) bool {
	switch typed := value.(type) {
	case ValueSource:
		return isCredentialSource(typed.Source)
	case *ValueSource:
		return typed != nil && isCredentialSource(typed.Source)
	case []any:
		for _, item := range typed {
			if containsProviderCredential(item) {
				return true
			}
		}

		return false
	case map[string]any:
		if source, ok := valueSourceOf(typed); ok {
			return isCredentialSource(source.Source)
		}

		for _, item := range typed {
			if containsProviderCredential(item) {
				return true
			}
		}

		return false
	default:
		return false
	}
}

// isCredentialSource reports the sources that carry the gateway credential.
func isCredentialSource(name ValueSourceName) bool {
	return name == SourceGatewayAPIKey || name == SourceGatewayAuthorization
}

// errInvalidProfileValue reports a profile value that is not JSON
// compatible.
var errInvalidProfileValue = errors.New("acp profile values must be JSON-compatible data or value sources")

// validateProfileValue checks that a profile value encodes to JSON.
func validateProfileValue(value any) error {
	if _, err := json.Marshal(value); err != nil {
		return fmt.Errorf("%w: %w", errInvalidProfileValue, err)
	}

	return nil
}
