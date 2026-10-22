package openaicompatible

import (
	"encoding/json"
	"fmt"
	"maps"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// Provider options namespaces shared by every OpenAI-compatible provider.
const (
	// Namespace is the provider options key read by every model in this
	// package, in addition to the provider's own name.
	Namespace = "openaiCompatible"

	// deprecatedNamespace is the former spelling of Namespace.
	deprecatedNamespace = "openai-compatible"
)

// ChatOptions are the provider options of chat models, read from the
// "openaiCompatible" namespace and the provider's own namespace. Keys of the
// provider's namespace that are not listed here are sent in the request body
// unchanged.
type ChatOptions struct {
	// User identifies the end user, to help the provider detect abuse.
	User *string `json:"user,omitempty"`

	// ReasoningEffort selects the reasoning effort of reasoning models.
	ReasoningEffort *string `json:"reasoningEffort,omitempty"`

	// TextVerbosity controls the verbosity of the generated text.
	TextVerbosity *string `json:"textVerbosity,omitempty"`

	// StrictJSONSchema requests constrained decoding for JSON schema response
	// formats. It defaults to true.
	StrictJSONSchema *bool `json:"strictJsonSchema,omitempty"`
}

// chatOptionKeys are the ChatOptions keys, which are never passed through to
// the request body.
var chatOptionKeys = []string{
	"user",
	"reasoningEffort",
	"textVerbosity",
	"strictJsonSchema",
}

// CompletionOptions are the provider options of completion models, read from
// the provider's own namespace. Every key of that namespace is also sent in
// the request body unchanged.
type CompletionOptions struct {
	// Echo returns the prompt in addition to the completion.
	Echo *bool `json:"echo,omitempty"`

	// LogitBias maps token IDs to a bias from -100 to 100.
	LogitBias map[string]float64 `json:"logitBias,omitempty"`

	// Suffix is the text that follows the inserted completion.
	Suffix *string `json:"suffix,omitempty"`

	// User identifies the end user, to help the provider detect abuse.
	User *string `json:"user,omitempty"`
}

// EmbeddingOptions are the provider options of embedding models, read from
// the "openaiCompatible" namespace and the provider's own namespace.
type EmbeddingOptions struct {
	// Dimensions requests a vector size. It takes precedence over
	// model.EmbedOptions.Dimensions.
	Dimensions *int `json:"dimensions,omitempty"`

	// User identifies the end user, to help the provider detect abuse.
	User *string `json:"user,omitempty"`
}

// ImageOptions are common provider options of image models, read from the
// provider's own namespace. Every key of that namespace, including keys not
// listed here, is sent in the request unchanged, so the JSON names follow the
// wire format.
type ImageOptions struct {
	// Size is the dimensions of the generated image.
	Size *string `json:"size,omitempty"`

	// Quality is the quality of the generated image.
	Quality *string `json:"quality,omitempty"`

	// OutputFormat is the file format of the generated image.
	OutputFormat *string `json:"output_format,omitempty"`

	// OutputCompression is the compression level, from 0 to 100, of JPEG and
	// WebP images.
	OutputCompression *int `json:"output_compression,omitempty"`

	// Background selects the background behavior of the generated image.
	Background *string `json:"background,omitempty"`
}

// camelCaseSeparator matches a separator followed by a lowercase letter.
var camelCaseSeparator = regexp.MustCompile(`[_-][a-z]`)

// toCamelCase converts "my-provider" and "my_provider" to "myProvider".
func toCamelCase(name string) string {
	return camelCaseSeparator.ReplaceAllStringFunc(name, func(match string) string {
		return strings.ToUpper(match[1:])
	})
}

// optionsName returns the provider options namespace of a provider
// identifier such as "name.chat".
func optionsName(provider string) string {
	name, _, _ := strings.Cut(provider, ".")

	return strings.TrimSpace(name)
}

// metadataKey returns the namespace for provider metadata: the camelCase
// spelling of rawName when the caller used it in the provider options,
// otherwise rawName.
func metadataKey(rawName string, options model.ProviderOptions) string {
	camelName := toCamelCase(rawName)
	if camelName != rawName && options[camelName] != nil {
		return camelName
	}

	return rawName
}

// deprecatedNameWarnings warns when the caller used the raw spelling of a
// provider name that has a camelCase spelling.
func deprecatedNameWarnings(rawName string, options model.ProviderOptions) []model.Warning {
	camelName := toCamelCase(rawName)
	if camelName == rawName || options[rawName] == nil {
		return nil
	}

	return []model.Warning{
		{
			Type:    model.WarningDeprecated,
			Setting: fmt.Sprintf("providerOptions key '%s'", rawName),
			Message: fmt.Sprintf("Use '%s' instead.", camelName),
		},
	}
}

// deprecatedNamespaceWarnings warns when the caller used the former
// "openai-compatible" namespace.
func deprecatedNamespaceWarnings(options model.ProviderOptions) []model.Warning {
	if options[deprecatedNamespace] == nil {
		return nil
	}

	return []model.Warning{
		{
			Type:    model.WarningDeprecated,
			Setting: fmt.Sprintf("providerOptions key '%s'", deprecatedNamespace),
			Message: fmt.Sprintf("Use '%s' instead.", Namespace),
		},
	}
}

// mergedOptions decodes the options of each namespace into T and combines
// them, later namespaces replacing the fields that they set.
func mergedOptions[T any](options model.ProviderOptions, namespaces ...string) (T, error) {
	var merged T

	members, err := mergedMembers(options, namespaces...)
	if err != nil {
		return merged, err
	}

	for _, namespace := range namespaces {
		if _, _, err := model.ProviderValue[T](options, namespace); err != nil {
			return merged, err
		}
	}

	encoded, err := json.Marshal(members)
	if err != nil {
		return merged, fmt.Errorf("%w: encode provider options: %w", model.ErrInvalidArgument, err)
	}

	if err := json.Unmarshal(encoded, &merged); err != nil {
		return merged, fmt.Errorf("%w: decode provider options: %w", model.ErrInvalidArgument, err)
	}

	return merged, nil
}

// mergedMembers returns the JSON members of each namespace's options, later
// namespaces replacing members of earlier ones. The values are arbitrary JSON
// that is forwarded to the provider.
func mergedMembers(options model.ProviderOptions, namespaces ...string) (map[string]json.RawMessage, error) {
	merged := make(map[string]json.RawMessage)

	for _, namespace := range namespaces {
		members, _, err := model.ProviderValue[map[string]json.RawMessage](options, namespace)
		if err != nil {
			return nil, err
		}

		maps.Copy(merged, members)
	}

	return merged, nil
}
