package openai

import (
	"maps"

	"github.com/aarock1234/ai/pkg/model"
)

// CompletionOptions are the legacy Completions settings read from the
// "openai" provider options namespace and, when the provider has a custom
// name, from that namespace too.
type CompletionOptions struct {
	// Echo returns the prompt in addition to the completion.
	Echo *bool `json:"echo,omitempty"`

	// LogitBias maps token IDs to a bias from -100 to 100 that is added to
	// the token's logit before sampling. For example {"50256": -100}
	// prevents the <|endoftext|> token.
	LogitBias map[string]float64 `json:"logitBias,omitempty"`

	// Suffix is the text that follows the inserted completion.
	Suffix string `json:"suffix,omitempty"`

	// User identifies the end user for abuse monitoring.
	User string `json:"user,omitempty"`

	// Logprobs requests the log probabilities of the generated tokens.
	Logprobs *LogprobsSetting `json:"logprobs,omitempty"`
}

// CompletionMetadata is the "openai" provider metadata of a Completions
// result.
type CompletionMetadata struct {
	// Logprobs holds the token log probabilities, when requested.
	Logprobs *CompletionLogprobs `json:"logprobs,omitempty"`
}

// CompletionLogprobs are the log probabilities of the generated tokens.
type CompletionLogprobs struct {
	Tokens        []string             `json:"tokens"`
	TokenLogprobs []float64            `json:"token_logprobs"`
	TopLogprobs   []map[string]float64 `json:"top_logprobs,omitempty"`
}

// readCompletionOptions reads the "openai" options, then overrides them with
// the options of the provider's own namespace.
func readCompletionOptions(options model.ProviderOptions, namespace string) (CompletionOptions, error) {
	values, _, err := model.ProviderValue[CompletionOptions](options, Namespace)
	if err != nil {
		return CompletionOptions{}, err
	}

	if namespace == Namespace {
		return values, nil
	}

	overrides, ok, err := model.ProviderValue[CompletionOptions](options, namespace)
	if err != nil || !ok {
		return values, err
	}

	return values.merge(overrides), nil
}

// merge returns o with every setting of overrides applied on top.
func (o CompletionOptions) merge(overrides CompletionOptions) CompletionOptions {
	merged := o

	if overrides.Echo != nil {
		merged.Echo = overrides.Echo
	}

	if overrides.LogitBias != nil {
		merged.LogitBias = maps.Clone(overrides.LogitBias)
	}

	if overrides.Suffix != "" {
		merged.Suffix = overrides.Suffix
	}

	if overrides.User != "" {
		merged.User = overrides.User
	}

	if overrides.Logprobs != nil {
		merged.Logprobs = overrides.Logprobs
	}

	return merged
}

// wireLogprobs returns the logprobs request value: zero for the generated
// tokens only, the number of alternatives, or nil when not requested.
func (o CompletionOptions) wireLogprobs() *int {
	if !o.Logprobs.requested() {
		return nil
	}

	return new(o.Logprobs.Top)
}
