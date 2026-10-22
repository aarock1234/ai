package middleware

import (
	"cmp"
	"context"
	"net/http"
	"slices"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/merge"
	"github.com/aarock1234/ai/pkg/model"
)

// DefaultSettings returns middleware that fills unset call options from
// settings. Options set by the call take precedence; headers merge per key
// and provider options merge deeply. The settings' Prompt and
// IncludeRawChunks are ignored.
func DefaultSettings(settings model.CallOptions) Language {
	return Language{
		TransformParams: func(_ context.Context, transform LanguageTransform) (model.CallOptions, error) {
			return applyDefaultSettings(settings, transform.Params), nil
		},
	}
}

func applyDefaultSettings(defaults, params model.CallOptions) model.CallOptions {
	merged := params

	merged.MaxOutputTokens = cmp.Or(params.MaxOutputTokens, defaults.MaxOutputTokens)
	merged.Temperature = cmp.Or(params.Temperature, defaults.Temperature)
	merged.TopP = cmp.Or(params.TopP, defaults.TopP)
	merged.TopK = cmp.Or(params.TopK, defaults.TopK)
	merged.PresencePenalty = cmp.Or(params.PresencePenalty, defaults.PresencePenalty)
	merged.FrequencyPenalty = cmp.Or(params.FrequencyPenalty, defaults.FrequencyPenalty)
	merged.Seed = cmp.Or(params.Seed, defaults.Seed)
	merged.ResponseFormat = cmp.Or(params.ResponseFormat, defaults.ResponseFormat)
	merged.ToolChoice = cmp.Or(params.ToolChoice, defaults.ToolChoice)
	merged.Reasoning = cmp.Or(params.Reasoning, defaults.Reasoning)

	if params.StopSequences == nil {
		merged.StopSequences = defaults.StopSequences
	}

	if params.Tools == nil {
		merged.Tools = defaults.Tools
	}

	merged.Headers = mergeHeaders(defaults.Headers, params.Headers)
	merged.ProviderOptions = merge.Deep(defaults.ProviderOptions, params.ProviderOptions)

	return merged
}

// DefaultEmbeddingSettings returns middleware that fills unset embedding
// options from settings. Options set by the call take precedence; headers
// merge per key and provider options merge deeply. The settings' Values are
// ignored.
func DefaultEmbeddingSettings(settings model.EmbedOptions) Embedding {
	return Embedding{
		TransformParams: func(_ context.Context, transform EmbeddingTransform) (model.EmbedOptions, error) {
			merged := transform.Params

			merged.Dimensions = cmp.Or(merged.Dimensions, settings.Dimensions)
			merged.Headers = mergeHeaders(settings.Headers, merged.Headers)
			merged.ProviderOptions = merge.Deep(settings.ProviderOptions, merged.ProviderOptions)

			return merged, nil
		},
	}
}

// mergeHeaders lets the call's headers replace default headers per key. It
// keeps nil when neither set has headers.
func mergeHeaders(defaults, overrides http.Header) http.Header {
	if defaults == nil && overrides == nil {
		return nil
	}

	return apicall.MergeHeaders(defaults, overrides)
}

// DefaultInstructions returns middleware that prepends instructions to
// calls whose prompt has no system message.
func DefaultInstructions(instructions []model.SystemMessage) Language {
	instructions = slices.Clone(instructions)

	return Language{
		TransformParams: func(_ context.Context, transform LanguageTransform) (model.CallOptions, error) {
			params := transform.Params
			if len(instructions) == 0 || hasSystemMessage(params.Prompt) {
				return params, nil
			}

			prompt := make([]model.Message, 0, len(instructions)+len(params.Prompt))
			for _, instruction := range instructions {
				prompt = append(prompt, instruction)
			}

			params.Prompt = append(prompt, params.Prompt...)

			return params, nil
		},
	}
}

func hasSystemMessage(prompt []model.Message) bool {
	return slices.ContainsFunc(prompt, func(message model.Message) bool {
		return message.Role() == model.RoleSystem
	})
}
