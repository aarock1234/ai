package mistral

import (
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// preparedCompletion holds the settings and prompt shared by chat
// completions and conversations.
type preparedCompletion struct {
	Settings completionSettings
	Prompt   []model.Message
	Warnings []model.Warning
}

// prepareCompletion converts the standard call settings. Without structured
// outputs or a schema, JSON output is requested through the system prompt,
// as Mistral's JSON mode requires.
func prepareCompletion(
	options model.CallOptions,
	chatOptions ChatOptions,
	supportsReasoningEffort bool,
) (preparedCompletion, error) {
	var warnings []model.Warning
	if options.TopK != nil {
		warnings = append(warnings, model.UnsupportedWarning("topK", ""))
	}

	effort, reasoningWarnings := reasoningEffort(supportsReasoningEffort, chatOptions.ReasoningEffort, options.Reasoning)
	warnings = append(warnings, reasoningWarnings...)

	structuredOutputs := optional.Or(chatOptions.StructuredOutputs, true)
	strictJSONSchema := optional.Or(chatOptions.StrictJSONSchema, false)

	prompt := options.Prompt
	format := options.ResponseFormat
	wantsJSON := format != nil && format.Type == model.ResponseFormatJSON
	usesSchema := wantsJSON && structuredOutputs && format.Schema != nil

	if wantsJSON && !usesSchema {
		injected, err := injectJSONInstruction(prompt, format.Schema)
		if err != nil {
			return preparedCompletion{}, err
		}

		prompt = injected
	}

	settings := completionSettings{
		MaxTokens:        options.MaxOutputTokens,
		Temperature:      options.Temperature,
		TopP:             options.TopP,
		FrequencyPenalty: options.FrequencyPenalty,
		PresencePenalty:  options.PresencePenalty,
		Stop:             options.StopSequences,
		RandomSeed:       options.Seed,
		ReasoningEffort:  effort,
	}

	switch {
	case usesSchema:
		settings.ResponseFormat = schemaResponseFormat(*format, strictJSONSchema)

	case wantsJSON:
		settings.ResponseFormat = &responseFormat{
			Type: "json_object",
		}
	}

	return preparedCompletion{
		Settings: settings,
		Prompt:   prompt,
		Warnings: warnings,
	}, nil
}

func schemaResponseFormat(format model.ResponseFormat, strict bool) *responseFormat {
	name := format.Name
	if name == "" {
		name = "response"
	}

	return &responseFormat{
		Type: "json_schema",
		JSONSchema: &jsonSchema{
			Schema:      format.Schema,
			Strict:      strict,
			Name:        name,
			Description: format.Description,
		},
	}
}
