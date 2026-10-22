package openrouter

import (
	"encoding/json"
	"fmt"
	"maps"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

// transformRequestBody moves the provider-independent reasoning_effort into
// OpenRouter's reasoning object. A reasoning object from the provider options
// takes precedence.
func transformRequestBody(body map[string]json.RawMessage) (map[string]json.RawMessage, []model.Warning, error) {
	encodedEffort, ok := body["reasoning_effort"]
	if !ok {
		return body, nil, nil
	}

	delete(body, "reasoning_effort")

	if _, ok := body["reasoning"]; ok {
		return body, nil, nil
	}

	var effort string
	if err := json.Unmarshal(encodedEffort, &effort); err != nil {
		return nil, nil, fmt.Errorf("%w: reasoning effort: %w", model.ErrInvalidArgument, err)
	}

	reasoning, warnings := effortReasoning(effort)

	encodedReasoning, err := json.Marshal(reasoning)
	if err != nil {
		return nil, nil, fmt.Errorf("encode reasoning: %w", err)
	}

	body["reasoning"] = encodedReasoning

	return body, warnings, nil
}

// effortReasoning maps a reasoning level onto OpenRouter's efforts. The
// maximum level has no OpenRouter equivalent and uses the highest effort.
func effortReasoning(effort string) (ReasoningOptions, []model.Warning) {
	if model.ReasoningEffort(effort) != model.ReasoningMax {
		return ReasoningOptions{
			Effort: ReasoningEffort(effort),
		}, nil
	}

	reasoning := ReasoningOptions{
		Effort: ReasoningEffortXHigh,
	}

	warnings := []model.Warning{
		model.CompatibilityWarning(
			"reasoning",
			fmt.Sprintf("reasoning %q is not directly supported by this model. mapped to effort %q.", effort, ReasoningEffortXHigh),
		),
	}

	return reasoning, warnings
}

// replayReasoningDetails returns the prompt with the reasoning details of
// each assistant message moved onto the message, where OpenRouter expects
// them. The caller's messages are not modified.
func replayReasoningDetails(prompt []model.Message) ([]model.Message, error) {
	replayed := make([]model.Message, len(prompt))

	for index, message := range prompt {
		assistant, ok := message.(model.AssistantMessage)
		if !ok {
			replayed[index] = message

			continue
		}

		converted, err := replayMessageDetails(assistant)
		if err != nil {
			return nil, err
		}

		replayed[index] = converted
	}

	return replayed, nil
}

func replayMessageDetails(message model.AssistantMessage) (model.AssistantMessage, error) {
	var details []ReasoningDetail

	for _, part := range message.Content {
		reasoning, ok := part.(model.ReasoningPart)
		if !ok {
			continue
		}

		metadata, _, err := model.ProviderValue[ReasoningMetadata](reasoning.ProviderOptions, Namespace)
		if err != nil {
			return model.AssistantMessage{}, err
		}

		details = append(details, metadata.ReasoningDetails...)
	}

	if len(details) == 0 {
		return message, nil
	}

	extra, _, err := model.ProviderValue[map[string]json.RawMessage](message.ProviderOptions, openaicompatible.Namespace)
	if err != nil {
		return model.AssistantMessage{}, err
	}

	encodedDetails, err := json.Marshal(details)
	if err != nil {
		return model.AssistantMessage{}, fmt.Errorf("%w: encode reasoning details: %w", model.ErrInvalidPrompt, err)
	}

	extra = maps.Clone(extra)
	if extra == nil {
		extra = make(map[string]json.RawMessage)
	}

	extra["reasoning_details"] = encodedDetails

	options := maps.Clone(message.ProviderOptions)
	if options == nil {
		options = make(model.ProviderOptions)
	}

	options[openaicompatible.Namespace] = extra
	message.ProviderOptions = options

	return message, nil
}
