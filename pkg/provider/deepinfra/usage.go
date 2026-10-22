package deepinfra

import (
	"encoding/json"
	"strconv"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

// convertUsage corrects the usage of Gemini and Gemma models before the
// default conversion. For those models DeepInfra reports completion_tokens
// without the reasoning tokens, which contradicts the OpenAI-compatible
// format and would produce negative text token counts. When the reasoning
// tokens exceed the completion tokens, they are added to the completion and
// total tokens, and the raw usage reports the corrected counts.
func convertUsage(usage *openaicompatible.ChatUsage) model.Usage {
	reasoningTokens := reasoningTokensOf(usage)
	if reasoningTokens == 0 {
		return openaicompatible.ConvertChatUsage(usage)
	}

	completionTokens := valueOrZero(usage.CompletionTokens)
	if reasoningTokens <= completionTokens {
		return openaicompatible.ConvertChatUsage(usage)
	}

	correctedCompletionTokens := completionTokens + reasoningTokens
	promptTokens := valueOrZero(usage.PromptTokens)

	cacheReadTokens := 0
	if usage.PromptTokensDetails != nil {
		cacheReadTokens = valueOrZero(usage.PromptTokensDetails.CachedTokens)
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:     new(promptTokens),
			NoCache:   new(promptTokens - cacheReadTokens),
			CacheRead: new(cacheReadTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(correctedCompletionTokens),
			Text:      new(correctedCompletionTokens - reasoningTokens),
			Reasoning: new(reasoningTokens),
		},
		Raw: correctedRawUsage(usage, correctedCompletionTokens, reasoningTokens),
	}
}

func reasoningTokensOf(usage *openaicompatible.ChatUsage) int {
	if usage == nil || usage.CompletionTokensDetails == nil {
		return 0
	}

	return valueOrZero(usage.CompletionTokensDetails.ReasoningTokens)
}

// correctedRawUsage returns the raw usage with the corrected completion and
// total token counts. A missing total stays missing. Raw usage that is not an
// object is returned unchanged.
func correctedRawUsage(usage *openaicompatible.ChatUsage, completionTokens int, reasoningTokens int) json.RawMessage {
	object, err := jsonobject.Parse(usage.Raw)
	if err != nil {
		return usage.Raw
	}

	object = object.With("completion_tokens", encodeInt(completionTokens))

	if usage.TotalTokens == nil {
		object = object.Without("total_tokens")
	} else {
		object = object.With("total_tokens", encodeInt(*usage.TotalTokens+reasoningTokens))
	}

	corrected, err := object.Encode()
	if err != nil {
		return usage.Raw
	}

	return corrected
}

func encodeInt(value int) json.RawMessage {
	return json.RawMessage(strconv.Itoa(value))
}

func valueOrZero(value *int) int {
	if value == nil {
		return 0
	}

	return *value
}
