package cerebras

import (
	"encoding/json"
	"fmt"
	"maps"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// forwardedOptionKeys are the ChatOptions members that the openaicompatible
// package forwards in their camelCase spelling. Cerebras expects them in
// snake case, so they are replaced by requestOptions.
var forwardedOptionKeys = []string{
	"parallelToolCalls",
	"topLogprobs",
	"logitBias",
	"serviceTier",
	"reasoningFormat",
	"promptCacheKey",
}

// requestOptions are the Cerebras request members of the chat options.
type requestOptions struct {
	ParallelToolCalls *bool              `json:"parallel_tool_calls,omitempty"`
	TopLogprobs       *int               `json:"top_logprobs,omitempty"`
	LogitBias         map[string]float64 `json:"logit_bias,omitempty"`
	ServiceTier       ServiceTier        `json:"service_tier,omitempty"`
	ReasoningFormat   ReasoningFormat    `json:"reasoning_format,omitempty"`
	PromptCacheKey    *string            `json:"prompt_cache_key,omitempty"`
}

// transformRequestBody adapts an OpenAI-compatible chat request to Cerebras:
// max_tokens becomes max_completion_tokens, the Cerebras options are sent in
// snake case, and assistant reasoning history moves from reasoning_content to
// reasoning.
func transformRequestBody(body map[string]json.RawMessage, options ChatOptions) (map[string]json.RawMessage, []model.Warning, error) {
	if maxTokens, ok := body["max_tokens"]; ok {
		delete(body, "max_tokens")
		body["max_completion_tokens"] = maxTokens
	}

	for _, key := range forwardedOptionKeys {
		delete(body, key)
	}

	members, err := encodeRequestOptions(options)
	if err != nil {
		return nil, nil, err
	}

	maps.Copy(body, members)

	messages, ok := body["messages"]
	if !ok {
		return body, nil, nil
	}

	renamed, err := renameReasoningHistory(messages)
	if err != nil {
		return nil, nil, err
	}

	body["messages"] = renamed

	return body, nil, nil
}

func encodeRequestOptions(options ChatOptions) (map[string]json.RawMessage, error) {
	encoded, err := json.Marshal(requestOptions{
		ParallelToolCalls: options.ParallelToolCalls,
		TopLogprobs:       options.TopLogprobs,
		LogitBias:         options.LogitBias,
		ServiceTier:       options.ServiceTier,
		ReasoningFormat:   options.ReasoningFormat,
		PromptCacheKey:    options.PromptCacheKey,
	})
	if err != nil {
		return nil, fmt.Errorf("encode cerebras options: %w", err)
	}

	var members map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &members); err != nil {
		return nil, fmt.Errorf("decode cerebras options: %w", err)
	}

	return members, nil
}

// renameReasoningHistory moves the reasoning_content of assistant messages to
// reasoning, unless a message already has reasoning. The messages are
// produced by the openaicompatible package, which owns their shape, so they
// are edited as ordered JSON objects.
func renameReasoningHistory(encoded json.RawMessage) (json.RawMessage, error) {
	// Messages that are not an array are passed on unchanged.
	var messages []json.RawMessage
	if err := json.Unmarshal(encoded, &messages); err != nil {
		return encoded, nil
	}

	for index, message := range messages {
		object, err := jsonobject.Parse(message)
		if err != nil {
			continue
		}

		role, _ := object.GetString("role")
		reasoning, hasReasoningContent := object.Get("reasoning_content")

		if role != "assistant" || !hasReasoningContent {
			continue
		}

		object = object.Without("reasoning_content")
		if !object.Has("reasoning") {
			object = object.With("reasoning", reasoning)
		}

		renamedMessage, err := object.Encode()
		if err != nil {
			return nil, fmt.Errorf("encode message: %w", err)
		}

		messages[index] = renamedMessage
	}

	renamed, err := json.Marshal(messages)
	if err != nil {
		return nil, fmt.Errorf("encode messages: %w", err)
	}

	return renamed, nil
}
