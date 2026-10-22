package fireworks

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"regexp"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

// forwardedOptionKeys are the ChatOptions members that the openaicompatible
// package forwards in their camelCase spelling. They are replaced by
// requestOptions.
var forwardedOptionKeys = []string{
	"thinking",
	"reasoningHistory",
	"promptCacheKey",
	"serviceTier",
}

// reasoningEfforts maps reasoning efforts onto the low, medium and high
// levels that Fireworks supports.
var reasoningEfforts = map[string]string{
	"minimal": "low",
	"xhigh":   "high",
	"max":     "high",
}

// ChatModel is a Fireworks chat model. It sends OpenAI-compatible chat
// completions with structured outputs, Fireworks' thinking and caching
// options, and reports usage in streams.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "fireworks.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name + ".chat"
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Fireworks downloads no URLs itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return nil
}

// Generate performs a non-streaming chat completion.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	return m.chat(chatOptions).Generate(ctx, options)
}

// Stream performs a streaming chat completion.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	return m.chat(chatOptions).Stream(ctx, options)
}

// chat returns the OpenAI-compatible model of one call. The request body
// transform adds the call's Fireworks options.
func (m *ChatModel) chat(chatOptions ChatOptions) *openaicompatible.ChatModel {
	return openaicompatible.NewChatModel(m.modelID, openaicompatible.ChatConfig{
		Provider:                  Namespace + ".chat",
		BuildURL:                  openaicompatible.InfallibleURL(m.provider.url),
		Headers:                   m.provider.requestHeaders,
		HTTPClient:                m.provider.httpClient,
		ParseError:                parseError,
		IncludeUsage:              true,
		SupportsStructuredOutputs: true,
		GenerateID:                m.provider.generateID,
		TransformRequestBody: func(body map[string]json.RawMessage) (map[string]json.RawMessage, []model.Warning, error) {
			return transformRequestBody(body, chatOptions)
		},
	})
}

// requestOptions are the Fireworks request members of the chat options.
type requestOptions struct {
	PromptCacheKey   *string          `json:"prompt_cache_key,omitempty"`
	ServiceTier      ServiceTier      `json:"service_tier,omitempty"`
	Thinking         *requestThinking `json:"thinking,omitempty"`
	ReasoningHistory ReasoningHistory `json:"reasoning_history,omitempty"`
}

type requestThinking struct {
	Type         ThinkingType `json:"type,omitempty"`
	BudgetTokens *int         `json:"budget_tokens,omitempty"`
}

// transformRequestBody maps the reasoning effort onto the levels Fireworks
// supports and sends the Fireworks options in snake case.
func transformRequestBody(body map[string]json.RawMessage, options ChatOptions) (map[string]json.RawMessage, []model.Warning, error) {
	for _, key := range forwardedOptionKeys {
		delete(body, key)
	}

	warnings, err := mapReasoningEffort(body)
	if err != nil {
		return nil, nil, err
	}

	members, err := encodeRequestOptions(options)
	if err != nil {
		return nil, nil, err
	}

	maps.Copy(body, members)

	return body, warnings, nil
}

// mapReasoningEffort replaces an effort that Fireworks lacks with the
// nearest supported effort and warns about the change.
func mapReasoningEffort(body map[string]json.RawMessage) ([]model.Warning, error) {
	encoded, ok := body["reasoning_effort"]
	if !ok {
		return nil, nil
	}

	var effort string
	if err := json.Unmarshal(encoded, &effort); err != nil {
		return nil, fmt.Errorf("%w: reasoning effort: %w", model.ErrInvalidArgument, err)
	}

	mapped, ok := reasoningEfforts[effort]
	if !ok {
		return nil, nil
	}

	remapped, err := json.Marshal(mapped)
	if err != nil {
		return nil, fmt.Errorf("encode reasoning effort: %w", err)
	}

	body["reasoning_effort"] = remapped

	details := fmt.Sprintf("reasoning %q is not directly supported by this model. mapped to effort %q.", effort, mapped)

	return []model.Warning{
		model.CompatibilityWarning("reasoning", details),
	}, nil
}

func encodeRequestOptions(options ChatOptions) (map[string]json.RawMessage, error) {
	request := requestOptions{
		PromptCacheKey:   options.PromptCacheKey,
		ServiceTier:      options.ServiceTier,
		ReasoningHistory: options.ReasoningHistory,
	}

	if options.Thinking != nil {
		request.Thinking = &requestThinking{
			Type:         options.Thinking.Type,
			BudgetTokens: options.Thinking.BudgetTokens,
		}
	}

	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode fireworks options: %w", err)
	}

	var members map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &members); err != nil {
		return nil, fmt.Errorf("decode fireworks options: %w", err)
	}

	return members, nil
}

var _ model.LanguageModel = (*ChatModel)(nil)
