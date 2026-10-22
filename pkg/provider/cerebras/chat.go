package cerebras

import (
	"context"
	"encoding/json"
	"iter"
	"regexp"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

// rawToolCallsFinishReason is the finish reason that Cerebras reports with
// tool calls.
const rawToolCallsFinishReason = "tool_calls"

// ChatModel is a Cerebras chat model. It sends OpenAI-compatible chat
// completions with Cerebras' request options. Some Cerebras models repeat a
// tool call after valid structured output; such responses are reported as
// finished text.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "cerebras.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name + ".chat"
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Cerebras downloads no URLs itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return nil
}

// Generate performs a non-streaming chat completion.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	result, err := m.chat(chatOptions).Generate(ctx, options)
	if err != nil {
		return nil, err
	}

	if !isStructuredOutputWithToolCalls(result.FinishReason.Raw, hasText(result.Content), options.ResponseFormat) {
		return result, nil
	}

	result.Content = withoutToolCalls(result.Content)
	result.FinishReason.Unified = model.FinishStop

	return result, nil
}

// Stream performs a streaming chat completion.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	inner, err := m.chat(chatOptions).Stream(ctx, options)
	if err != nil {
		return nil, err
	}

	repair := &streamRepair{
		responseFormat: options.ResponseFormat,
	}

	result := model.NewStreamResult(repair.parts(inner.Parts()), inner.Close)
	result.Request = inner.Request
	result.Response = inner.Response

	return result, nil
}

// chat returns the OpenAI-compatible model of one call. The request body
// transform adds the call's Cerebras options.
func (m *ChatModel) chat(chatOptions ChatOptions) *openaicompatible.ChatModel {
	return openaicompatible.NewChatModel(m.modelID, openaicompatible.ChatConfig{
		Provider:                  Namespace + ".chat",
		BuildURL:                  openaicompatible.InfallibleURL(m.provider.url),
		Headers:                   m.provider.requestHeaders,
		HTTPClient:                m.provider.httpClient,
		ParseError:                parseError,
		SupportsStructuredOutputs: true,
		GenerateID:                m.provider.generateID,
		TransformRequestBody: func(body map[string]json.RawMessage) (map[string]json.RawMessage, []model.Warning, error) {
			return transformRequestBody(body, chatOptions)
		},
	})
}

// isStructuredOutputWithToolCalls reports a JSON response that has text and
// also finished with tool calls.
func isStructuredOutputWithToolCalls(rawFinishReason string, hasText bool, format *model.ResponseFormat) bool {
	return isJSONFormat(format) && rawFinishReason == rawToolCallsFinishReason && hasText
}

func isJSONFormat(format *model.ResponseFormat) bool {
	return format != nil && format.Type == model.ResponseFormatJSON
}

func hasText(content []model.Content) bool {
	for _, item := range content {
		if text, ok := item.(model.Text); ok && text.Text != "" {
			return true
		}
	}

	return false
}

func withoutToolCalls(content []model.Content) []model.Content {
	kept := make([]model.Content, 0, len(content))
	for _, item := range content {
		if _, ok := item.(model.ToolCall); !ok {
			kept = append(kept, item)
		}
	}

	return kept
}

// streamRepair drops tool calls that follow structured output text and
// reports such streams as finished text.
type streamRepair struct {
	responseFormat *model.ResponseFormat
	hasText        bool
}

func (r *streamRepair) parts(parts iter.Seq[model.StreamPart]) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		for part := range parts {
			repaired, keep := r.repair(part)
			if !keep {
				continue
			}

			if !yield(repaired) {
				return
			}
		}
	}
}

// repair returns the part to emit, or false when the part is dropped.
func (r *streamRepair) repair(part model.StreamPart) (model.StreamPart, bool) {
	switch typed := part.(type) {
	case model.TextDelta:
		if typed.Delta != "" {
			r.hasText = true
		}

	case model.Finish:
		if isStructuredOutputWithToolCalls(typed.FinishReason.Raw, r.hasText, r.responseFormat) {
			typed.FinishReason.Unified = model.FinishStop

			return typed, true
		}

	case model.ToolInputStart, model.ToolInputDelta, model.ToolInputEnd, model.ToolCall:
		if isJSONFormat(r.responseFormat) && r.hasText {
			return nil, false
		}
	}

	return part, true
}

var _ model.LanguageModel = (*ChatModel)(nil)
