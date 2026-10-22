package alibaba

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

// chatPath is the chat completions endpoint.
const chatPath = "/chat/completions"

// maxThinkingBudget is the output limit that reasoning levels take their
// thinking budget share of, and the largest budget.
const maxThinkingBudget = 16384

// imageURLs are the image URLs that Alibaba downloads itself.
var imageURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
}

// jsonSchemaModelPrefixes name the models with JSON schema output, alone or
// followed by a dated or variant suffix.
var jsonSchemaModelPrefixes = []string{
	"qwen3.7-plus",
	"qwen3.7-flash",
	"qwen3.7-max",
	"qwen3.8-max",
	"qwen3.8-flash",
}

// preservedThinkingModels support replaying reasoning from earlier turns.
var preservedThinkingModels = map[string]bool{
	"kimi-k2.7-code":           true,
	"qwen3.6-max-preview":      true,
	"qwen3.6-plus":             true,
	"qwen3.6-plus-2026-04-02":  true,
	"qwen3.7-flash":            true,
	"qwen3.7-flash-2026-07-15": true,
	"qwen3.7-max":              true,
	"qwen3.7-max-2026-05-17":   true,
	"qwen3.7-max-2026-05-20":   true,
	"qwen3.7-max-2026-06-08":   true,
	"qwen3.7-max-preview":      true,
	"qwen3.7-plus":             true,
	"qwen3.7-plus-2026-05-26":  true,
	"qwen3.8-flash":            true,
	"qwen3.8-max":              true,
	"qwen3.8-max-0902":         true,
}

// ChatModel is an Alibaba Qwen chat completions language model with
// thinking, preserved thinking and explicit context caching.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "alibaba.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name + ".chat"
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Alibaba downloads image URLs itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return imageURLs
}

// chatArgs is a prepared request.
type chatArgs struct {
	body     chatRequest
	warnings []model.Warning
}

// Generate performs a non-streaming chat completion.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	args, err := m.prepareArgs(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(m.provider.baseURL+chatPath, parseError, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, args.body)
	if err != nil {
		return nil, err
	}

	var decoded chatResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if err := decoded.validate(); err != nil {
		return nil, err
	}

	if len(decoded.Choices) == 0 {
		return nil, fmt.Errorf("%w: response did not contain any choices", model.ErrInvalidResponse)
	}

	choice := decoded.Choices[0]

	return &model.GenerateResult{
		Content:      m.generatedContent(choice.Message),
		FinishReason: mapFinishReason(choice.FinishReason),
		Usage:        convertUsage(decoded.Usage),
		Warnings:     args.warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: responseMetadata(decoded.ID, decoded.Model, decoded.Created),
			Headers:          response.Headers,
			Body:             response.Body,
		},
	}, nil
}

// Stream performs a streaming chat completion. Usage is requested unless the
// provider disables it.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	args, err := m.prepareArgs(options)
	if err != nil {
		return nil, err
	}

	args.body.Stream = true
	if m.provider.includeUsage {
		args.body.StreamOptions = &streamOptions{
			IncludeUsage: true,
		}
	}

	request, err := m.provider.apiRequest(m.provider.baseURL+chatPath, parseError, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, args.body)
	if err != nil {
		return nil, err
	}

	processor := newChatStream(options.IncludeRawChunks, m.provider.generateID)

	start := model.StreamStart{
		Warnings: args.warnings,
	}

	result := model.NewStreamResult(chatsse.Parts(ctx, response.Body, start, processor), response.Body.Close)
	result.Request = model.RequestInfo{
		Body: response.RequestBody,
	}
	result.Response = model.StreamResponseInfo{
		Headers: response.Headers,
	}

	return result, nil
}

// prepareArgs builds the request body and warnings of a call.
func (m *ChatModel) prepareArgs(options model.CallOptions) (chatArgs, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return chatArgs{}, err
	}

	var warnings []model.Warning

	if options.FrequencyPenalty != nil {
		warnings = append(warnings, model.UnsupportedWarning("frequencyPenalty", ""))
	}

	preserveThinking := chatOptions.PreserveThinking
	if preserveThinking == nil && preservedThinkingModels[m.modelID] {
		preserveThinking = new(true)
	}

	format := options.ResponseFormat
	useJSONSchema := format != nil && format.Type == model.ResponseFormatJSON && format.Schema != nil && supportsJSONSchema(m.modelID)
	useJSONObject := format != nil && format.Type == model.ResponseFormatJSON && !useJSONSchema

	prompt := options.Prompt
	if useJSONObject {
		if format.Schema != nil {
			warnings = append(warnings, model.CompatibilityWarning(
				"responseFormat JSON schema",
				fmt.Sprintf(
					"Alibaba does not support JSON Schema output for model %s. JSON Object mode is used instead. "+
						"The schema was injected into the system message and will only be validated locally.",
					m.modelID,
				),
			))
		}

		prompt, err = injectJSONInstruction(prompt, format.Schema)
		if err != nil {
			return chatArgs{}, err
		}
	}

	thinkingSettings, thinkingWarnings := resolveThinking(chatOptions, options.Reasoning)
	warnings = append(warnings, thinkingWarnings...)

	cache := &cacheControls{}

	messages, err := convertChatPrompt(prompt, cache, preserveThinking != nil && *preserveThinking)
	if err != nil {
		return chatArgs{}, err
	}

	tools, err := prepareTools(options.Tools, options.ToolChoice)
	if err != nil {
		return chatArgs{}, err
	}

	body := chatRequest{
		Model:            m.modelID,
		MaxTokens:        options.MaxOutputTokens,
		Temperature:      options.Temperature,
		TopP:             options.TopP,
		TopK:             options.TopK,
		PresencePenalty:  options.PresencePenalty,
		Stop:             options.StopSequences,
		Seed:             options.Seed,
		ResponseFormat:   responseFormatOf(format, useJSONSchema),
		EnableThinking:   thinkingSettings.enableThinking,
		ThinkingBudget:   thinkingSettings.thinkingBudget,
		PreserveThinking: preserveThinking,
		Messages:         messages,
		Tools:            tools.Tools,
		ToolChoice:       tools.ToolChoice,
	}

	if tools.Tools != nil {
		body.ParallelToolCalls = chatOptions.ParallelToolCalls
	}

	warnings = append(warnings, cache.warnings...)

	return chatArgs{
		body:     body,
		warnings: append(warnings, tools.Warnings...),
	}, nil
}

// supportsJSONSchema reports whether the model has JSON schema output.
func supportsJSONSchema(modelID string) bool {
	for _, prefix := range jsonSchemaModelPrefixes {
		if modelID == prefix || strings.HasPrefix(modelID, prefix+"-") {
			return true
		}
	}

	return false
}

// responseFormatOf requests a JSON schema when the model supports it and a
// JSON object otherwise.
func responseFormatOf(format *model.ResponseFormat, useJSONSchema bool) *responseFormat {
	if format == nil || format.Type != model.ResponseFormatJSON {
		return nil
	}

	if !useJSONSchema {
		return &responseFormat{
			Type: responseFormatJSONObject,
		}
	}

	name := format.Name
	if name == "" {
		name = "response"
	}

	return &responseFormat{
		Type: responseFormatJSONSchema,
		JSONSchema: &jsonSchemaFormat{
			Schema:      format.Schema,
			Name:        name,
			Description: format.Description,
		},
	}
}

// thinkingSettings are the thinking members of a request.
type thinkingSettings struct {
	enableThinking *bool
	thinkingBudget *int
}

// resolveThinking prefers the provider options and maps the
// provider-independent reasoning level to a thinking budget otherwise.
func resolveThinking(options ChatOptions, level model.ReasoningEffort) (thinkingSettings, []model.Warning) {
	if options.EnableThinking != nil || options.ThinkingBudget != nil {
		return thinkingSettings{
			enableThinking: options.EnableThinking,
			thinkingBudget: options.ThinkingBudget,
		}, nil
	}

	if !reasoning.IsCustom(level) {
		return thinkingSettings{}, nil
	}

	if level == model.ReasoningNone {
		return thinkingSettings{
			enableThinking: new(false),
		}, nil
	}

	settings := thinkingSettings{
		enableThinking: new(true),
	}

	budget, warnings, ok := reasoning.Budget(level, reasoning.BudgetLimits{
		MaxOutputTokens: maxThinkingBudget,
		MinBudget:       reasoning.DefaultMinBudget,
		MaxBudget:       maxThinkingBudget,
	})
	if ok {
		settings.thinkingBudget = new(budget)
	}

	return settings, warnings
}

// generatedContent converts a response message: reasoning, then text, then
// tool calls.
func (m *ChatModel) generatedContent(message responseMessage) []model.Content {
	var content []model.Content

	if text := valueOrEmpty(message.ReasoningContent); text != "" {
		content = append(content, model.Reasoning{
			Text: text,
		})
	}

	if text := valueOrEmpty(message.Content); text != "" {
		content = append(content, model.Text{
			Text: text,
		})
	}

	for _, toolCall := range message.ToolCalls {
		toolCallID := toolCall.ID
		if toolCallID == "" {
			toolCallID = m.provider.generateID()
		}

		content = append(content, model.ToolCall{
			ToolCallID: toolCallID,
			ToolName:   toolCall.Function.Name,
			Input:      toolCall.Function.Arguments,
		})
	}

	return content
}

var _ model.LanguageModel = (*ChatModel)(nil)
