package groq

import (
	"context"
	"fmt"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

// chatPath is the chat completions endpoint.
const chatPath = "/chat/completions"

// noneReasoningModel is the model that accepts the "none" reasoning effort.
const noneReasoningModel = "qwen/qwen3.6-27b"

// jsonToolDescription describes the JSON response tool without a response
// format description.
const jsonToolDescription = "Respond with a JSON object."

// imageURLs are the image URLs that Groq downloads itself.
var imageURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
}

// reasoningEfforts maps provider-independent reasoning levels to Groq
// efforts.
var reasoningEfforts = map[model.ReasoningEffort]ReasoningEffort{
	model.ReasoningMinimal: ReasoningEffortLow,
	model.ReasoningLow:     ReasoningEffortLow,
	model.ReasoningMedium:  ReasoningEffortMedium,
	model.ReasoningHigh:    ReasoningEffortHigh,
	model.ReasoningXHigh:   ReasoningEffortHigh,
	model.ReasoningMax:     ReasoningEffortHigh,
}

// ChatModel is a Groq chat completions language model.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "groq.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name + ".chat"
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Groq downloads image URLs itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return imageURLs
}

// chatArgs is a prepared request.
type chatArgs struct {
	body     chatRequest
	warnings []model.Warning

	// jsonToolName names the tool that carries JSON output, when Groq cannot
	// combine a JSON response format with function tools.
	jsonToolName string
}

// Generate performs a non-streaming chat completion.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	args, err := m.prepareArgs(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(chatPath, options.Headers)
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

	if len(decoded.Choices) == 0 {
		return nil, fmt.Errorf("%w: response did not contain any choices", model.ErrInvalidResponse)
	}

	choice := decoded.Choices[0]

	content, err := m.generatedContent(choice.Message, args.jsonToolName)
	if err != nil {
		return nil, err
	}

	return &model.GenerateResult{
		Content:      content,
		FinishReason: generateFinishReason(choice.FinishReason, content, args.jsonToolName),
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

// Stream performs a streaming chat completion.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	args, err := m.prepareArgs(options)
	if err != nil {
		return nil, err
	}

	args.body.Stream = true

	request, err := m.provider.apiRequest(chatPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, args.body)
	if err != nil {
		return nil, err
	}

	processor := newChatStream(chatStreamSettings{
		includeRawChunks: options.IncludeRawChunks,
		generateID:       m.provider.generateID,
		origin: streamOrigin{
			url:     request.URL,
			headers: response.Headers,
		},
	})

	start := model.StreamStart{
		Warnings: args.warnings,
	}

	parts := chatsse.Parts(ctx, response.Body, start, processor)
	if args.jsonToolName != "" {
		parts = jsonToolParts(parts, args.jsonToolName)
	}

	result := model.NewStreamResult(parts, response.Body.Close)
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

	jsonTool := jsonResponseTool(options.ResponseFormat, options.Tools)
	structuredOutputs := optional.Or(chatOptions.StructuredOutputs, true)

	if options.TopK != nil {
		warnings = append(warnings, model.UnsupportedWarning("topK", ""))
	}

	if hasSchema(options.ResponseFormat) && !structuredOutputs && jsonTool == nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"responseFormat",
			"JSON response format schema is only supported with structuredOutputs",
		))
	}

	tools, toolChoice := options.Tools, options.ToolChoice
	if jsonTool != nil {
		tools = append(slices.Clone(options.Tools), *jsonTool)
		toolChoice = jsonToolChoice(options.ToolChoice, jsonTool.Name)
	}

	prepared, err := prepareTools(tools, toolChoice, m.modelID)
	if err != nil {
		return chatArgs{}, err
	}

	effort, effortWarnings := m.reasoningEffort(chatOptions, options.Reasoning)
	warnings = append(warnings, effortWarnings...)

	messages, err := convertChatPrompt(options.Prompt)
	if err != nil {
		return chatArgs{}, err
	}

	body := chatRequest{
		Model:             m.modelID,
		User:              chatOptions.User,
		ParallelToolCalls: chatOptions.ParallelToolCalls,
		MaxTokens:         options.MaxOutputTokens,
		Temperature:       options.Temperature,
		TopP:              options.TopP,
		FrequencyPenalty:  options.FrequencyPenalty,
		PresencePenalty:   options.PresencePenalty,
		Stop:              options.StopSequences,
		Seed:              options.Seed,
		ReasoningFormat:   chatOptions.ReasoningFormat,
		ReasoningEffort:   effort,
		ServiceTier:       chatOptions.ServiceTier,
		Messages:          messages,
		Tools:             prepared.Tools,
		ToolChoice:        prepared.ToolChoice,
	}

	args := chatArgs{
		body:     body,
		warnings: append(warnings, prepared.Warnings...),
	}

	if jsonTool != nil {
		args.body.ParallelToolCalls = new(false)
		args.jsonToolName = jsonTool.Name

		return args, nil
	}

	args.body.ResponseFormat = responseFormatOf(options.ResponseFormat, structuredOutputs, optional.Or(chatOptions.StrictJSONSchema, true))

	return args, nil
}

// reasoningEffort prefers the provider option and maps the
// provider-independent reasoning level otherwise. Only one model accepts
// disabling reasoning.
func (m *ChatModel) reasoningEffort(options ChatOptions, level model.ReasoningEffort) (ReasoningEffort, []model.Warning) {
	if options.ReasoningEffort != "" || !reasoning.IsCustom(level) {
		return options.ReasoningEffort, nil
	}

	if level == model.ReasoningNone {
		if m.modelID == noneReasoningModel {
			return ReasoningEffortNone, nil
		}

		return "", []model.Warning{
			model.UnsupportedWarning("reasoning", fmt.Sprintf("reasoning %q is not supported by this model.", level)),
		}
	}

	effort, warnings, _ := reasoning.Effort(level, reasoningEfforts)

	return effort, warnings
}

// responseFormatOf requests a JSON schema when structured outputs are enabled
// and a JSON object otherwise.
func responseFormatOf(format *model.ResponseFormat, structuredOutputs bool, strict bool) *responseFormat {
	if format == nil || format.Type != model.ResponseFormatJSON {
		return nil
	}

	if !structuredOutputs || format.Schema == nil {
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
			Strict:      strict,
			Name:        name,
			Description: format.Description,
		},
	}
}

// generatedContent converts a response message: reasoning, then text, then
// tool calls. The JSON response tool's arguments become the text.
func (m *ChatModel) generatedContent(message responseMessage, jsonToolName string) ([]model.Content, error) {
	var content []model.Content

	if text := valueOrEmpty(message.Reasoning); text != "" {
		content = append(content, model.Reasoning{
			Text: text,
		})
	}

	if text := valueOrEmpty(message.Content); jsonToolName == "" && text != "" {
		content = append(content, model.Text{
			Text: text,
		})
	}

	for _, toolCall := range message.ToolCalls {
		if toolCall.Type != "function" {
			return nil, fmt.Errorf("%w: expected 'function' tool call type", model.ErrInvalidResponse)
		}

		if jsonToolName != "" && toolCall.Function.Name == jsonToolName {
			content = append(content, model.Text{
				Text: toolCall.Function.Arguments,
			})

			continue
		}

		toolCallID := valueOrEmpty(toolCall.ID)
		if toolCallID == "" {
			toolCallID = m.provider.generateID()
		}

		content = append(content, model.ToolCall{
			ToolCallID: toolCallID,
			ToolName:   toolCall.Function.Name,
			Input:      toolCall.Function.Arguments,
		})
	}

	return content, nil
}

// generateFinishReason reports a response that only called the JSON
// response tool as stopped.
func generateFinishReason(finishReason *string, content []model.Content, jsonToolName string) model.FinishReason {
	mapped := mapFinishReason(finishReason)

	if jsonToolName != "" && mapped.Raw == "tool_calls" && hasText(content) && !hasToolCall(content) {
		mapped.Unified = model.FinishStop
	}

	return mapped
}

func hasText(content []model.Content) bool {
	return slices.ContainsFunc(content, func(item model.Content) bool {
		_, ok := item.(model.Text)

		return ok
	})
}

func hasToolCall(content []model.Content) bool {
	return slices.ContainsFunc(content, func(item model.Content) bool {
		_, ok := item.(model.ToolCall)

		return ok
	})
}

func hasSchema(format *model.ResponseFormat) bool {
	return format != nil && format.Type == model.ResponseFormatJSON && format.Schema != nil
}

var _ model.LanguageModel = (*ChatModel)(nil)
