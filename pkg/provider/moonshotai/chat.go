package moonshotai

import (
	"context"
	"fmt"
	"regexp"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// chatPath is the chat completions endpoint.
const chatPath = "/chat/completions"

// supportedURLs are the Moonshot Files API references that Moonshot reads
// itself. Moonshot does not fetch external URLs, so other URL files are
// downloaded and sent inline.
var supportedURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^ms://`),
	},
	"video/*": {
		regexp.MustCompile(`^ms://`),
	},
}

// ChatModel is a Moonshot AI chat completions model.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "moonshotai.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name + ".chat"
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Moonshot reads ms:// file references itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return supportedURLs
}

// Generate performs a non-streaming chat completion.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	body, warnings, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(chatPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
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

	return &model.GenerateResult{
		Content:      m.generatedContent(choice.Message),
		FinishReason: mapFinishReason(choice.FinishReason),
		Usage:        convertUsage(decoded.Usage),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: generateMetadata(decoded, choice),
		},
		Warnings: warnings,
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
	body, warnings, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	body.Stream = true
	body.StreamOptions = &streamOptions{
		IncludeUsage: true,
	}

	request, err := m.provider.apiRequest(chatPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, body)
	if err != nil {
		return nil, err
	}

	processor := newChatStream(chatStreamSettings{
		includeRawChunks: options.IncludeRawChunks,
		generateID:       m.provider.generateID,
	})

	start := model.StreamStart{
		Warnings: warnings,
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

// prepareRequest builds the request body and warnings of a call.
func (m *ChatModel) prepareRequest(options model.CallOptions) (chatRequest, []model.Warning, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return chatRequest{}, nil, err
	}

	var warnings []model.Warning

	if options.TopK != nil {
		warnings = append(warnings, model.UnsupportedWarning("topK", ""))
	}

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning("seed", ""))
	}

	sampling := m.sampling(options)
	warnings = append(warnings, sampling.warnings...)

	tools, err := prepareTools(options.Tools, options.ToolChoice, m.modelID)
	if err != nil {
		return chatRequest{}, nil, err
	}

	reasoningSettings := resolveReasoning(reasoningInputs{
		modelID: m.modelID,
		options: chatOptions,
		level:   options.Reasoning,
	})
	warnings = append(warnings, reasoningSettings.warnings...)

	format, err := responseFormatOf(options.ResponseFormat, chatOptions)
	if err != nil {
		return chatRequest{}, nil, err
	}

	messages, messageWarnings, err := convertPrompt(options.Prompt, promptSettings{
		modelID:          m.modelID,
		jsonObjectFormat: format != nil && format.Type == responseFormatJSONObject,
	})
	if err != nil {
		return chatRequest{}, nil, err
	}

	warnings = append(warnings, messageWarnings...)
	warnings = append(warnings, tools.warnings...)

	body := chatRequest{
		Model:               m.modelID,
		TopLogprobs:         chatOptions.TopLogprobs,
		MaxCompletionTokens: options.MaxOutputTokens,
		Temperature:         sampling.temperature,
		TopP:                sampling.topP,
		FrequencyPenalty:    sampling.frequencyPenalty,
		PresencePenalty:     sampling.presencePenalty,
		ResponseFormat:      format,
		Stop:                options.StopSequences,
		Messages:            messages,
		Tools:               tools.tools,
		ToolChoice:          tools.toolChoice,
		Prediction:          chatOptions.Prediction,
		Thinking:            reasoningSettings.thinking,
		ReasoningEffort:     reasoningSettings.effort,
		PromptCacheKey:      chatOptions.PromptCacheKey,
		SafetyIdentifier:    chatOptions.SafetyIdentifier,
	}

	if optional.Or(chatOptions.Logprobs, false) || chatOptions.TopLogprobs != nil {
		body.Logprobs = new(true)
	}

	return body, warnings, nil
}

// samplingSettings are the sampling members of a request.
type samplingSettings struct {
	temperature      *float64
	topP             *float64
	frequencyPenalty *float64
	presencePenalty  *float64
	warnings         []model.Warning
}

// sampling omits the sampling settings of Kimi models, which fix them.
func (m *ChatModel) sampling(options model.CallOptions) samplingSettings {
	if !ModelFamily(m.modelID).isKimi() {
		return samplingSettings{
			temperature:      options.Temperature,
			topP:             options.TopP,
			frequencyPenalty: options.FrequencyPenalty,
			presencePenalty:  options.PresencePenalty,
		}
	}

	var settings samplingSettings

	fixed := []struct {
		feature string
		set     bool
	}{
		{
			feature: "temperature",
			set:     options.Temperature != nil,
		},
		{
			feature: "topP",
			set:     options.TopP != nil,
		},
		{
			feature: "frequencyPenalty",
			set:     options.FrequencyPenalty != nil,
		},
		{
			feature: "presencePenalty",
			set:     options.PresencePenalty != nil,
		},
	}

	for _, setting := range fixed {
		if !setting.set {
			continue
		}

		settings.warnings = append(settings.warnings, model.UnsupportedWarning(
			setting.feature,
			fmt.Sprintf("%s is fixed by model \"%s\" and has been omitted.", setting.feature, m.modelID),
		))
	}

	return settings
}

// responseFormatOf requests a JSON schema when the call has one and a JSON
// object otherwise. The top-level $schema keyword is removed, because
// kimi-k2.5 produces nonsensical output when it is present.
func responseFormatOf(format *model.ResponseFormat, options ChatOptions) (*responseFormat, error) {
	if format == nil || format.Type != model.ResponseFormatJSON {
		return nil, nil
	}

	if format.Schema == nil {
		return &responseFormat{
			Type: responseFormatJSONObject,
		}, nil
	}

	schema := format.Schema
	if object, err := jsonobject.Parse(schema); err == nil {
		schema, err = object.Without("$schema").Encode()
		if err != nil {
			return nil, err
		}
	}

	normalized, err := normalizeSchema(schema)
	if err != nil {
		return nil, err
	}

	name := format.Name
	if name == "" {
		name = "response"
	}

	return &responseFormat{
		Type: responseFormatJSONSchema,
		JSONSchema: &jsonSchema{
			Name:   name,
			Strict: optional.Or(options.StrictJSONSchema, true),
			Schema: normalized,
		},
	}, nil
}

// generatedContent converts a response message: reasoning, then tool calls,
// then text.
func (m *ChatModel) generatedContent(message chatResponseMessage) []model.Content {
	var content []model.Content

	if reasoning := optional.Or(message.ReasoningContent, ""); reasoning != "" {
		content = append(content, model.Reasoning{
			Text: reasoning,
		})
	}

	for _, toolCall := range message.ToolCalls {
		toolCallID := optional.Or(toolCall.ID, "")
		if toolCallID == "" {
			toolCallID = m.provider.generateID()
		}

		content = append(content, model.ToolCall{
			ToolCallID: toolCallID,
			ToolName:   toolCall.Function.Name,
			Input:      toolCall.Function.Arguments,
		})
	}

	if text := optional.Or(message.Content, ""); text != "" {
		content = append(content, model.Text{
			Text: text,
		})
	}

	return content
}

// generateMetadata describes the response choice.
func generateMetadata(response chatResponse, choice chatChoice) ChatMetadata {
	metadata := ChatMetadata{
		Logprobs:       choice.Logprobs,
		ResponseObject: optional.Or(response.Object, ""),
		ChoiceIndex:    choice.Index,
		MessageRole:    optional.Or(choice.Message.Role, ""),
	}

	for _, toolCall := range choice.Message.ToolCalls {
		if toolCall.Type != nil {
			metadata.ToolCallTypes = append(metadata.ToolCallTypes, *toolCall.Type)
		}
	}

	return metadata
}

var _ model.LanguageModel = (*ChatModel)(nil)
