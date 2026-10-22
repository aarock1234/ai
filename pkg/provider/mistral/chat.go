package mistral

import (
	"context"
	"fmt"
	"regexp"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/pkg/model"
)

// chatPath is the chat completions endpoint.
const chatPath = "/chat/completions"

// documentURLs are the PDF URLs that Mistral downloads itself.
var documentURLs = map[string][]*regexp.Regexp{
	"application/pdf": {
		regexp.MustCompile(`^https://.*$`),
	},
}

// ChatModel is a Mistral chat completions language model.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "mistral.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name + ".chat"
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Mistral downloads https PDF URLs itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return documentURLs
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

	return &model.GenerateResult{
		Content:      generatedContent(choice.Message),
		FinishReason: mapFinishReason(valueOrEmpty(choice.FinishReason)),
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

func (m *ChatModel) prepareArgs(options model.CallOptions) (chatArgs, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return chatArgs{}, err
	}

	completion, err := prepareCompletion(options, chatOptions, !chatModelsWithoutReasoningEffort[m.modelID])
	if err != nil {
		return chatArgs{}, err
	}

	messages, err := convertChatPrompt(completion.Prompt)
	if err != nil {
		return chatArgs{}, err
	}

	tools, err := prepareTools(options.Tools, options.ToolChoice, toolSettings{})
	if err != nil {
		return chatArgs{}, err
	}

	body := chatRequest{
		Model:              m.modelID,
		SafePrompt:         chatOptions.SafePrompt,
		completionSettings: completion.Settings,
		DocumentImageLimit: chatOptions.DocumentImageLimit,
		DocumentPageLimit:  chatOptions.DocumentPageLimit,
		PromptCacheKey:     chatOptions.PromptCacheKey,
		Messages:           messages,
		Tools:              tools.Tools,
		ToolChoice:         tools.ToolChoice,
	}

	if tools.Tools != nil {
		body.ParallelToolCalls = chatOptions.ParallelToolCalls
	}

	return chatArgs{
		body:     body,
		warnings: append(completion.Warnings, tools.Warnings...),
	}, nil
}

// generatedContent converts the message content in order, then the tool
// calls.
func generatedContent(message responseMessage) []model.Content {
	var content []model.Content

	if message.Content.Parts == nil {
		if message.Content.Text != "" {
			content = append(content, model.Text{
				Text: message.Content.Text,
			})
		}
	}

	for _, part := range message.Content.Parts {
		switch part.Type {
		case contentThinking:
			if reasoning := joinThinking(part.Thinking); reasoning != "" {
				content = append(content, model.Reasoning{
					Text: reasoning,
				})
			}

		case contentText:
			if part.Text != "" {
				content = append(content, model.Text{
					Text: part.Text,
				})
			}
		}
	}

	for _, toolCall := range message.ToolCalls {
		content = append(content, model.ToolCall{
			ToolCallID: toolCall.ID,
			ToolName:   toolCall.Function.Name,
			Input:      toolCall.Function.Arguments,
		})
	}

	return content
}

func valueOrEmpty(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}

var _ model.LanguageModel = (*ChatModel)(nil)
