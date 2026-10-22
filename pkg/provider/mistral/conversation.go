package mistral

import (
	"context"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

// conversationPath is the Conversations API endpoint.
const conversationPath = "/conversations"

// ConversationModel is a Mistral language model that uses the Conversations
// API. Conversations are not stored, and built-in web search runs on
// Mistral's servers.
type ConversationModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "mistral.conversation" by default.
func (m *ConversationModel) Provider() string {
	return m.provider.name + ".conversation"
}

// ModelID returns the model identifier.
func (m *ConversationModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Mistral downloads https PDF URLs itself.
func (m *ConversationModel) SupportedURLs() map[string][]*regexp.Regexp {
	return documentURLs
}

// conversationArgs is a prepared request.
type conversationArgs struct {
	body     conversationRequest
	warnings []model.Warning
}

// Generate performs a non-streaming conversation turn.
func (m *ConversationModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	args, err := m.prepareArgs(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(conversationPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, args.body)
	if err != nil {
		return nil, err
	}

	var decoded conversationResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	converter := newContentConverter(options.Tools, m.provider.generateID)
	output, err := convertOutputs(converter, decoded.Outputs)
	if err != nil {
		return nil, err
	}

	finishReason := model.FinishReason{
		Unified: model.FinishStop,
	}
	if output.hasFunctionCalls {
		finishReason.Unified = model.FinishToolCalls
	}

	return &model.GenerateResult{
		Content:      output.content,
		FinishReason: finishReason,
		Usage:        convertUsage(decoded.Usage),
		Warnings:     args.warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID:      decoded.ConversationID,
				ModelID: output.modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// Stream performs a streaming conversation turn.
func (m *ConversationModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	args, err := m.prepareArgs(options)
	if err != nil {
		return nil, err
	}

	args.body.Stream = true

	request, err := m.provider.apiRequest(conversationPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, args.body)
	if err != nil {
		return nil, err
	}

	processor := newConversationStream(conversationStreamSettings{
		includeRawChunks: options.IncludeRawChunks,
		converter:        newContentConverter(options.Tools, m.provider.generateID),
		generateID:       m.provider.generateID,
	})

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

func (m *ConversationModel) prepareArgs(options model.CallOptions) (conversationArgs, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return conversationArgs{}, err
	}

	completion, err := prepareCompletion(options, chatOptions, conversationModelsWithReasoningEffort[m.modelID])
	if err != nil {
		return conversationArgs{}, err
	}

	tools, err := prepareTools(options.Tools, options.ToolChoice, toolSettings{
		builtIn: true,
	})
	if err != nil {
		return conversationArgs{}, err
	}

	prompt, err := convertConversationPrompt(completion.Prompt, toolnames.New(options.Tools, providerToolNames))
	if err != nil {
		return conversationArgs{}, err
	}

	warnings := slices.Concat(
		completion.Warnings,
		unsupportedConversationOptions(chatOptions),
		tools.Warnings,
	)

	return conversationArgs{
		body: conversationRequest{
			Model:        m.modelID,
			Store:        false,
			Inputs:       prompt.Inputs,
			Instructions: prompt.Instructions,
			Tools:        tools.Tools,
			CompletionArgs: completionArgs{
				completionSettings: completion.Settings,
				ToolChoice:         tools.ToolChoice,
			},
		},
		warnings: warnings,
	}, nil
}

// unsupportedConversationOptions warns about chat options that the
// Conversations API does not accept.
func unsupportedConversationOptions(options ChatOptions) []model.Warning {
	settings := []struct {
		feature string
		set     bool
	}{
		{
			feature: "safePrompt",
			set:     options.SafePrompt != nil,
		},
		{
			feature: "documentImageLimit",
			set:     options.DocumentImageLimit != nil,
		},
		{
			feature: "documentPageLimit",
			set:     options.DocumentPageLimit != nil,
		},
		{
			feature: "promptCacheKey",
			set:     options.PromptCacheKey != nil,
		},
		{
			feature: "parallelToolCalls",
			set:     options.ParallelToolCalls != nil,
		},
	}

	var warnings []model.Warning
	for _, setting := range settings {
		if setting.set {
			warnings = append(warnings, model.UnsupportedWarning(setting.feature, "Not supported by the Mistral Conversations API."))
		}
	}

	return warnings
}

// convertedOutputs is the content of a conversation response.
type convertedOutputs struct {
	content          []model.Content
	hasFunctionCalls bool

	// modelID is the model of the last message output.
	modelID string
}

func convertOutputs(converter *contentConverter, outputs []conversationOutput) (convertedOutputs, error) {
	var converted convertedOutputs

	for _, output := range outputs {
		switch {
		case output.ToolExecution != nil:
			call, result, err := converter.toolExecution(*output.ToolExecution)
			if err != nil {
				return convertedOutputs{}, err
			}

			converted.content = append(converted.content, call, result)

		case output.FunctionCall != nil:
			converted.hasFunctionCalls = true
			converted.content = append(converted.content, model.ToolCall{
				ToolCallID: output.FunctionCall.ToolCallID,
				ToolName:   output.FunctionCall.Name,
				Input:      output.FunctionCall.Arguments.Text,
			})

		case output.Message != nil:
			if output.Message.Model != nil {
				converted.modelID = *output.Message.Model
			}

			for _, part := range output.Message.Content.Parts {
				if content, ok := converter.part(part); ok {
					converted.content = append(converted.content, content)
				}
			}
		}
	}

	return converted, nil
}

var _ model.LanguageModel = (*ConversationModel)(nil)
