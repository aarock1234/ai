package zai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"maps"
	"regexp"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

// supportedURLs are the image and video URLs that Z.AI downloads itself.
var supportedURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://`),
	},
	"video/*": {
		regexp.MustCompile(`^https?://`),
	},
}

// removedBodyMembers are OpenAI-compatible request members that Z.AI does
// not accept.
var removedBodyMembers = []string{
	"frequency_penalty",
	"presence_penalty",
	"seed",
	"user",
	"verbosity",
}

// ChatModel is a Z.AI chat model. It sends OpenAI-compatible chat
// completions with Z.AI's request extensions and finish reasons.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "zai.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name + ".chat"
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Z.AI downloads image and video URLs itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return supportedURLs
}

// Generate performs a non-streaming chat completion.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	call, err := prepareCall(options)
	if err != nil {
		return nil, err
	}

	result, err := m.chat(call.chatOptions).Generate(ctx, call.options)
	if err != nil {
		return nil, err
	}

	result.FinishReason = mapFinishReason(result.FinishReason)
	result.Warnings = append(result.Warnings, call.warnings...)

	return result, nil
}

// Stream performs a streaming chat completion.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	call, err := prepareCall(options)
	if err != nil {
		return nil, err
	}

	inner, err := m.chat(call.chatOptions).Stream(ctx, call.options)
	if err != nil {
		return nil, err
	}

	result := model.NewStreamResult(streamParts(inner.Parts(), call.warnings), inner.Close)
	result.Request = inner.Request
	result.Response = inner.Response

	return result, nil
}

// chat returns the OpenAI-compatible model of one call. The request body
// transform adds the call's Z.AI options.
func (m *ChatModel) chat(chatOptions *ChatOptions) *openaicompatible.ChatModel {
	transform := func(body map[string]json.RawMessage) (map[string]json.RawMessage, []model.Warning, error) {
		return transformRequestBody(body, chatOptions)
	}

	return openaicompatible.NewChatModel(m.modelID, openaicompatible.ChatConfig{
		Provider:             Namespace + ".chat",
		BuildURL:             openaicompatible.InfallibleURL(m.provider.url),
		Headers:              m.provider.requestHeaders,
		HTTPClient:           m.provider.httpClient,
		ParseError:           parseError,
		SupportedURLs:        supportedURLs,
		TransformRequestBody: transform,
		GenerateID:           m.provider.generateID,
	})
}

// preparedCall is a call normalized for Z.AI.
type preparedCall struct {
	options     model.CallOptions
	chatOptions *ChatOptions
	warnings    []model.Warning
}

// prepareCall validates the Z.AI options and removes the settings that Z.AI
// does not support, with warnings. The Z.AI options are sent by the request
// body transform, so their namespace is removed from the forwarded options.
func prepareCall(options model.CallOptions) (preparedCall, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return preparedCall{}, err
	}

	var warnings []model.Warning

	if options.FrequencyPenalty != nil {
		warnings = append(warnings, model.UnsupportedWarning("frequencyPenalty", ""))
	}

	if options.PresencePenalty != nil {
		warnings = append(warnings, model.UnsupportedWarning("presencePenalty", ""))
	}

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning("seed", ""))
	}

	options.FrequencyPenalty = nil
	options.PresencePenalty = nil
	options.Seed = nil

	warnings = append(warnings, normalizeToolChoice(&options)...)

	forwarded := maps.Clone(options.ProviderOptions)
	delete(forwarded, Namespace)
	options.ProviderOptions = forwarded

	return preparedCall{
		options:     options,
		chatOptions: chatOptions,
		warnings:    warnings,
	}, nil
}

// normalizeToolChoice implements tool choice "none" by omitting the tools
// and drops every other choice except "auto", which is the only one Z.AI
// supports.
func normalizeToolChoice(options *model.CallOptions) []model.Warning {
	if options.ToolChoice == nil {
		return nil
	}

	switch options.ToolChoice.Type {
	case model.ToolChoiceAuto:
		return nil

	case model.ToolChoiceNone:
		options.Tools = nil
		options.ToolChoice = nil

		return nil

	default:
		warning := model.UnsupportedWarning(
			"toolChoice "+string(options.ToolChoice.Type),
			"Z.AI currently supports only automatic tool selection.",
		)
		options.ToolChoice = nil

		return []model.Warning{
			warning,
		}
	}
}

// requestExtension is the Z.AI-specific part of a chat request.
type requestExtension struct {
	DoSample        *bool            `json:"do_sample,omitempty"`
	Thinking        *thinkingRequest `json:"thinking,omitempty"`
	ReasoningEffort *ReasoningEffort `json:"reasoning_effort,omitempty"`
	ToolStream      *bool            `json:"tool_stream,omitempty"`
	RequestID       *string          `json:"request_id,omitempty"`
	UserID          *string          `json:"user_id,omitempty"`
}

type thinkingRequest struct {
	Type          *ThinkingType `json:"type,omitempty"`
	ClearThinking *bool         `json:"clear_thinking,omitempty"`
}

// transformRequestBody removes the members Z.AI does not accept and adds the
// call's Z.AI options.
func transformRequestBody(body map[string]json.RawMessage, chatOptions *ChatOptions) (map[string]json.RawMessage, []model.Warning, error) {
	for _, member := range removedBodyMembers {
		delete(body, member)
	}

	if chatOptions == nil {
		return body, nil, nil
	}

	extension := requestExtension{
		DoSample:        chatOptions.DoSample,
		ReasoningEffort: chatOptions.ReasoningEffort,
		ToolStream:      chatOptions.ToolStream,
		RequestID:       chatOptions.RequestID,
		UserID:          chatOptions.UserID,
	}

	if chatOptions.Thinking != nil {
		extension.Thinking = &thinkingRequest{
			Type:          chatOptions.Thinking.Type,
			ClearThinking: chatOptions.Thinking.ClearThinking,
		}
	}

	members, err := jsonobject.Members(extension)
	if err != nil {
		return nil, nil, fmt.Errorf("encode zai request options: %w", err)
	}

	maps.Copy(body, members)

	return body, nil, nil
}

// streamParts adds the call warnings to the stream start and maps the
// finish reason.
func streamParts(inner iter.Seq[model.StreamPart], warnings []model.Warning) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		for part := range inner {
			switch converted := part.(type) {
			case model.StreamStart:
				converted.Warnings = append(converted.Warnings, warnings...)
				part = converted

			case model.Finish:
				converted.FinishReason = mapFinishReason(converted.FinishReason)
				part = converted
			}

			if !yield(part) {
				return
			}
		}
	}
}

// mapFinishReason maps Z.AI's own finish reasons.
func mapFinishReason(finishReason model.FinishReason) model.FinishReason {
	switch finishReason.Raw {
	case "sensitive":
		finishReason.Unified = model.FinishContentFilter
	case "model_context_window_exceeded":
		finishReason.Unified = model.FinishLength
	case "network_error":
		finishReason.Unified = model.FinishError
	}

	return finishReason
}

var _ model.LanguageModel = (*ChatModel)(nil)
