package openai

import (
	"context"
	"fmt"
	"regexp"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// chatHTTPURLPattern matches the image URLs Chat Completions downloads itself.
var chatHTTPURLPattern = regexp.MustCompile(`^https?://.*$`)

// ChatModel is a Chat Completions language model.
type ChatModel struct {
	modelID string
	config  config
}

func newChatModel(modelID string, c config) *ChatModel {
	return &ChatModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as "openai.chat".
func (m *ChatModel) Provider() string {
	return m.config.modelProvider("chat")
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that HTTP and HTTPS image URLs are sent to OpenAI
// unchanged.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{
		"image/*": {
			chatHTTPURLPattern,
		},
	}
}

// Generate performs one Chat Completions call.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	call, err := m.prepareCall(options)
	if err != nil {
		return nil, err
	}

	request, err := m.config.request(ctx, "/chat/completions", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, call.request)
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
		Content: chatContent(choice.Message, m.config.generateID),
		FinishReason: model.FinishReason{
			Unified: chatFinishReason(choice.FinishReason),
			Raw:     choice.FinishReason,
		},
		Usage: chatConvertUsage(decoded.Usage),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: chatMetadata(decoded.Usage, choice.Logprobs),
		},
		Warnings: call.warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: chatResponseMetadata(decoded.ID, decoded.Model, decoded.Created),
			Headers:          response.Headers,
			Body:             response.Body,
		},
	}, nil
}

// chatContent returns the text, tool calls and URL citations of a message.
// Audio responses contribute their transcript as text.
func chatContent(message chatResponseMessage, generateID func() string) []model.Content {
	var content []model.Content

	text := message.Content
	if text == "" && message.Audio != nil {
		text = message.Audio.Transcript
	}

	if text != "" {
		content = append(content, model.Text{
			Text: text,
		})
	}

	for _, toolCall := range message.ToolCalls {
		toolCallID := toolCall.ID
		if toolCallID == "" {
			toolCallID = generateID()
		}

		content = append(content, model.ToolCall{
			ToolCallID: toolCallID,
			ToolName:   toolCall.Function.Name,
			Input:      toolCall.Function.Arguments,
		})
	}

	for _, annotation := range message.Annotations {
		content = append(content, chatCitationSource(annotation, generateID))
	}

	return content
}

func chatCitationSource(annotation chatAnnotation, generateID func() string) model.Source {
	return model.Source{
		SourceType: model.SourceURL,
		ID:         generateID(),
		URL:        annotation.URLCitation.URL,
		Title:      annotation.URLCitation.Title,
	}
}

// chatMetadata collects prediction token counts and log probabilities.
func chatMetadata(usage *chatUsage, logprobs *chatLogprobs) ChatMetadata {
	var metadata ChatMetadata

	if usage != nil && usage.CompletionTokensDetails != nil {
		metadata.AcceptedPredictionTokens = usage.CompletionTokensDetails.AcceptedPredictionTokens
		metadata.RejectedPredictionTokens = usage.CompletionTokensDetails.RejectedPredictionTokens
	}

	if logprobs != nil && logprobs.Content != nil {
		metadata.Logprobs = logprobs.Content
	}

	return metadata
}

// chatFinishReason maps a Chat Completions or Completions finish reason.
func chatFinishReason(reason string) model.UnifiedFinishReason {
	switch reason {
	case "stop":
		return model.FinishStop
	case "length":
		return model.FinishLength
	case "content_filter":
		return model.FinishContentFilter
	case "function_call", "tool_calls":
		return model.FinishToolCalls
	default:
		return model.FinishOther
	}
}

// chatConvertUsage maps token usage. Cached prompt tokens count as cache
// reads, and reasoning tokens are subtracted from text tokens.
func chatConvertUsage(usage *chatUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	promptTokens := chatCount(usage.PromptTokens)
	completionTokens := chatCount(usage.CompletionTokens)

	var cachedTokens int
	var cacheWriteTokens *int
	if details := usage.PromptTokensDetails; details != nil {
		cachedTokens = chatCount(details.CachedTokens)
		cacheWriteTokens = details.CacheWriteTokens
	}

	var reasoningTokens int
	if details := usage.CompletionTokensDetails; details != nil {
		reasoningTokens = chatCount(details.ReasoningTokens)
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:      &promptTokens,
			NoCache:    new(promptTokens - cachedTokens - chatCount(cacheWriteTokens)),
			CacheRead:  &cachedTokens,
			CacheWrite: cacheWriteTokens,
		},
		OutputTokens: model.OutputTokens{
			Total:     &completionTokens,
			Text:      new(max(0, completionTokens-reasoningTokens)),
			Reasoning: &reasoningTokens,
		},
		Raw: usage.raw,
	}
}

// chatCount returns a reported count, or zero when it is absent.
func chatCount(count *int) int {
	if count == nil {
		return 0
	}

	return *count
}

// chatResponseMetadata identifies a response. A zero created timestamp, which
// Azure uses as a placeholder, is treated as absent.
func chatResponseMetadata(responseID, modelID string, created int64) model.ResponseMetadata {
	metadata := model.ResponseMetadata{
		ID:      responseID,
		ModelID: modelID,
	}

	if created != 0 {
		metadata.Timestamp = time.Unix(created, 0).UTC()
	}

	return metadata
}
