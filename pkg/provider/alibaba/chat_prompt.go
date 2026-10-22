package alibaba

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// promptConverter converts one prompt. Cache controls of a message apply to
// its last part unless the part has its own.
type promptConverter struct {
	cache            *cacheControls
	preserveThinking bool
	lastUserIndex    int
}

// convertChatPrompt converts the prompt into Alibaba chat messages.
// Reasoning from earlier rounds is replayed only with preserved thinking.
func convertChatPrompt(prompt []model.Message, cache *cacheControls, preserveThinking bool) ([]chatMessage, error) {
	converter := promptConverter{
		cache:            cache,
		preserveThinking: preserveThinking,
		lastUserIndex:    lastUserIndex(prompt),
	}

	messages := make([]chatMessage, 0, len(prompt))

	for index, message := range prompt {
		converted, err := converter.convertMessage(message, index)
		if err != nil {
			return nil, err
		}

		messages = append(messages, converted...)
	}

	return messages, nil
}

func lastUserIndex(prompt []model.Message) int {
	for index := len(prompt) - 1; index >= 0; index-- {
		if prompt[index].Role() == model.RoleUser {
			return index
		}
	}

	return -1
}

func (c promptConverter) convertMessage(message model.Message, index int) ([]chatMessage, error) {
	switch message := message.(type) {
	case model.SystemMessage:
		converted, err := c.convertSystemMessage(message)
		if err != nil {
			return nil, err
		}

		return []chatMessage{
			converted,
		}, nil

	case model.UserMessage:
		converted, err := c.convertUserMessage(message)
		if err != nil {
			return nil, err
		}

		return []chatMessage{
			converted,
		}, nil

	case model.AssistantMessage:
		return c.convertAssistantMessage(message, index)

	case model.ToolMessage:
		return c.convertToolMessage(message)

	default:
		return nil, fmt.Errorf("%w: unsupported message role %q", model.ErrInvalidPrompt, message.Role())
	}
}

// convertSystemMessage sends a cached system message as a text part.
func (c promptConverter) convertSystemMessage(message model.SystemMessage) (systemMessage, error) {
	cacheControl, err := c.cache.read(message.ProviderOptions)
	if err != nil {
		return systemMessage{}, err
	}

	return systemMessage{
		Role:    "system",
		Content: cachedText(message.Content, cacheControl),
	}, nil
}

// convertUserMessage sends text and image parts. The message's cache control
// applies to its last part.
func (c promptConverter) convertUserMessage(message model.UserMessage) (userMessage, error) {
	messageCache, err := c.cache.read(message.ProviderOptions)
	if err != nil {
		return userMessage{}, err
	}

	parts := make([]userPart, 0, len(message.Content))

	for index, part := range message.Content {
		var fallback *CacheControl
		if index == len(message.Content)-1 {
			fallback = messageCache
		}

		converted, err := c.userPart(part, fallback)
		if err != nil {
			return userMessage{}, err
		}

		parts = append(parts, converted)
	}

	return userMessage{
		Role:    "user",
		Content: parts,
	}, nil
}

// userPart converts a text or image part. Alibaba accepts no other files.
func (c promptConverter) userPart(part model.UserPart, fallback *CacheControl) (userPart, error) {
	switch part := part.(type) {
	case model.TextPart:
		cacheControl, err := c.partCacheControl(part.ProviderOptions, fallback)
		if err != nil {
			return userPart{}, err
		}

		return userPart{
			Type:         userPartText,
			Text:         new(part.Text),
			CacheControl: cacheControl,
		}, nil

	case model.FilePart:
		cacheControl, err := c.partCacheControl(part.ProviderOptions, fallback)
		if err != nil {
			return userPart{}, err
		}

		address, err := imageAddress(part)
		if err != nil {
			return userPart{}, err
		}

		return userPart{
			Type: userPartImageURL,
			ImageURL: &imageURL{
				URL: address,
			},
			CacheControl: cacheControl,
		}, nil

	default:
		return userPart{}, fmt.Errorf("%w: user part %T", model.ErrUnsupported, part)
	}
}

// imageAddress returns the URL of an image URL or a data URL of inline
// image bytes.
func imageAddress(part model.FilePart) (string, error) {
	switch part.Data.(type) {
	case model.FileReference:
		return "", fmt.Errorf("%w: file parts with provider references", model.ErrUnsupported)

	case model.FileText:
		return "", fmt.Errorf("%w: text file parts", model.ErrUnsupported)
	}

	if mediatype.TopLevel(part.MediaType) != "image" {
		return "", fmt.Errorf("%w: only image file parts are supported", model.ErrUnsupported)
	}

	switch data := part.Data.(type) {
	case model.FileURL:
		return data.URL.String(), nil

	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(part.MediaType, part.Data)
		if err != nil {
			return "", err
		}

		return dataurl.Encode(mediaType, data.Data), nil

	default:
		return "", fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, data)
	}
}

// convertAssistantMessage joins the text, reasoning and tool calls of an
// assistant message. Reasoning of the current round always accompanies its
// tool calls; earlier rounds are replayed only with preserved thinking.
// Messages without content are dropped.
func (c promptConverter) convertAssistantMessage(message model.AssistantMessage, index int) ([]chatMessage, error) {
	messageCache, err := c.cache.read(message.ProviderOptions)
	if err != nil {
		return nil, err
	}

	var text strings.Builder
	var reasoning strings.Builder
	var toolCalls []messageToolCall

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			text.WriteString(part.Text)

		case model.ToolCallPart:
			arguments, err := toolresult.CompactJSON(part.Input)
			if err != nil {
				return nil, err
			}

			toolCalls = append(toolCalls, messageToolCall{
				ID:   part.ToolCallID,
				Type: functionType,
				Function: functionCall{
					Name:      part.ToolName,
					Arguments: arguments,
				},
			})

		case model.ReasoningPart:
			if c.preserveThinking || index > c.lastUserIndex {
				reasoning.WriteString(part.Text)
			}
		}
	}

	if text.Len() == 0 && len(toolCalls) == 0 && reasoning.Len() == 0 {
		return nil, nil
	}

	return []chatMessage{
		assistantMessage{
			Role:             "assistant",
			Content:          assistantMessageContent(text.String(), len(toolCalls) > 0, messageCache),
			ReasoningContent: reasoning.String(),
			ToolCalls:        toolCalls,
		},
	}, nil
}

// assistantMessageContent is null for reasoning-only messages and empty
// text, a cached text part, or the text otherwise.
func assistantMessageContent(text string, hasToolCalls bool, cacheControl *CacheControl) assistantContent {
	if text == "" && !hasToolCalls {
		return assistantContent{}
	}

	if cacheControl != nil {
		return assistantContent{
			Parts: []textPart{
				{
					Type:         "text",
					Text:         text,
					CacheControl: cacheControl,
				},
			},
		}
	}

	if text == "" {
		return assistantContent{}
	}

	return assistantContent{
		Text: new(text),
	}
}

// convertToolMessage returns one message per tool result. The message's
// cache control applies to its last result. Approval responses are not sent.
func (c promptConverter) convertToolMessage(message model.ToolMessage) ([]chatMessage, error) {
	messageCache, err := c.cache.read(message.ProviderOptions)
	if err != nil {
		return nil, err
	}

	var results []model.ToolResultPart
	for _, part := range message.Content {
		if result, ok := part.(model.ToolResultPart); ok {
			results = append(results, result)
		}
	}

	messages := make([]chatMessage, 0, len(results))

	for index, result := range results {
		var fallback *CacheControl
		if index == len(results)-1 {
			fallback = messageCache
		}

		cacheControl, err := c.partCacheControl(result.ProviderOptions, fallback)
		if err != nil {
			return nil, err
		}

		content, err := toolresult.Text(result.Output)
		if err != nil {
			return nil, err
		}

		messages = append(messages, toolMessage{
			Role:       "tool",
			ToolCallID: result.ToolCallID,
			Content:    cachedText(content, cacheControl),
		})
	}

	return messages, nil
}

// partCacheControl prefers the part's own cache control over the fallback.
func (c promptConverter) partCacheControl(options model.ProviderOptions, fallback *CacheControl) (*CacheControl, error) {
	cacheControl, err := c.cache.read(options)
	if err != nil {
		return nil, err
	}

	if cacheControl != nil {
		return cacheControl, nil
	}

	return fallback, nil
}

// cachedText sends text as a cached text part when it has a cache control.
func cachedText(text string, cacheControl *CacheControl) textContent {
	if cacheControl == nil {
		return textContent{
			Text: text,
		}
	}

	return textContent{
		Parts: []textPart{
			{
				Type:         "text",
				Text:         text,
				CacheControl: cacheControl,
			},
		},
	}
}
