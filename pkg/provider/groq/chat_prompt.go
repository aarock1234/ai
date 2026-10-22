package groq

import (
	"fmt"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// convertChatPrompt converts the prompt into Groq chat messages.
func convertChatPrompt(prompt []model.Message) ([]chatMessage, error) {
	messages := make([]chatMessage, 0, len(prompt))

	for _, message := range prompt {
		switch message := message.(type) {
		case model.SystemMessage:
			messages = append(messages, systemMessage{
				Role:    "system",
				Content: message.Content,
			})

		case model.UserMessage:
			content, err := userMessageContent(message)
			if err != nil {
				return nil, err
			}

			messages = append(messages, userMessage{
				Role:    "user",
				Content: content,
			})

		case model.AssistantMessage:
			converted, err := convertAssistantMessage(message)
			if err != nil {
				return nil, err
			}

			messages = append(messages, converted)

		case model.ToolMessage:
			converted, err := convertToolMessage(message)
			if err != nil {
				return nil, err
			}

			messages = append(messages, converted...)

		default:
			return nil, fmt.Errorf("%w: unsupported message role %q", model.ErrInvalidPrompt, message.Role())
		}
	}

	return messages, nil
}

// userMessageContent sends a single text part as plain text and other
// content as text and image parts.
func userMessageContent(message model.UserMessage) (userContent, error) {
	if len(message.Content) == 1 {
		if text, ok := message.Content[0].(model.TextPart); ok {
			return userContent{
				Text: text.Text,
			}, nil
		}
	}

	parts := make([]contentPart, 0, len(message.Content))

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			parts = append(parts, contentPart{
				Type: contentPartText,
				Text: new(part.Text),
			})

		case model.FilePart:
			converted, err := imagePart(part)
			if err != nil {
				return userContent{}, err
			}

			parts = append(parts, converted)

		default:
			return userContent{}, fmt.Errorf("%w: user part %T", model.ErrUnsupported, part)
		}
	}

	return userContent{
		Parts: parts,
	}, nil
}

// imagePart sends image URLs as they are and inline images as data URLs.
// Groq accepts no other files.
func imagePart(part model.FilePart) (contentPart, error) {
	switch data := part.Data.(type) {
	case model.FileReference:
		return contentPart{}, fmt.Errorf("%w: file parts with provider references", model.ErrUnsupported)

	case model.FileText:
		return contentPart{}, fmt.Errorf("%w: text file parts", model.ErrUnsupported)

	case model.FileURL:
		if mediatype.TopLevel(part.MediaType) != "image" {
			return contentPart{}, fmt.Errorf("%w: non-image file content parts", model.ErrUnsupported)
		}

		return contentPart{
			Type: contentPartImageURL,
			ImageURL: &imageURL{
				URL: data.URL.String(),
			},
		}, nil

	case model.FileBytes:
		if mediatype.TopLevel(part.MediaType) != "image" {
			return contentPart{}, fmt.Errorf("%w: non-image file content parts", model.ErrUnsupported)
		}

		mediaType, err := mediatype.ResolveFile(part.MediaType, part.Data)
		if err != nil {
			return contentPart{}, err
		}

		return contentPart{
			Type: contentPartImageURL,
			ImageURL: &imageURL{
				URL: dataurl.Encode(mediaType, data.Data),
			},
		}, nil

	default:
		return contentPart{}, fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, data)
	}
}

// convertAssistantMessage joins the text and reasoning of an assistant
// message. Groq accepts reasoning with tool calls in multi-turn
// conversations.
func convertAssistantMessage(message model.AssistantMessage) (assistantMessage, error) {
	var text string
	var reasoning string
	var toolCalls []messageToolCall

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.ReasoningPart:
			reasoning += part.Text

		case model.TextPart:
			text += part.Text

		case model.ToolCallPart:
			arguments, err := toolresult.CompactJSON(part.Input)
			if err != nil {
				return assistantMessage{}, err
			}

			toolCalls = append(toolCalls, messageToolCall{
				ID:   part.ToolCallID,
				Type: "function",
				Function: functionCall{
					Name:      part.ToolName,
					Arguments: arguments,
				},
			})
		}
	}

	return assistantMessage{
		Role:      "assistant",
		Content:   text,
		Reasoning: reasoning,
		ToolCalls: toolCalls,
	}, nil
}

// convertToolMessage returns one message per tool result. Approval responses
// are not sent.
func convertToolMessage(message model.ToolMessage) ([]chatMessage, error) {
	var messages []chatMessage

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			continue
		}

		content, err := toolresult.Text(result.Output)
		if err != nil {
			return nil, err
		}

		messages = append(messages, toolMessage{
			Role:       "tool",
			Content:    content,
			ToolCallID: result.ToolCallID,
		})
	}

	return messages, nil
}
