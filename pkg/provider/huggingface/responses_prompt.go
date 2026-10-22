package huggingface

import (
	"fmt"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// Input roles and part types.
const (
	roleSystem    = "system"
	roleUser      = "user"
	roleAssistant = "assistant"

	partInputText  = "input_text"
	partInputImage = "input_image"
	partOutputText = "output_text"
)

// convertPrompt converts the prompt into Responses API input messages.
// Assistant text and reasoning become output text messages; tool calls,
// tool results and tool messages are not sent.
func convertPrompt(prompt []model.Message) ([]inputMessage, []model.Warning, error) {
	messages := []inputMessage{}
	var warnings []model.Warning

	for _, message := range prompt {
		switch message := message.(type) {
		case model.SystemMessage:
			messages = append(messages, systemInput{
				Role:    roleSystem,
				Content: message.Content,
			})

		case model.UserMessage:
			converted, err := convertUserMessage(message)
			if err != nil {
				return nil, nil, err
			}

			messages = append(messages, converted)

		case model.AssistantMessage:
			messages = append(messages, convertAssistantMessage(message)...)

		case model.ToolMessage:
			warnings = append(warnings, model.UnsupportedWarning("tool messages", ""))

		default:
			return nil, nil, fmt.Errorf("%w: unsupported role %q", model.ErrInvalidPrompt, message.Role())
		}
	}

	return messages, warnings, nil
}

func convertUserMessage(message model.UserMessage) (userInput, error) {
	parts := make([]inputPart, 0, len(message.Content))

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			parts = append(parts, inputText{
				Type: partInputText,
				Text: part.Text,
			})

		case model.FilePart:
			image, err := convertFilePart(part)
			if err != nil {
				return userInput{}, err
			}

			parts = append(parts, image)

		default:
			return userInput{}, fmt.Errorf("%w: user part %T", model.ErrUnsupported, part)
		}
	}

	return userInput{
		Role:    roleUser,
		Content: parts,
	}, nil
}

// convertFilePart sends images as URLs or data URLs. Other files are not
// supported.
func convertFilePart(part model.FilePart) (inputImage, error) {
	switch data := part.Data.(type) {
	case model.FileReference:
		return inputImage{}, fmt.Errorf("%w: file parts with provider references", model.ErrUnsupported)

	case model.FileText:
		return inputImage{}, fmt.Errorf("%w: text file parts", model.ErrUnsupported)

	case model.FileURL:
		if mediatype.TopLevel(part.MediaType) != "image" {
			return inputImage{}, unsupportedMediaType(part.MediaType)
		}

		return inputImage{
			Type:     partInputImage,
			ImageURL: data.URL.String(),
		}, nil

	case model.FileBytes:
		if mediatype.TopLevel(part.MediaType) != "image" {
			return inputImage{}, unsupportedMediaType(part.MediaType)
		}

		mediaType, err := mediatype.ResolveFile(part.MediaType, data)
		if err != nil {
			return inputImage{}, err
		}

		return inputImage{
			Type:     partInputImage,
			ImageURL: dataurl.Encode(mediaType, data.Data),
		}, nil

	default:
		return inputImage{}, fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, data)
	}
}

func unsupportedMediaType(mediaType string) error {
	return fmt.Errorf("%w: file part media type %s", model.ErrUnsupported, mediaType)
}

// convertAssistantMessage sends each text and reasoning part as its own
// assistant message.
func convertAssistantMessage(message model.AssistantMessage) []inputMessage {
	var messages []inputMessage

	for _, part := range message.Content {
		var text string

		switch part := part.(type) {
		case model.TextPart:
			text = part.Text
		case model.ReasoningPart:
			text = part.Text
		default:
			continue
		}

		messages = append(messages, assistantInput{
			Role: roleAssistant,
			Content: []outputText{
				{
					Type: partOutputText,
					Text: text,
				},
			},
		})
	}

	return messages
}
