package openai

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// chatPrompt converts a prompt into Chat Completions messages.
func chatPrompt(prompt []model.Message, systemMode ChatSystemMessageMode) ([]chatMessage, []model.Warning, error) {
	messages := make([]chatMessage, 0, len(prompt))
	var warnings []model.Warning

	for _, message := range prompt {
		switch message := message.(type) {
		case model.SystemMessage:
			if systemMode == ChatSystemMessageRemove {
				warnings = append(warnings, model.Warning{
					Type:    model.WarningOther,
					Message: "system messages are removed for this model",
				})

				continue
			}

			converted, err := chatSystemMessage(message, systemMode)
			if err != nil {
				return nil, nil, err
			}

			messages = append(messages, converted)

		case model.UserMessage:
			converted, err := chatUserMessage(message)
			if err != nil {
				return nil, nil, err
			}

			messages = append(messages, converted)

		case model.AssistantMessage:
			converted, err := chatAssistantMessage(message)
			if err != nil {
				return nil, nil, err
			}

			messages = append(messages, converted)

		case model.ToolMessage:
			converted, err := chatToolMessages(message)
			if err != nil {
				return nil, nil, err
			}

			messages = append(messages, converted...)

		default:
			return nil, nil, fmt.Errorf("%w: unsupported message role %q", model.ErrInvalidPrompt, message.Role())
		}
	}

	return messages, warnings, nil
}

func chatSystemMessage(message model.SystemMessage, systemMode ChatSystemMessageMode) (chatMessage, error) {
	role := chatRoleSystem
	if systemMode == ChatSystemMessageDeveloper {
		role = chatRoleDeveloper
	}

	breakpoint, err := chatCacheBreakpoint(message.ProviderOptions)
	if err != nil {
		return chatMessage{}, err
	}

	return chatMessage{
		Role:    role,
		Content: chatTextOrBreakpointContent(message.Content, breakpoint),
	}, nil
}

func chatUserMessage(message model.UserMessage) (chatMessage, error) {
	text, isSingleText, err := chatSingleUserText(message)
	if err != nil {
		return chatMessage{}, err
	}

	if isSingleText {
		return chatMessage{
			Role:    chatRoleUser,
			Content: chatTextContent(text),
		}, nil
	}

	parts := make([]chatContentPart, 0, len(message.Content))
	for index, part := range message.Content {
		converted, err := chatUserPart(part, index)
		if err != nil {
			return chatMessage{}, err
		}

		parts = append(parts, converted)
	}

	return chatMessage{
		Role:    chatRoleUser,
		Content: chatPartsContent(parts),
	}, nil
}

// chatSingleUserText returns the text of a message consisting of one text
// part without a cache breakpoint, which is sent as plain string content.
func chatSingleUserText(message model.UserMessage) (string, bool, error) {
	if len(message.Content) != 1 {
		return "", false, nil
	}

	textPart, ok := message.Content[0].(model.TextPart)
	if !ok {
		return "", false, nil
	}

	breakpoint, err := chatCacheBreakpoint(textPart.ProviderOptions)
	if err != nil || breakpoint != nil {
		return "", false, err
	}

	return textPart.Text, true, nil
}

func chatUserPart(part model.UserPart, index int) (chatContentPart, error) {
	switch part := part.(type) {
	case model.TextPart:
		breakpoint, err := chatCacheBreakpoint(part.ProviderOptions)
		if err != nil {
			return chatContentPart{}, err
		}

		return chatContentPart{
			Type:                  chatPartText,
			Text:                  &part.Text,
			PromptCacheBreakpoint: breakpoint,
		}, nil

	case model.FilePart:
		return chatFilePart(part, index)

	default:
		return chatContentPart{}, fmt.Errorf("%w: unsupported user part %T", model.ErrInvalidPrompt, part)
	}
}

func chatFilePart(part model.FilePart, index int) (chatContentPart, error) {
	options, err := readChatPartOptions(part.ProviderOptions)
	if err != nil {
		return chatContentPart{}, err
	}

	converted, err := chatFilePayload(part, index, options)
	if err != nil {
		return chatContentPart{}, err
	}

	converted.PromptCacheBreakpoint = options.PromptCacheBreakpoint

	return converted, nil
}

func chatFilePayload(part model.FilePart, index int, options ChatPartOptions) (chatContentPart, error) {
	switch data := part.Data.(type) {
	case model.FileReference:
		return chatFileReferencePart(data)

	case model.FileText:
		return chatContentPart{}, fmt.Errorf("%w: text file parts", model.ErrUnsupported)

	case model.FileURL:
		return chatURLFilePart(part, data, options)

	case model.FileBytes:
		return chatInlineFilePart(part, data, index, options)

	default:
		return chatContentPart{}, fmt.Errorf("%w: unsupported file data %T", model.ErrInvalidPrompt, data)
	}
}

func chatFileReferencePart(reference model.FileReference) (chatContentPart, error) {
	fileID, ok := reference.IDs[Namespace]
	if !ok {
		return chatContentPart{}, fmt.Errorf("%w: file reference has no openai file ID", model.ErrInvalidPrompt)
	}

	return chatContentPart{
		Type: chatPartFile,
		File: &chatFile{
			FileID: fileID,
		},
	}, nil
}

func chatURLFilePart(part model.FilePart, data model.FileURL, options ChatPartOptions) (chatContentPart, error) {
	switch mediatype.TopLevel(part.MediaType) {
	case "image":
		return chatImagePart(data.URL.String(), options), nil

	case "audio":
		return chatContentPart{}, fmt.Errorf("%w: audio file parts with URLs", model.ErrUnsupported)
	}

	mediaType, err := mediatype.ResolveFile(part.MediaType, data)
	if err != nil {
		return chatContentPart{}, err
	}

	if mediaType != "application/pdf" {
		return chatContentPart{}, fmt.Errorf("%w: file part media type %s", model.ErrUnsupported, mediaType)
	}

	return chatContentPart{}, fmt.Errorf("%w: PDF file parts with URLs", model.ErrUnsupported)
}

func chatInlineFilePart(part model.FilePart, data model.FileBytes, index int, options ChatPartOptions) (chatContentPart, error) {
	mediaType, err := mediatype.ResolveFile(part.MediaType, data)
	if err != nil {
		return chatContentPart{}, err
	}

	encoded := base64.StdEncoding.EncodeToString(data.Data)

	switch mediatype.TopLevel(part.MediaType) {
	case "image":
		return chatImagePart("data:"+mediaType+";base64,"+encoded, options), nil

	case "audio":
		return chatAudioPart(mediaType, encoded)
	}

	if mediaType != "application/pdf" {
		return chatContentPart{}, fmt.Errorf("%w: file part media type %s", model.ErrUnsupported, mediaType)
	}

	filename := part.Filename
	if filename == "" {
		filename = fmt.Sprintf("part-%d.pdf", index)
	}

	return chatContentPart{
		Type: chatPartFile,
		File: &chatFile{
			Filename: filename,
			FileData: "data:application/pdf;base64," + encoded,
		},
	}, nil
}

func chatImagePart(url string, options ChatPartOptions) chatContentPart {
	return chatContentPart{
		Type: chatPartImageURL,
		ImageURL: &chatImageURL{
			URL:    url,
			Detail: options.ImageDetail,
		},
	}
}

func chatAudioPart(mediaType, encoded string) (chatContentPart, error) {
	var format chatAudioFormat

	switch mediaType {
	case "audio/wav":
		format = chatAudioWAV
	case "audio/mp3", "audio/mpeg":
		format = chatAudioMP3
	default:
		return chatContentPart{}, fmt.Errorf("%w: audio content parts with media type %s", model.ErrUnsupported, mediaType)
	}

	return chatContentPart{
		Type: chatPartInputAudio,
		InputAudio: &chatInputAudio{
			Data:   encoded,
			Format: format,
		},
	}, nil
}

// chatAssistantMessage joins text parts and collects tool calls. Other
// assistant parts, such as reasoning, have no Chat Completions form.
func chatAssistantMessage(message model.AssistantMessage) (chatMessage, error) {
	var text strings.Builder
	var textParts []chatContentPart
	var toolCalls []chatToolCall
	hasBreakpoint := false

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			breakpoint, err := chatCacheBreakpoint(part.ProviderOptions)
			if err != nil {
				return chatMessage{}, err
			}

			text.WriteString(part.Text)
			textParts = append(textParts, chatContentPart{
				Type:                  chatPartText,
				Text:                  &part.Text,
				PromptCacheBreakpoint: breakpoint,
			})
			hasBreakpoint = hasBreakpoint || breakpoint != nil

		case model.ToolCallPart:
			toolCalls = append(toolCalls, chatToolCall{
				ID:   part.ToolCallID,
				Type: "function",
				Function: chatFunctionCall{
					Name:      part.ToolName,
					Arguments: chatToolCallArguments(part.Input),
				},
			})
		}
	}

	return chatMessage{
		Role:      chatRoleAssistant,
		Content:   chatAssistantContent(text.String(), textParts, hasBreakpoint, len(toolCalls) > 0),
		ToolCalls: toolCalls,
	}, nil
}

// chatAssistantContent sends text parts when one carries a cache breakpoint;
// otherwise the joined text, or null when there is no text beside tool calls.
func chatAssistantContent(text string, textParts []chatContentPart, hasBreakpoint, hasToolCalls bool) chatMessageContent {
	if hasBreakpoint {
		return chatPartsContent(textParts)
	}

	if hasToolCalls && text == "" {
		return chatMessageContent{}
	}

	return chatTextContent(text)
}

// chatToolCallArguments encodes tool input. Inputs that are not JSON objects
// are sent as an empty object.
func chatToolCallArguments(input json.RawMessage) string {
	if !chatIsJSONObject(input) {
		return "{}"
	}

	var compacted bytes.Buffer
	if err := json.Compact(&compacted, input); err != nil {
		return "{}"
	}

	return compacted.String()
}

// chatToolMessages converts each tool result into its own tool message.
// Approval responses have no Chat Completions form and are skipped.
func chatToolMessages(message model.ToolMessage) ([]chatMessage, error) {
	var messages []chatMessage

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			continue
		}

		converted, err := chatToolResultMessage(result)
		if err != nil {
			return nil, err
		}

		messages = append(messages, converted)
	}

	return messages, nil
}

func chatToolResultMessage(result model.ToolResultPart) (chatMessage, error) {
	content, err := chatToolResultContent(result.Output)
	if err != nil {
		return chatMessage{}, err
	}

	breakpoint, err := chatToolResultBreakpoint(result)
	if err != nil {
		return chatMessage{}, err
	}

	return chatMessage{
		Role:       chatRoleTool,
		ToolCallID: result.ToolCallID,
		Content:    chatTextOrBreakpointContent(content, breakpoint),
	}, nil
}

// chatToolResultContent encodes a tool result output as text.
func chatToolResultContent(output model.ToolResultOutput) (string, error) {
	switch output := output.(type) {
	case model.TextOutput:
		return output.Value, nil

	case model.ErrorTextOutput:
		return chatToolErrorJSON(jsonobject.EncodeString(output.Value))

	case model.ErrorJSONOutput:
		return chatToolErrorJSON(output.Value)

	case model.ExecutionDeniedOutput:
		if output.Reason == "" {
			return toolresult.DeniedText, nil
		}

		return output.Reason, nil

	case model.JSONOutput:
		return toolresult.CompactJSON(output.Value)

	case model.ContentOutput:
		return toolresult.ContentJSON(output.Value)

	default:
		return "", fmt.Errorf("%w: unsupported tool result output %T", model.ErrInvalidPrompt, output)
	}
}

// chatToolError is the JSON form of a failed tool result.
type chatToolError struct {
	Error json.RawMessage `json:"error"`
}

func chatToolErrorJSON(value json.RawMessage) (string, error) {
	encoded, err := json.Marshal(chatToolError{
		Error: value,
	})
	if err != nil {
		return "", fmt.Errorf("%w: encode tool error result: %w", model.ErrInvalidPrompt, err)
	}

	return string(encoded), nil
}

// chatToolResultBreakpoint returns the first cache breakpoint among the
// output's content items or the output itself, then the result part.
func chatToolResultBreakpoint(result model.ToolResultPart) (*ChatPromptCacheBreakpoint, error) {
	for _, options := range chatToolOutputOptions(result.Output) {
		breakpoint, err := chatCacheBreakpoint(options)
		if err != nil || breakpoint != nil {
			return breakpoint, err
		}
	}

	return chatCacheBreakpoint(result.ProviderOptions)
}

// chatToolOutputOptions lists the provider options attached to an output.
func chatToolOutputOptions(output model.ToolResultOutput) []model.ProviderOptions {
	switch output := output.(type) {
	case model.TextOutput:
		return []model.ProviderOptions{
			output.ProviderOptions,
		}

	case model.JSONOutput:
		return []model.ProviderOptions{
			output.ProviderOptions,
		}

	case model.ErrorTextOutput:
		return []model.ProviderOptions{
			output.ProviderOptions,
		}

	case model.ErrorJSONOutput:
		return []model.ProviderOptions{
			output.ProviderOptions,
		}

	case model.ExecutionDeniedOutput:
		return []model.ProviderOptions{
			output.ProviderOptions,
		}

	case model.ContentOutput:
		return chatToolContentOptions(output.Value)

	default:
		return nil
	}
}

func chatToolContentOptions(items []model.ToolResultContent) []model.ProviderOptions {
	options := make([]model.ProviderOptions, 0, len(items))

	for _, item := range items {
		switch item := item.(type) {
		case model.ToolResultText:
			options = append(options, item.ProviderOptions)
		case model.ToolResultFile:
			options = append(options, item.ProviderOptions)
		case model.ToolResultCustom:
			options = append(options, item.ProviderOptions)
		}
	}

	return options
}

// chatCacheBreakpoint reads an explicit prompt cache breakpoint from options.
func chatCacheBreakpoint(options model.ProviderOptions) (*ChatPromptCacheBreakpoint, error) {
	values, err := readChatPartOptions(options)
	if err != nil {
		return nil, err
	}

	return values.PromptCacheBreakpoint, nil
}

// chatTextOrBreakpointContent sends text as a string, or as a single text
// part carrying the breakpoint.
func chatTextOrBreakpointContent(text string, breakpoint *ChatPromptCacheBreakpoint) chatMessageContent {
	if breakpoint == nil {
		return chatTextContent(text)
	}

	return chatPartsContent([]chatContentPart{
		{
			Type:                  chatPartText,
			Text:                  &text,
			PromptCacheBreakpoint: breakpoint,
		},
	})
}
