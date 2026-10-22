package amazonbedrock

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// assistantTurn collects the content of an assistant block. Tool results
// that appear in assistant messages split the block: the assistant content
// so far becomes a message, and the results join the following user
// message.
type assistantTurn struct {
	converter   *promptConverter
	assistant   []contentBlock
	toolResults []contentBlock
}

func (c *promptConverter) assistantBlock(messages []model.Message, isLastBlock bool) error {
	turn := &assistantTurn{
		converter: c,
	}

	for index, message := range messages {
		assistant, ok := message.(model.AssistantMessage)
		if !ok {
			return fmt.Errorf("%w: assistant message of type %T", model.ErrInvalidPrompt, message)
		}

		isLastMessage := isLastBlock && index == len(messages)-1

		if err := turn.message(assistant, isLastMessage); err != nil {
			return err
		}
	}

	turn.flushToolResults()
	turn.flushAssistant()

	return nil
}

// message converts one assistant message. isLast reports whether it is the
// final message of the prompt, whose last text Bedrock requires without
// trailing whitespace.
func (t *assistantTurn) message(message model.AssistantMessage, isLast bool) error {
	hasReasoning := false
	for _, part := range message.Content {
		if _, ok := part.(model.ReasoningPart); ok {
			hasReasoning = true
		}
	}

	for index, part := range message.Content {
		_, isToolResult := part.(model.ToolResultPart)
		if !isToolResult {
			t.flushToolResults()
		}

		isLastPart := isLast && index == len(message.Content)-1

		options, err := t.part(part, hasReasoning, isLastPart)
		if err != nil {
			return err
		}

		if err := appendCachePoint(t.target(isToolResult), options); err != nil {
			return err
		}
	}

	_, endsWithToolResult := lastPart(message.Content).(model.ToolResultPart)

	return appendCachePoint(t.target(endsWithToolResult), message.ProviderOptions)
}

// part converts one assistant part and returns its provider options.
func (t *assistantTurn) part(part model.AssistantPart, hasReasoning bool, isLastPart bool) (model.ProviderOptions, error) {
	switch part := part.(type) {
	case model.TextPart:
		// Empty text is skipped unless reasoning is present.
		if strings.TrimSpace(part.Text) == "" && !hasReasoning {
			return part.ProviderOptions, nil
		}

		text := part.Text
		if isLastPart {
			text = strings.TrimSpace(text)
		}

		t.assistant = append(t.assistant, textBlock(text))

		return part.ProviderOptions, nil

	case model.ReasoningPart:
		return part.ProviderOptions, t.reasoning(part)

	case model.ToolCallPart:
		input, err := toolInputOf(part.Input)
		if err != nil {
			return nil, err
		}

		t.assistant = append(t.assistant, contentBlock{
			ToolUse: &toolUseBlock{
				ToolUseID: normalizeToolCallID(part.ToolCallID, t.converter.mistral),
				Name:      sanitizeToolName(part.ToolName),
				Input:     input,
			},
		})

		return part.ProviderOptions, nil

	case model.ToolResultPart:
		t.flushAssistant()

		result, err := t.converter.toolResult(part)
		if err != nil {
			return nil, err
		}

		t.toolResults = append(t.toolResults, result)

		return part.ProviderOptions, nil

	case model.FilePart:
		return part.ProviderOptions, nil

	case model.ReasoningFilePart:
		return part.ProviderOptions, nil

	case model.CustomPart:
		return part.ProviderOptions, nil

	default:
		// Tool approval requests have no Converse form.
		return nil, nil
	}
}

// reasoning replays signed or redacted reasoning. Unsigned reasoning is not
// replayed: models such as OpenAI gpt-oss return it without a signature, and
// sending it back can leak raw reasoning into the visible response.
func (t *assistantTurn) reasoning(part model.ReasoningPart) error {
	metadata, _, err := namespacedValue[ReasoningMetadata](part.ProviderOptions)
	if err != nil {
		return err
	}

	switch {
	case metadata.Signature != "":
		// The signature covers the exact original text, so it is not
		// trimmed.
		t.assistant = append(t.assistant, contentBlock{
			ReasoningContent: &reasoningContentBlock{
				ReasoningText: &reasoningText{
					Text:      part.Text,
					Signature: metadata.Signature,
				},
			},
		})

	case metadata.RedactedContent != "":
		t.assistant = append(t.assistant, contentBlock{
			ReasoningContent: &reasoningContentBlock{
				RedactedContent: &metadata.RedactedContent,
			},
		})

	case metadata.RedactedData != "":
		t.assistant = append(t.assistant, contentBlock{
			ReasoningContent: &reasoningContentBlock{
				RedactedReasoning: &redactedReasoning{
					Data: metadata.RedactedData,
				},
			},
		})
	}

	return nil
}

func (t *assistantTurn) target(toolResult bool) *[]contentBlock {
	if toolResult {
		return &t.toolResults
	}

	return &t.assistant
}

// flushAssistant ends the current assistant message. Content of only cache
// points is dropped.
func (t *assistantTurn) flushAssistant() {
	hasContent := false
	for _, block := range t.assistant {
		if block.CachePoint == nil {
			hasContent = true
		}
	}

	if hasContent {
		t.converter.messages = append(t.converter.messages, converseMessage{
			Role:    roleAssistant,
			Content: t.assistant,
		})
	}

	t.assistant = nil
}

func (t *assistantTurn) flushToolResults() {
	if len(t.toolResults) == 0 {
		return
	}

	t.converter.appendToUserMessage(t.toolResults)
	t.toolResults = nil
}

func lastPart(parts []model.AssistantPart) model.AssistantPart {
	if len(parts) == 0 {
		return nil
	}

	return parts[len(parts)-1]
}

// toolResultOutput converts a tool result into Converse tool result
// content.
func (c *promptConverter) toolResultOutput(output model.ToolResultOutput) ([]contentBlock, error) {
	content, ok := output.(model.ContentOutput)
	if !ok {
		text, err := toolresult.Text(output)
		if err != nil {
			return nil, err
		}

		return []contentBlock{
			textBlock(text),
		}, nil
	}

	converted := make([]contentBlock, 0, len(content.Value))
	for _, item := range content.Value {
		block, err := c.toolResultItem(item)
		if err != nil {
			return nil, err
		}

		converted = append(converted, block)
	}

	return converted, nil
}

func (c *promptConverter) toolResultItem(item model.ToolResultContent) (contentBlock, error) {
	switch item := item.(type) {
	case model.ToolResultText:
		return textBlock(item.Text), nil

	case model.ToolResultFile:
		return c.toolResultFile(item)

	default:
		return contentBlock{}, fmt.Errorf("%w: unsupported tool content part type: %T", model.ErrUnsupported, item)
	}
}

func (c *promptConverter) toolResultFile(file model.ToolResultFile) (contentBlock, error) {
	functionality := fmt.Sprintf("tool result file data of type %q", fileDataType(file.Data))

	_, inline := file.Data.(model.FileBytes)
	url, isURL := file.Data.(model.FileURL)

	if !inline && (!isURL || !isS3URL(url)) {
		return contentBlock{}, fmt.Errorf("%w: %s", model.ErrUnsupported, functionality)
	}

	mediaType, err := mediatype.ResolveFile(file.MediaType, file.Data)
	if err != nil {
		return contentBlock{}, err
	}

	switch mediatype.TopLevel(mediaType) {
	case "image":
		image, err := mediaBlockOf(mediaType, file.Data, imageFormat)

		return contentBlock{
			Image: image,
		}, err

	case "video":
		video, err := mediaBlockOf(mediaType, file.Data, videoFormat)

		return contentBlock{
			Video: video,
		}, err
	}

	if !inline {
		return contentBlock{}, fmt.Errorf("%w: %s", model.ErrUnsupported, functionality)
	}

	document, err := c.document(mediaType, file.Filename, file.Data.(model.FileBytes).Data, file.ProviderOptions)
	if err != nil {
		return contentBlock{}, err
	}

	return contentBlock{
		Document: document,
	}, nil
}

// fileDataType names file data the way upstream reports it.
func fileDataType(data model.FileData) string {
	switch data.(type) {
	case model.FileBytes:
		return "data"
	case model.FileURL:
		return "url"
	case model.FileReference:
		return "reference"
	case model.FileText:
		return "text"
	default:
		return fmt.Sprintf("%T", data)
	}
}
