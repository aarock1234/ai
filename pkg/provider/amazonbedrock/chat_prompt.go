package amazonbedrock

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

const fileURLFunctionality = "File URL data"

// convertedPrompt is a prompt in Converse form.
type convertedPrompt struct {
	system   []contentBlock
	messages []converseMessage
}

// promptConverter converts one prompt. Documents without a usable filename
// are numbered in order.
type promptConverter struct {
	mistral       bool
	documentCount int

	system   []contentBlock
	messages []converseMessage
}

// convertPrompt converts a prompt into Converse system blocks and messages.
// Consecutive user and tool messages merge into one user message, and
// consecutive assistant messages into one assistant message. Tool call IDs
// are normalized for Mistral models.
func convertPrompt(prompt []model.Message, mistral bool) (convertedPrompt, error) {
	converter := &promptConverter{
		mistral: mistral,
		system:  []contentBlock{},
	}

	blocks := groupIntoBlocks(prompt)

	for index, block := range blocks {
		isLastBlock := index == len(blocks)-1

		var err error

		switch block.role {
		case model.RoleSystem:
			err = converter.systemBlock(block.messages)
		case model.RoleAssistant:
			err = converter.assistantBlock(block.messages, isLastBlock)
		default:
			err = converter.userBlock(block.messages)
		}

		if err != nil {
			return convertedPrompt{}, err
		}
	}

	if converter.messages == nil {
		converter.messages = []converseMessage{}
	}

	return convertedPrompt{
		system:   converter.system,
		messages: converter.messages,
	}, nil
}

// messageBlock is a run of messages that convert into one Converse turn.
// Tool messages join user blocks.
type messageBlock struct {
	role     model.Role
	messages []model.Message
}

func groupIntoBlocks(prompt []model.Message) []messageBlock {
	var blocks []messageBlock

	for _, message := range prompt {
		role := message.Role()
		if role == model.RoleTool {
			role = model.RoleUser
		}

		if len(blocks) == 0 || blocks[len(blocks)-1].role != role {
			blocks = append(blocks, messageBlock{
				role: role,
			})
		}

		last := &blocks[len(blocks)-1]
		last.messages = append(last.messages, message)
	}

	return blocks
}

func (c *promptConverter) systemBlock(messages []model.Message) error {
	if len(c.messages) > 0 {
		return fmt.Errorf("%w: multiple system messages that are separated by user/assistant messages", model.ErrUnsupported)
	}

	for _, message := range messages {
		system, ok := message.(model.SystemMessage)
		if !ok {
			return fmt.Errorf("%w: system message of type %T", model.ErrInvalidPrompt, message)
		}

		c.system = append(c.system, textBlock(system.Content))

		if err := appendCachePoint(&c.system, system.ProviderOptions); err != nil {
			return err
		}
	}

	return nil
}

// userBlock combines the user and tool messages of a block into one user
// message.
func (c *promptConverter) userBlock(messages []model.Message) error {
	content := []contentBlock{}

	for _, message := range messages {
		var (
			options model.ProviderOptions
			err     error
		)

		switch message := message.(type) {
		case model.UserMessage:
			options = message.ProviderOptions
			err = c.userParts(&content, message.Content)

		case model.ToolMessage:
			options = message.ProviderOptions
			err = c.toolParts(&content, message.Content)

		default:
			err = fmt.Errorf("%w: unsupported role: %s", model.ErrInvalidPrompt, message.Role())
		}

		if err != nil {
			return err
		}

		if err := appendCachePoint(&content, options); err != nil {
			return err
		}
	}

	c.appendToUserMessage(content)

	return nil
}

func (c *promptConverter) userParts(content *[]contentBlock, parts []model.UserPart) error {
	for _, part := range parts {
		var (
			options model.ProviderOptions
			err     error
		)

		switch part := part.(type) {
		case model.TextPart:
			options = part.ProviderOptions
			err = appendUserText(content, part)

		case model.FilePart:
			options = part.ProviderOptions
			err = c.appendUserFile(content, part)

		default:
			err = fmt.Errorf("%w: user content part %T", model.ErrUnsupported, part)
		}

		if err != nil {
			return err
		}

		if err := appendCachePoint(content, options); err != nil {
			return err
		}
	}

	return nil
}

func appendUserText(content *[]contentBlock, part model.TextPart) error {
	options, err := readTextPartOptions(part.ProviderOptions)
	if err != nil {
		return err
	}

	if options.GuardContent == nil || !*options.GuardContent {
		*content = append(*content, textBlock(part.Text))

		return nil
	}

	*content = append(*content, contentBlock{
		GuardContent: &guardContentBlock{
			Text: &guardText{
				Text:       part.Text,
				Qualifiers: options.GuardContentQualifiers,
			},
		},
	})

	return nil
}

func (c *promptConverter) appendUserFile(content *[]contentBlock, part model.FilePart) error {
	switch data := part.Data.(type) {
	case model.FileReference:
		return fmt.Errorf("%w: file parts with provider references", model.ErrUnsupported)

	case model.FileURL:
		if !isS3URL(data) {
			return fmt.Errorf("%w: %s", model.ErrUnsupported, fileURLFunctionality)
		}

		return c.appendUserMedia(content, part, false)

	case model.FileText:
		return c.appendTextDocument(content, part, data)

	case model.FileBytes:
		return c.appendUserMedia(content, part, true)

	default:
		return fmt.Errorf("%w: file data %T", model.ErrUnsupported, data)
	}
}

// appendUserMedia appends an image, a video or, for inline bytes, a
// document.
func (c *promptConverter) appendUserMedia(content *[]contentBlock, part model.FilePart, inline bool) error {
	mediaType, err := mediatype.ResolveFile(part.MediaType, part.Data)
	if err != nil {
		return err
	}

	switch mediatype.TopLevel(mediaType) {
	case "image":
		image, err := mediaBlockOf(mediaType, part.Data, imageFormat)
		if err != nil {
			return err
		}

		return appendUserImage(content, part.ProviderOptions, image, inline)

	case "video":
		video, err := mediaBlockOf(mediaType, part.Data, videoFormat)
		if err != nil {
			return err
		}

		*content = append(*content, contentBlock{
			Video: video,
		})

		return nil
	}

	if !inline {
		return fmt.Errorf("%w: %s", model.ErrUnsupported, fileURLFunctionality)
	}

	document, err := c.document(mediaType, part.Filename, part.Data.(model.FileBytes).Data, part.ProviderOptions)
	if err != nil {
		return err
	}

	*content = append(*content, contentBlock{
		Document: document,
	})

	return nil
}

// appendUserImage appends an image, as guardrail content when the part's
// options request it. Only inline images can be guarded.
func appendUserImage(content *[]contentBlock, options model.ProviderOptions, image *mediaBlock, inline bool) error {
	if !inline {
		*content = append(*content, contentBlock{
			Image: image,
		})

		return nil
	}

	imageOptions, err := readImagePartOptions(options)
	if err != nil {
		return err
	}

	if imageOptions.GuardContent != nil && *imageOptions.GuardContent {
		*content = append(*content, contentBlock{
			GuardContent: &guardContentBlock{
				Image: image,
			},
		})

		return nil
	}

	*content = append(*content, contentBlock{
		Image: image,
	})

	return nil
}

func (c *promptConverter) appendTextDocument(content *[]contentBlock, part model.FilePart, data model.FileText) error {
	mediaType := part.MediaType
	if !mediatype.IsFull(mediaType) {
		mediaType = "text/plain"
	}

	document, err := c.document(mediaType, part.Filename, []byte(data.Text), part.ProviderOptions)
	if err != nil {
		return err
	}

	*content = append(*content, contentBlock{
		Document: document,
	})

	return nil
}

// document returns a document block, with citations when the options
// enable them.
func (c *promptConverter) document(mediaType string, filename string, data []byte, options model.ProviderOptions) (*documentBlock, error) {
	format, err := documentFormat(mediaType)
	if err != nil {
		return nil, err
	}

	citations, err := citationsEnabled(options)
	if err != nil {
		return nil, err
	}

	document := &documentBlock{
		Format: format,
		Name:   c.documentName(filename),
		Source: documentSource{
			Bytes: base64.StdEncoding.EncodeToString(data),
		},
	}

	if citations {
		document.Citations = &DocumentCitations{
			Enabled: true,
		}
	}

	return document, nil
}

// documentName returns the sanitized filename, or the next generated name.
func (c *promptConverter) documentName(filename string) string {
	if name := sanitizeDocumentName(filename); name != "" {
		return name
	}

	c.documentCount++

	return "document-" + strconv.Itoa(c.documentCount)
}

func mediaBlockOf(mediaType string, data model.FileData, format func(string) (string, error)) (*mediaBlock, error) {
	converted, err := format(mediaType)
	if err != nil {
		return nil, err
	}

	source, err := mediaSourceOf(data, fileURLFunctionality)
	if err != nil {
		return nil, err
	}

	return &mediaBlock{
		Format: converted,
		Source: source,
	}, nil
}

func (c *promptConverter) toolParts(content *[]contentBlock, parts []model.ToolPart) error {
	for _, part := range parts {
		switch part := part.(type) {
		case model.ToolApprovalResponsePart:
			continue

		case model.ToolResultPart:
			result, err := c.toolResult(part)
			if err != nil {
				return err
			}

			*content = append(*content, result)

			if err := appendCachePoint(content, part.ProviderOptions); err != nil {
				return err
			}

		default:
			return fmt.Errorf("%w: tool content part %T", model.ErrUnsupported, part)
		}
	}

	return nil
}

func (c *promptConverter) toolResult(part model.ToolResultPart) (contentBlock, error) {
	output, err := c.toolResultOutput(part.Output)
	if err != nil {
		return contentBlock{}, err
	}

	return contentBlock{
		ToolResult: &toolResultBlock{
			ToolUseID: normalizeToolCallID(part.ToolCallID, c.mistral),
			Content:   output,
		},
	}, nil
}

// appendToUserMessage adds user content. A user message that holds tool
// results absorbs the content, so that roles keep alternating after a tool
// result that ended an assistant block.
func (c *promptConverter) appendToUserMessage(content []contentBlock) {
	if len(c.messages) > 0 {
		last := &c.messages[len(c.messages)-1]

		if last.Role == roleUser && hasToolResult(last.Content) {
			last.Content = append(last.Content, content...)

			return
		}
	}

	c.messages = append(c.messages, converseMessage{
		Role:    roleUser,
		Content: content,
	})
}

func hasToolResult(content []contentBlock) bool {
	for _, block := range content {
		if block.ToolResult != nil {
			return true
		}
	}

	return false
}

// appendCachePoint appends the cache point of options, if any.
func appendCachePoint(content *[]contentBlock, options model.ProviderOptions) error {
	point, err := cachePointOf(options)
	if err != nil || point == nil {
		return err
	}

	*content = append(*content, cachePointBlock(point))

	return nil
}

// toolInputOf returns tool call input as an object, wrapping other values,
// because Converse requires an object.
func toolInputOf(input json.RawMessage) (json.RawMessage, error) {
	if len(strings.TrimSpace(string(input))) == 0 {
		return json.RawMessage(`{}`), nil
	}

	if jsonobject.IsObject(input) {
		return input, nil
	}

	if !json.Valid(input) {
		return nil, fmt.Errorf("%w: tool call input is not valid JSON", model.ErrInvalidPrompt)
	}

	wrapped := jsonobject.Object{
		{
			Key:   "rawInvalidInput",
			Value: input,
		},
	}

	return wrapped.Encode()
}
