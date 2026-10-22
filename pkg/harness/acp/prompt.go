package acp

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/model"
)

// textBlock is an ACP text content block of a prompt.
type textBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// newTextBlock returns a text block.
func newTextBlock(text string) textBlock {
	return textBlock{
		Type: "text",
		Text: text,
	}
}

// promptTextBlocks converts a turn prompt into ACP text blocks. Only text
// parts are portable across ACP implementations.
func promptTextBlocks(prompt model.UserMessage, harnessID string) ([]textBlock, error) {
	blocks := make([]textBlock, 0, len(prompt.Content))

	for _, part := range prompt.Content {
		switch typed := part.(type) {
		case model.TextPart:
			blocks = append(blocks, newTextBlock(typed.Text))
		case model.FilePart:
			return nil, unsupportedPromptContent(harnessID, filePartCategory(typed.MediaType))
		default:
			return nil, unsupportedPromptContent(harnessID, fmt.Sprintf("content parts of type %T", part))
		}
	}

	return blocks, nil
}

// filePartCategory describes an unsupported file part.
func filePartCategory(mediaType string) string {
	topLevel, _, _ := strings.Cut(strings.ToLower(mediaType), "/")

	switch topLevel {
	case "image":
		return "image content"
	case "audio":
		return "audio content"
	default:
		return fmt.Sprintf("embedded resource content with media type %q", mediaType)
	}
}

// unsupportedPromptContent reports prompt content ACP prompts cannot carry.
func unsupportedPromptContent(harnessID string, category string) error {
	return &harness.CapabilityUnsupportedError{
		Message: fmt.Sprintf(
			"The %s ACP harness supports only portable text prompts and does not support %s. Pass a string or a user message whose content contains only text parts.",
			harnessID,
			category,
		),
		HarnessID: harnessID,
	}
}

// prependInstructionGuidance prepends the instructions to a prompt as
// harness guidance, for implementations without a native instruction
// mapping.
func prependInstructionGuidance(prompt []textBlock, instructions string) []textBlock {
	if instructions == "" {
		return prompt
	}

	guidance := strings.Join([]string{
		"<session-guidance>",
		"This block is operating guidance from the harness, not user-authored content.",
		"<instructions>",
		instructions,
		"</instructions>",
		"</session-guidance>",
	}, "\n")

	return append([]textBlock{
		newTextBlock(guidance),
	}, prompt...)
}
