package harness

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// PromptContent is a turn prompt as runtimes accept it: plain text when the
// prompt has only text parts, and text and inline image parts otherwise. It
// encodes as a JSON string or array.
type PromptContent struct {
	// Text is the joined text of a text-only prompt.
	Text string

	// Parts are the parts of a prompt with images. Nil means text only.
	Parts []PromptContentPart
}

// PromptContentPartType distinguishes the parts of PromptContent.
type PromptContentPartType string

// Prompt content part types.
const (
	PromptContentText  PromptContentPartType = "text"
	PromptContentImage PromptContentPartType = "image"
)

// PromptContentPart is one part of PromptContent.
type PromptContentPart struct {
	Type PromptContentPartType `json:"type"`
	Text string                `json:"text,omitempty"`

	// Data is the base64 data of an image.
	Data      string `json:"data,omitempty"`
	MediaType string `json:"mediaType,omitempty"`
}

// MarshalJSON encodes the text as a string, or the parts as an array.
func (c PromptContent) MarshalJSON() ([]byte, error) {
	if c.Parts == nil {
		return json.Marshal(c.Text)
	}

	return json.Marshal(c.Parts)
}

// ExtractPromptContent converts a turn prompt into text and inline images.
// A text-only prompt joins its parts with blank lines. Images may be inline
// bytes or base64 data URLs; remote URLs, provider references, text data
// and non-image files are reported with *CapabilityUnsupportedError.
func ExtractPromptContent(prompt model.UserMessage) (PromptContent, error) {
	if texts, ok := textOnlyParts(prompt.Content); ok {
		return PromptContent{
			Text: strings.Join(texts, "\n\n"),
		}, nil
	}

	parts := make([]PromptContentPart, 0, len(prompt.Content))

	for _, part := range prompt.Content {
		switch typed := part.(type) {
		case model.TextPart:
			parts = append(parts, PromptContentPart{
				Type: PromptContentText,
				Text: typed.Text,
			})
		case model.FilePart:
			image, err := promptImage(typed)
			if err != nil {
				return PromptContent{}, err
			}

			parts = append(parts, image)
		default:
			return PromptContent{}, unsupportedPromptContent("content parts other than text and file parts")
		}
	}

	return PromptContent{
		Parts: parts,
	}, nil
}

// PromptText returns the text of a prompt, ignoring other parts.
func PromptText(prompt model.UserMessage) string {
	var text strings.Builder

	for _, part := range prompt.Content {
		if typed, ok := part.(model.TextPart); ok {
			text.WriteString(typed.Text)
		}
	}

	return text.String()
}

// TextPrompt returns a prompt with a single text part.
func TextPrompt(text string) model.UserMessage {
	return model.UserMessage{
		Content: []model.UserPart{
			model.TextPart{
				Text: text,
			},
		},
	}
}

// textOnlyParts returns the texts of a prompt when every part is text.
func textOnlyParts(content []model.UserPart) ([]string, bool) {
	texts := make([]string, 0, len(content))

	for _, part := range content {
		text, ok := part.(model.TextPart)
		if !ok {
			return nil, false
		}

		texts = append(texts, text.Text)
	}

	return texts, true
}

// promptImage converts an image file part into an inline image.
func promptImage(part model.FilePart) (PromptContentPart, error) {
	if mediatype.TopLevel(part.MediaType) != "image" {
		return PromptContentPart{}, unsupportedPromptContent(fmt.Sprintf("files of media type '%s'", part.MediaType))
	}

	data, dataMediaType, err := inlineImageData(part.Data)
	if err != nil {
		return PromptContentPart{}, err
	}

	mediaType := part.MediaType
	if !mediatype.IsFull(mediaType) {
		mediaType = dataMediaType
	}

	if !mediatype.IsFull(mediaType) {
		detected, ok := mediatype.Detect(data, "image")
		if !ok {
			return PromptContentPart{}, unsupportedPromptContent("images whose media type cannot be detected")
		}

		mediaType = detected
	}

	return PromptContentPart{
		Type:      PromptContentImage,
		Data:      base64.StdEncoding.EncodeToString(data),
		MediaType: mediaType,
	}, nil
}

// inlineImageData returns the bytes of inline image data and the media type
// of a data URL.
func inlineImageData(data model.FileData) ([]byte, string, error) {
	switch typed := data.(type) {
	case model.FileBytes:
		return typed.Data, "", nil
	case model.FileURL:
		return dataURLImage(typed.URL.String())
	case model.FileReference:
		return nil, "", unsupportedPromptContent("provider references")
	case model.FileText:
		return nil, "", unsupportedPromptContent("text data for images")
	default:
		return nil, "", unsupportedPromptContent(fmt.Sprintf("file data of type %T", data))
	}
}

// dataURLImage decodes a base64 data URL.
func dataURLImage(rawURL string) ([]byte, string, error) {
	rest, isDataURL := strings.CutPrefix(rawURL, "data:")
	if !isDataURL {
		return nil, "", unsupportedPromptContent("remote image URLs")
	}

	header, encoded, found := strings.Cut(rest, ",")
	if !found || !strings.HasSuffix(header, ";base64") {
		return nil, "", unsupportedPromptContent("data URLs that are not base64 encoded")
	}

	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, "", unsupportedPromptContent("data URLs with invalid base64 data")
	}

	mediaType, _, _ := strings.Cut(header, ";")

	return decoded, mediaType, nil
}

// unsupportedPromptContent reports prompt content harnesses do not accept.
func unsupportedPromptContent(description string) error {
	return &CapabilityUnsupportedError{
		Message: fmt.Sprintf(
			"The harness layer does not support %s in user messages. Pass text parts and inline images as bytes or base64 data URLs.",
			description,
		),
	}
}
