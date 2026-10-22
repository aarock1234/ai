package google

import (
	"encoding/base64"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

// skipThoughtSignatureValidator is the value Google documents for replaying
// function calls whose original thought signature is not available.
const skipThoughtSignatureValidator = "skip_thought_signature_validator"

type promptSettings struct {
	isGemma   bool
	isGemini3 bool
	toolNames toolnames.Mapping

	// metadataNamespaces are read in order for replayed part metadata.
	metadataNamespaces []string

	// isVertex rejects provider file references, which only the Gemini API
	// Files API resolves, and omits function call IDs.
	isVertex bool

	// functionResponseURLs are the file URLs that function responses may
	// reference instead of describing them as text.
	functionResponseURLs map[string][]*regexp.Regexp
}

type convertedPrompt struct {
	systemInstruction *systemInstruction
	contents          []content
	warnings          []model.Warning
}

// promptConverter converts a prompt into Gemini contents. System messages
// become the system instruction, assistant messages become model contents,
// and user and tool messages become user contents.
type promptConverter struct {
	settings promptSettings

	systemParts   []textPart
	contents      []content
	systemAllowed bool

	// unsignedToolNames lists replayed Gemini 3 function calls that received
	// the skip-validator signature.
	unsignedToolNames []string
}

func convertPrompt(prompt []model.Message, settings promptSettings) (convertedPrompt, error) {
	converter := &promptConverter{
		settings:      settings,
		systemAllowed: true,
	}

	for _, message := range prompt {
		if err := converter.convertMessage(message); err != nil {
			return convertedPrompt{}, err
		}
	}

	return converter.result(), nil
}

func (c *promptConverter) convertMessage(message model.Message) error {
	switch message := message.(type) {
	case model.SystemMessage:
		if !c.systemAllowed {
			return fmt.Errorf("%w: system messages are only supported at the beginning of the conversation", model.ErrUnsupported)
		}

		c.systemParts = append(c.systemParts, textPart{
			Text: message.Content,
		})

		return nil

	case model.UserMessage:
		c.systemAllowed = false

		return c.convertUserMessage(message)

	case model.AssistantMessage:
		c.systemAllowed = false

		return c.convertAssistantMessage(message)

	case model.ToolMessage:
		c.systemAllowed = false

		return c.convertToolMessage(message)

	default:
		return fmt.Errorf("%w: unsupported message type %T", model.ErrInvalidPrompt, message)
	}
}

func (c *promptConverter) result() convertedPrompt {
	if c.settings.isVertex {
		removeFunctionCallIDs(c.contents)
	}

	converted := convertedPrompt{
		contents: c.contents,
	}

	if len(c.unsignedToolNames) > 0 {
		converted.warnings = append(converted.warnings, missingSignatureWarning(c.unsignedToolNames))
	}

	if len(c.systemParts) == 0 {
		return converted
	}

	if !c.settings.isGemma {
		converted.systemInstruction = &systemInstruction{
			Parts: c.systemParts,
		}

		return converted
	}

	// Gemma models have no system instruction, so the system text opens the
	// first user content instead.
	if len(converted.contents) > 0 && converted.contents[0].Role == roleUser {
		systemText := joinSystemText(c.systemParts) + "\n\n"
		converted.contents[0].Parts = slices.Insert(converted.contents[0].Parts, 0, part{
			Text: &systemText,
		})
	}

	return converted
}

// removeFunctionCallIDs clears the IDs of function calls and responses,
// which Vertex AI does not accept.
func removeFunctionCallIDs(contents []content) {
	for _, item := range contents {
		for index := range item.Parts {
			converted := &item.Parts[index]

			if converted.FunctionCall != nil {
				converted.FunctionCall.ID = ""
			}

			if converted.FunctionResponse != nil {
				converted.FunctionResponse.ID = ""
			}
		}
	}
}

func joinSystemText(parts []textPart) string {
	texts := make([]string, 0, len(parts))
	for _, systemPart := range parts {
		texts = append(texts, systemPart.Text)
	}

	return strings.Join(texts, "\n\n")
}

func missingSignatureWarning(toolNames []string) model.Warning {
	var uniqueNames []string
	for _, name := range toolNames {
		quoted := "`" + name + "`"
		if !slices.Contains(uniqueNames, quoted) {
			uniqueNames = append(uniqueNames, quoted)
		}
	}

	message := fmt.Sprintf(
		"Replayed %d `functionCall` part(s) for a Gemini 3 model without a `thoughtSignature` (tools: %s). "+
			"Injected the documented `skip_thought_signature_validator` sentinel to keep the request from failing with HTTP 400. "+
			"The likely cause is application code that drops `providerOptions.google.thoughtSignature` when persisting or serializing assistant tool-call messages. "+
			"See https://ai.google.dev/gemini-api/docs/thought-signatures.",
		len(toolNames),
		strings.Join(uniqueNames, ", "),
	)

	return model.Warning{
		Type:    model.WarningOther,
		Message: message,
	}
}

func (c *promptConverter) convertUserMessage(message model.UserMessage) error {
	parts := make([]part, 0, len(message.Content))

	for _, userPart := range message.Content {
		switch userPart := userPart.(type) {
		case model.TextPart:
			parts = append(parts, part{
				Text: new(userPart.Text),
			})

		case model.FilePart:
			converted, err := c.convertUserFile(userPart)
			if err != nil {
				return err
			}

			parts = append(parts, converted)

		default:
			return fmt.Errorf("%w: unsupported user part %T", model.ErrInvalidPrompt, userPart)
		}
	}

	c.contents = append(c.contents, content{
		Role:  roleUser,
		Parts: parts,
	})

	return nil
}

func (c *promptConverter) convertUserFile(file model.FilePart) (part, error) {
	switch data := file.Data.(type) {
	case model.FileURL:
		mediaType, err := mediatype.ResolveFile(file.MediaType, file.Data)
		if err != nil {
			return part{}, err
		}

		return part{
			FileData: &FileData{
				MimeType: mediaType,
				FileURI:  data.URL.String(),
			},
		}, nil

	case model.FileReference:
		return c.referencePart(file, data)

	case model.FileText:
		return part{
			InlineData: textInlineData(file.MediaType, data),
		}, nil

	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(file.MediaType, file.Data)
		if err != nil {
			return part{}, err
		}

		return part{
			InlineData: &InlineData{
				MimeType: mediaType,
				Data:     base64.StdEncoding.EncodeToString(data.Data),
			},
		}, nil

	default:
		return part{}, fmt.Errorf("%w: unsupported file data %T", model.ErrInvalidPrompt, file.Data)
	}
}

// referencePart refers to a file uploaded to the Google Files API.
func (c *promptConverter) referencePart(file model.FilePart, reference model.FileReference) (part, error) {
	if c.settings.isVertex {
		return part{}, errVertexFileReference
	}

	fileURI, ok := reference.IDs[Namespace]
	if !ok {
		return part{}, fmt.Errorf("%w: file reference has no %s file ID", model.ErrInvalidPrompt, Namespace)
	}

	resolved, err := mediatype.ResolveFile(file.MediaType, file.Data)
	if err != nil {
		return part{}, err
	}

	return part{
		FileData: &FileData{
			MimeType: resolved,
			FileURI:  fileURI,
		},
	}, nil
}

func textInlineData(mediaType string, text model.FileText) *InlineData {
	if !mediatype.IsFull(mediaType) {
		mediaType = "text/plain"
	}

	return &InlineData{
		MimeType: mediaType,
		Data:     base64.StdEncoding.EncodeToString([]byte(text.Text)),
	}
}

// partMetadata reads the Google metadata of a replayed part.
func (c *promptConverter) partMetadata(options model.ProviderOptions) (PartMetadata, error) {
	metadata, _, err := readProviderValue[PartMetadata](options, c.settings.metadataNamespaces)

	return metadata, err
}

func (m PartMetadata) isServerTool() bool {
	return m.ServerToolCallID != "" && m.ServerToolType != ""
}
