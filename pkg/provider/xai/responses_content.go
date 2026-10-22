package xai

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// emptyObjectInput is the input of tool calls without arguments.
const emptyObjectInput = "{}"

// contentBuilder converts response output items into content.
type contentBuilder struct {
	names      toolNames
	generateID func() string

	content         []model.Content
	hasFunctionCall bool
}

func (b *contentBuilder) addItem(item outputItem) error {
	switch {
	case item.Type == itemFileSearchCall:
		return b.addParts(fileSearchParts(item, b.names.fileSearch))

	case item.Type == itemImageGenerationCall:
		return b.addParts(imageGenerationParts(item, b.names.imageGeneration))

	case serverToolCallTypes[item.Type]:
		return b.addParts(serverToolParts(item, b.names.toolName(item), true))

	case item.Type == itemMessage:
		for _, part := range item.Content {
			if part.Text != "" {
				b.content = append(b.content, model.Text{
					Text: part.Text,
				})
			}

			for _, source := range citationSources(part.Annotations, b.generateID) {
				b.content = append(b.content, source)
			}
		}

	case item.Type == itemFunctionCall:
		b.hasFunctionCall = true
		b.content = append(b.content, model.ToolCall{
			ToolCallID: optional.Or(item.CallID, ""),
			ToolName:   optional.Or(item.Name, ""),
			Input:      optional.Or(item.Arguments, ""),
		})

	case item.Type == itemReasoning:
		if reasoningContent, ok := reasoningOf(item); ok {
			b.content = append(b.content, reasoningContent)
		}
	}

	return nil
}

// addParts appends tool call and result parts.
func (b *contentBuilder) addParts(parts []model.StreamPart, err error) error {
	if err != nil {
		return err
	}

	for _, part := range parts {
		switch part := part.(type) {
		case model.ToolCall:
			b.content = append(b.content, part)
		case model.ToolResult:
			b.content = append(b.content, part)
		}
	}

	return nil
}

// fileSearchParts returns a file search call and its result.
func fileSearchParts(item outputItem, toolName string) ([]model.StreamPart, error) {
	result, err := encodeResult(fileSearchResult(item))
	if err != nil {
		return nil, err
	}

	return []model.StreamPart{
		providerToolCall(item.ID, toolName, ""),
		model.ToolResult{
			ToolCallID: item.ID,
			ToolName:   toolName,
			Result:     result,
		},
	}, nil
}

func fileSearchResult(item outputItem) FileSearchResult {
	result := FileSearchResult{
		Queries: item.Queries,
	}

	if result.Queries == nil {
		result.Queries = []string{}
	}

	if item.Results != nil {
		matches := make([]FileSearchMatch, 0, len(*item.Results))
		for _, match := range *item.Results {
			matches = append(matches, FileSearchMatch(match))
		}

		result.Results = &matches
	}

	return result
}

// imageGenerationParts returns an image generation call and its result, or
// an error result when no image was produced.
func imageGenerationParts(item outputItem, toolName string) ([]model.StreamPart, error) {
	call := providerToolCall(item.ID, toolName, emptyObjectInput)

	result, err := imageGenerationResult(item, toolName)
	if err != nil {
		return nil, err
	}

	return []model.StreamPart{
		call,
		result,
	}, nil
}

func imageGenerationResult(item outputItem, toolName string) (model.ToolResult, error) {
	if item.Result == nil {
		message := fmt.Sprintf("Image generation failed (status: %s).", item.Status)

		return model.ToolResult{
			ToolCallID: item.ID,
			ToolName:   toolName,
			Result:     jsonobject.EncodeString(message),
			IsError:    true,
		}, nil
	}

	result, err := encodeResult(ImageGenerationResult{
		Result: *item.Result,
		Prompt: item.Prompt,
	})
	if err != nil {
		return model.ToolResult{}, err
	}

	return model.ToolResult{
		ToolCallID: item.ID,
		ToolName:   toolName,
		Result:     result,
	}, nil
}

// serverToolParts returns a server-side tool call and, for web searches,
// its result.
func serverToolParts(item outputItem, toolName string, withCall bool) ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if withCall {
		parts = append(parts, providerToolCall(item.ID, toolName, toolInput(item)))
	}

	if item.Type != itemWebSearchCall {
		return parts, nil
	}

	result, err := encodeResult(mapWebSearchAction(item.Action))
	if err != nil {
		return nil, err
	}

	return append(parts, model.ToolResult{
		ToolCallID: item.ID,
		ToolName:   toolName,
		Result:     result,
	}), nil
}

func providerToolCall(toolCallID string, toolName string, input string) model.ToolCall {
	return model.ToolCall{
		ToolCallID:       toolCallID,
		ToolName:         toolName,
		Input:            input,
		ProviderExecuted: true,
	}
}

// citationSources returns the URL citations of a content part. A citation
// without a title uses its URL.
func citationSources(annotations []annotation, generateID func() string) []model.Source {
	var sources []model.Source

	for _, citation := range annotations {
		if citation.Type != "url_citation" || citation.URL == nil {
			continue
		}

		sources = append(sources, model.Source{
			SourceType: model.SourceURL,
			ID:         generateID(),
			URL:        *citation.URL,
			Title:      optional.Or(citation.Title, *citation.URL),
		})
	}

	return sources
}

// reasoningOf joins the summary, or the content when there is no summary.
// Reasoning with neither text nor encrypted content is omitted.
func reasoningOf(item outputItem) (model.Reasoning, bool) {
	var text strings.Builder

	if len(item.Summary) > 0 {
		for _, part := range item.Summary {
			text.WriteString(part.Text)
		}
	} else {
		for _, part := range item.Content {
			text.WriteString(part.Text)
		}
	}

	encrypted := optional.Or(item.EncryptedContent, "")
	if text.Len() == 0 && encrypted == "" {
		return model.Reasoning{}, false
	}

	return model.Reasoning{
		Text:             text.String(),
		ProviderMetadata: reasoningMetadata(item.ID, encrypted),
	}, true
}

// reasoningMetadata returns the replay metadata of reasoning, or nil when
// there is nothing to replay.
func reasoningMetadata(itemID string, encryptedContent string) model.ProviderMetadata {
	if itemID == "" && encryptedContent == "" {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: ReasoningMetadata{
			ItemID:                    itemID,
			ReasoningEncryptedContent: encryptedContent,
		},
	}
}
