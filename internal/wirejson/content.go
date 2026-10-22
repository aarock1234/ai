package wirejson

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// Content is one encoded item of generated content. Every content type is
// also a StreamPart.
type Content interface {
	isContent()
}

// TextContent is encoded text or reasoning. Only text has citations.
type TextContent struct {
	Type             string                 `json:"type"`
	Text             string                 `json:"text"`
	Citations        []Citation             `json:"citations,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// CustomContent is encoded provider-specific content.
type CustomContent struct {
	Type             string                 `json:"type"`
	Kind             string                 `json:"kind"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// FileContent is an encoded generated file or reasoning file. Data holds the
// file content in the encoder's file data encoding.
type FileContent struct {
	Type             string                 `json:"type"`
	MediaType        string                 `json:"mediaType"`
	Data             FileData               `json:"data"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ToolApprovalRequestContent is an encoded tool approval request of a
// provider-executed tool.
type ToolApprovalRequestContent struct {
	Type             string                 `json:"type"`
	ApprovalID       string                 `json:"approvalId"`
	ToolCallID       string                 `json:"toolCallId"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ToolCallContent is an encoded tool call. Input is the JSON text the model
// generated.
type ToolCallContent struct {
	Type             string                 `json:"type"`
	ToolCallID       string                 `json:"toolCallId"`
	ToolName         string                 `json:"toolName"`
	Input            string                 `json:"input"`
	ProviderExecuted bool                   `json:"providerExecuted,omitempty"`
	Dynamic          bool                   `json:"dynamic,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ToolResultContent is an encoded result of a provider-executed tool.
type ToolResultContent struct {
	Type             string                 `json:"type"`
	ToolCallID       string                 `json:"toolCallId"`
	ToolName         string                 `json:"toolName"`
	Result           json.RawMessage        `json:"result"`
	IsError          bool                   `json:"isError,omitempty"`
	Preliminary      bool                   `json:"preliminary,omitempty"`
	Dynamic          bool                   `json:"dynamic,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// Source is an encoded URL or document source.
type Source struct {
	Type             string                 `json:"type"`
	SourceType       model.SourceType       `json:"sourceType"`
	ID               string                 `json:"id"`
	URL              string                 `json:"url,omitempty"`
	Title            string                 `json:"title,omitempty"`
	MediaType        string                 `json:"mediaType,omitempty"`
	Filename         string                 `json:"filename,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// Citation is an encoded citation of a source in generated text.
type Citation struct {
	Source     Source `json:"source"`
	StartIndex *int   `json:"startIndex,omitempty"`
	EndIndex   *int   `json:"endIndex,omitempty"`
	CitedText  string `json:"citedText,omitempty"`
}

func (TextContent) isContent()                {}
func (CustomContent) isContent()              {}
func (FileContent) isContent()                {}
func (ToolApprovalRequestContent) isContent() {}
func (ToolCallContent) isContent()            {}
func (ToolResultContent) isContent()          {}
func (Source) isContent()                     {}

// Contents encodes generated content, skipping unknown items.
func (e Encoder) Contents(contents []model.Content) []Content {
	encoded := make([]Content, 0, len(contents))

	for _, content := range contents {
		if wire, ok := e.content(content); ok {
			encoded = append(encoded, wire)
		}
	}

	return encoded
}

// content encodes the content types, which are stream parts too.
func (e Encoder) content(content any) (Content, bool) {
	switch typed := content.(type) {
	case model.Text:
		return TextContent{
			Type:             "text",
			Text:             typed.Text,
			Citations:        EncodeCitations(typed.Citations),
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.Reasoning:
		return TextContent{
			Type:             "reasoning",
			Text:             typed.Text,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.Custom:
		return CustomContent{
			Type:             "custom",
			Kind:             typed.Kind,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.File:
		return FileContent{
			Type:             "file",
			MediaType:        typed.MediaType,
			Data:             e.fileData(typed.Data),
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.ReasoningFile:
		return FileContent{
			Type:             "reasoning-file",
			MediaType:        typed.MediaType,
			Data:             e.fileData(typed.Data),
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.ToolApprovalRequest:
		return ToolApprovalRequestContent{
			Type:             "tool-approval-request",
			ApprovalID:       typed.ApprovalID,
			ToolCallID:       typed.ToolCallID,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.ToolCall:
		return ToolCallContent{
			Type:             "tool-call",
			ToolCallID:       typed.ToolCallID,
			ToolName:         typed.ToolName,
			Input:            typed.Input,
			ProviderExecuted: typed.ProviderExecuted,
			Dynamic:          typed.Dynamic,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.ToolResult:
		return ToolResultContent{
			Type:             "tool-result",
			ToolCallID:       typed.ToolCallID,
			ToolName:         typed.ToolName,
			Result:           RawValue(typed.Result),
			IsError:          typed.IsError,
			Preliminary:      typed.Preliminary,
			Dynamic:          typed.Dynamic,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.Source:
		return EncodeSource(typed), true

	default:
		return nil, false
	}
}

// EncodeSource encodes a source.
func EncodeSource(source model.Source) Source {
	return Source{
		Type:             "source",
		SourceType:       source.SourceType,
		ID:               source.ID,
		URL:              source.URL,
		Title:            source.Title,
		MediaType:        source.MediaType,
		Filename:         source.Filename,
		ProviderMetadata: source.ProviderMetadata,
	}
}

// EncodeCitations encodes citations, returning nil when there are none.
func EncodeCitations(citations []model.Citation) []Citation {
	if len(citations) == 0 {
		return nil
	}

	encoded := make([]Citation, 0, len(citations))

	for _, citation := range citations {
		encoded = append(encoded, Citation{
			Source:     EncodeSource(citation.Source),
			StartIndex: citation.StartIndex,
			EndIndex:   citation.EndIndex,
			CitedText:  citation.CitedText,
		})
	}

	return encoded
}
