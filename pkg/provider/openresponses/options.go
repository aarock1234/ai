package openresponses

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// ReasoningSummary controls reasoning summary output.
type ReasoningSummary string

// Reasoning summary modes.
const (
	ReasoningSummaryConcise  ReasoningSummary = "concise"
	ReasoningSummaryDetailed ReasoningSummary = "detailed"
	ReasoningSummaryAuto     ReasoningSummary = "auto"
)

// ResponsesOptions are the provider options of Responses API models, read
// from the provider's own namespace.
type ResponsesOptions struct {
	// ReasoningEffort is a provider-native reasoning effort. It is passed
	// through and takes precedence over the call's reasoning level.
	ReasoningEffort *string `json:"reasoningEffort,omitempty"`

	// ReasoningSummary controls reasoning summary output.
	ReasoningSummary *ReasoningSummary `json:"reasoningSummary,omitempty"`
}

// ImageDetail is the detail level of an input image.
type ImageDetail string

// Image detail levels.
const (
	ImageDetailLow  ImageDetail = "low"
	ImageDetailHigh ImageDetail = "high"
	ImageDetailAuto ImageDetail = "auto"
)

// FilePartOptions are the provider options of image file parts and image
// tool result content, read from the provider's own namespace.
type FilePartOptions struct {
	// ImageDetail defaults to ImageDetailAuto.
	ImageDetail ImageDetail `json:"imageDetail,omitempty"`
}

// CustomToolFormatType selects free text or a grammar.
type CustomToolFormatType string

// Custom tool format types.
const (
	CustomToolFormatText    CustomToolFormatType = "text"
	CustomToolFormatGrammar CustomToolFormatType = "grammar"
)

// GrammarSyntax is the syntax of a custom tool grammar.
type GrammarSyntax string

// Grammar syntaxes.
const (
	GrammarRegex GrammarSyntax = "regex"
	GrammarLark  GrammarSyntax = "lark"
)

// CustomToolArgs configure a caller-executed custom tool, which accepts a
// raw string.
type CustomToolArgs struct {
	// Description optionally describes what the tool does.
	Description string `json:"description,omitempty"`

	// Format optionally constrains the raw text.
	Format *CustomToolFormat `json:"format,omitempty"`
}

// CustomToolFormat is the raw text format of a custom tool. Grammars need a
// syntax and a definition.
type CustomToolFormat struct {
	Type       CustomToolFormatType `json:"type"`
	Syntax     GrammarSyntax        `json:"syntax,omitempty"`
	Definition string               `json:"definition,omitempty"`
}

// CustomTool returns a caller-executed custom tool named name. Custom tools
// accept a raw string, optionally constrained by a grammar, and require
// endpoint support.
func (p *Provider) CustomTool(name string, args CustomToolArgs) model.ProviderTool {
	encoded, err := json.Marshal(args)
	if err != nil {
		panic(fmt.Sprintf("encode custom tool arguments: %v", err))
	}

	return model.ProviderTool{
		ID:   p.customToolID,
		Name: name,
		Args: encoded,
	}
}

// URLCitation is a URL citation annotation of output text.
type URLCitation struct {
	Type       string `json:"type"`
	StartIndex int    `json:"start_index"`
	EndIndex   int    `json:"end_index"`
	URL        string `json:"url"`
	Title      string `json:"title"`
}

// SummaryText is one reasoning summary part.
type SummaryText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ReasoningText is one reasoning content part.
type ReasoningText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ItemMetadata is the provider metadata of tool calls, stored under the
// provider's name.
type ItemMetadata struct {
	ItemID string `json:"itemId"`
}

// TextMetadata is the provider metadata of text, stored under the
// provider's name. Replaying it keeps the message item and its citations.
type TextMetadata struct {
	ItemID      string        `json:"itemId"`
	Annotations []URLCitation `json:"annotations,omitempty"`
}

// ReasoningMetadata is the provider metadata of reasoning, stored under the
// provider's name. Replaying it restores the reasoning item.
type ReasoningMetadata struct {
	ItemID           string        `json:"itemId"`
	ReasoningSummary []SummaryText `json:"reasoningSummary"`

	// ReasoningContent is null when the item had no content.
	ReasoningContent *[]ReasoningText `json:"reasoningContent"`

	ReasoningEncryptedContent string `json:"reasoningEncryptedContent,omitempty"`
}

// Content part types used in replayed metadata.
const (
	summaryTextType   = "summary_text"
	reasoningTextType = "reasoning_text"
	urlCitationType   = "url_citation"
)

// readResponsesOptions decodes and validates the provider options.
func readResponsesOptions(options model.ProviderOptions, namespace string) (ResponsesOptions, error) {
	responsesOptions, _, err := model.ProviderValue[ResponsesOptions](options, namespace)
	if err != nil {
		return ResponsesOptions{}, err
	}

	if summary := responsesOptions.ReasoningSummary; summary != nil {
		switch *summary {
		case ReasoningSummaryConcise, ReasoningSummaryDetailed, ReasoningSummaryAuto:
		default:
			return ResponsesOptions{}, fmt.Errorf("%w: %s provider options: unsupported reasoningSummary %q", model.ErrInvalidArgument, namespace, *summary)
		}
	}

	return responsesOptions, nil
}

// replayData is the metadata of an earlier output part, replayed through
// its provider options. Members of an unexpected shape are ignored.
type replayData struct {
	ItemID                    json.RawMessage `json:"itemId"`
	ReasoningSummary          json.RawMessage `json:"reasoningSummary"`
	ReasoningContent          json.RawMessage `json:"reasoningContent"`
	ReasoningEncryptedContent json.RawMessage `json:"reasoningEncryptedContent"`
	Annotations               json.RawMessage `json:"annotations"`
	ImageDetail               json.RawMessage `json:"imageDetail"`
}

// readReplayData decodes the namespace of a part's provider options. A
// namespace that is not an object has no data.
func readReplayData(options model.ProviderOptions, namespace string) replayData {
	data, _, err := model.ProviderValue[replayData](options, namespace)
	if err != nil {
		return replayData{}
	}

	return data
}

// itemID returns the replayed item ID when it is a string.
func (d replayData) itemID() *string {
	return stringValue(d.ItemID)
}

// encryptedContent returns the replayed encrypted reasoning when it is a
// string.
func (d replayData) encryptedContent() *string {
	return stringValue(d.ReasoningEncryptedContent)
}

// hasReasoningContent reports whether the reasoning content member is
// present, even when it is null.
func (d replayData) hasReasoningContent() bool {
	return d.ReasoningContent != nil
}

// imageDetail returns the replayed image detail, or ImageDetailAuto.
func (d replayData) imageDetail() ImageDetail {
	detail := stringValue(d.ImageDetail)
	if detail == nil {
		return ImageDetailAuto
	}

	switch ImageDetail(*detail) {
	case ImageDetailLow, ImageDetailHigh, ImageDetailAuto:
		return ImageDetail(*detail)
	default:
		return ImageDetailAuto
	}
}

// summary returns the replayed summary when every part is a summary text.
func (d replayData) summary() []SummaryText {
	var parts []SummaryText
	if !decodeTypedParts(d.ReasoningSummary, &parts) {
		return nil
	}

	for _, part := range parts {
		if part.Type != summaryTextType {
			return nil
		}
	}

	return parts
}

// content returns the replayed reasoning content when every part is a
// reasoning text.
func (d replayData) content() []ReasoningText {
	var parts []ReasoningText
	if !decodeTypedParts(d.ReasoningContent, &parts) {
		return nil
	}

	for _, part := range parts {
		if part.Type != reasoningTextType {
			return nil
		}
	}

	return parts
}

// annotations returns the replayed annotations when every annotation is a
// complete URL citation.
func (d replayData) annotations() []URLCitation {
	citations, ok := validCitations(d.Annotations)
	if !ok {
		return nil
	}

	return citations
}

// citationShape detects missing citation members.
type citationShape struct {
	Type       *string  `json:"type"`
	StartIndex *float64 `json:"start_index"`
	EndIndex   *float64 `json:"end_index"`
	URL        *string  `json:"url"`
	Title      *string  `json:"title"`
}

// validCitations decodes annotations when they are an array of complete URL
// citations.
func validCitations(raw json.RawMessage) ([]URLCitation, bool) {
	var shapes []citationShape
	if !decodeTypedParts(raw, &shapes) {
		return nil, false
	}

	citations := make([]URLCitation, 0, len(shapes))
	for _, shape := range shapes {
		complete := shape.Type != nil && *shape.Type == urlCitationType &&
			shape.StartIndex != nil && shape.EndIndex != nil &&
			shape.URL != nil && shape.Title != nil
		if !complete {
			return nil, false
		}

		citations = append(citations, URLCitation{
			Type:       urlCitationType,
			StartIndex: int(*shape.StartIndex),
			EndIndex:   int(*shape.EndIndex),
			URL:        *shape.URL,
			Title:      *shape.Title,
		})
	}

	return citations, true
}

// decodeTypedParts decodes a JSON array into target and reports success.
func decodeTypedParts[T any](raw json.RawMessage, target *[]T) bool {
	if !bytes.HasPrefix(bytes.TrimSpace(raw), []byte("[")) {
		return false
	}

	return json.Unmarshal(raw, target) == nil
}

// stringValue returns raw when it is a JSON string.
func stringValue(raw json.RawMessage) *string {
	var text string
	if len(raw) == 0 || json.Unmarshal(raw, &text) != nil {
		return nil
	}

	return &text
}
