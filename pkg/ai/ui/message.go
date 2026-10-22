package ui

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// Role is the author of a UI message.
type Role string

// Message roles.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is a chat message as the UI renders it, the upstream UIMessage. It
// is exchanged between the frontend and the server.
type Message struct {
	// ID uniquely identifies the message.
	ID string `json:"id"`

	Role Role `json:"role"`

	// Metadata is application-defined message metadata. Nil means none.
	Metadata json.RawMessage `json:"metadata,omitempty"`

	// Parts are the contents of the message in display order. Prefer setting
	// system instructions on the server over sending system messages.
	Parts []Part `json:"parts"`
}

// MarshalJSON encodes the message, with an empty parts array when it has no
// parts.
func (m Message) MarshalJSON() ([]byte, error) {
	type fields Message

	encoded := fields(m)
	if encoded.Parts == nil {
		encoded.Parts = []Part{}
	}

	return encodeJSON(encoded)
}

// UnmarshalJSON decodes the message and its parts. It rejects parts of
// unknown types.
func (m *Message) UnmarshalJSON(data []byte) error {
	var encoded struct {
		ID       string            `json:"id"`
		Role     Role              `json:"role"`
		Metadata json.RawMessage   `json:"metadata"`
		Parts    []json.RawMessage `json:"parts"`
	}

	if err := json.Unmarshal(data, &encoded); err != nil {
		return err
	}

	parts := make([]Part, 0, len(encoded.Parts))
	for index, raw := range encoded.Parts {
		part, err := UnmarshalPart(raw)
		if err != nil {
			return fmt.Errorf("decode part %d: %w", index, err)
		}

		parts = append(parts, part)
	}

	*m = Message{
		ID:       encoded.ID,
		Role:     encoded.Role,
		Metadata: encoded.Metadata,
		Parts:    parts,
	}

	return nil
}

// clone returns a copy of the message whose parts can be replaced without
// affecting m. The parts themselves are values that are never modified in
// place.
func (m Message) clone() Message {
	m.Parts = slices.Clone(m.Parts)

	return m
}

// Part is one part of a Message: TextPart, CustomPart, ReasoningPart,
// ToolPart, SourceURLPart, SourceDocumentPart, FilePart, ReasoningFilePart,
// DataPart or StepStartPart.
type Part interface {
	isPart()
}

// PartState is the streaming state of a text or reasoning part.
type PartState string

// Part states.
const (
	PartStreaming PartState = "streaming"
	PartDone      PartState = "done"
)

// TextPart is text.
type TextPart struct {
	Text string `json:"text"`

	// Citations are references that support the text.
	Citations []Citation `json:"citations,omitempty"`

	State            PartState              `json:"state,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// CustomPart is provider-specific content. Kind has the form
// "provider.type".
type CustomPart struct {
	Kind             string                 `json:"kind"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ReasoningPart is reasoning text.
type ReasoningPart struct {
	ID               string                 `json:"id,omitempty"`
	Text             string                 `json:"text"`
	State            PartState              `json:"state,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// SourceURLPart is a URL source.
type SourceURLPart struct {
	SourceID         string                 `json:"sourceId"`
	URL              string                 `json:"url"`
	Title            string                 `json:"title,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// SourceDocumentPart is a document source.
type SourceDocumentPart struct {
	SourceID         string                 `json:"sourceId"`
	MediaType        string                 `json:"mediaType"`
	Title            string                 `json:"title"`
	Filename         string                 `json:"filename,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// FilePart is a file.
type FilePart struct {
	// MediaType is a full IANA media type, such as "image/png", or only its
	// top-level segment, such as "image".
	MediaType string `json:"mediaType"`

	Filename string `json:"filename,omitempty"`

	// URL is the URL of a hosted file or a data URL.
	URL string `json:"url"`

	// ProviderReference maps provider names to the IDs of the uploaded file.
	// When present, it takes precedence over URL in model messages.
	ProviderReference map[string]string `json:"providerReference,omitempty"`

	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ReasoningFilePart is a file produced during reasoning.
type ReasoningFilePart struct {
	MediaType string `json:"mediaType"`

	// URL is the URL of a hosted file or a data URL.
	URL string `json:"url"`

	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// StepStartPart marks the start of a step.
type StepStartPart struct{}

// DataPart is application-defined data, encoded with the type
// "data-" + Name.
type DataPart struct {
	Name string

	// ID identifies the part so that later data chunks can replace it.
	ID string

	Data json.RawMessage
}

// dataPartFields are the encoded fields of a DataPart besides its type.
type dataPartFields struct {
	ID   string          `json:"id,omitempty"`
	Data json.RawMessage `json:"data,omitempty"`
}

// MarshalJSON encodes the part with the type "text".
func (p TextPart) MarshalJSON() ([]byte, error) {
	type fields TextPart

	return marshalWithType("text", fields(p))
}

// MarshalJSON encodes the part with the type "custom".
func (p CustomPart) MarshalJSON() ([]byte, error) {
	type fields CustomPart

	return marshalWithType("custom", fields(p))
}

// MarshalJSON encodes the part with the type "reasoning".
func (p ReasoningPart) MarshalJSON() ([]byte, error) {
	type fields ReasoningPart

	return marshalWithType("reasoning", fields(p))
}

// MarshalJSON encodes the part with the type "source-url".
func (p SourceURLPart) MarshalJSON() ([]byte, error) {
	type fields SourceURLPart

	return marshalWithType("source-url", fields(p))
}

// MarshalJSON encodes the part with the type "source-document".
func (p SourceDocumentPart) MarshalJSON() ([]byte, error) {
	type fields SourceDocumentPart

	return marshalWithType("source-document", fields(p))
}

// MarshalJSON encodes the part with the type "file".
func (p FilePart) MarshalJSON() ([]byte, error) {
	type fields FilePart

	return marshalWithType("file", fields(p))
}

// MarshalJSON encodes the part with the type "reasoning-file".
func (p ReasoningFilePart) MarshalJSON() ([]byte, error) {
	type fields ReasoningFilePart

	return marshalWithType("reasoning-file", fields(p))
}

// MarshalJSON encodes the part with the type "step-start".
func (StepStartPart) MarshalJSON() ([]byte, error) {
	return marshalWithType("step-start", struct{}{})
}

// MarshalJSON encodes the part with the type "data-" + Name.
func (p DataPart) MarshalJSON() ([]byte, error) {
	return marshalWithType(dataTypePrefix+p.Name, dataPartFields{
		ID:   p.ID,
		Data: p.Data,
	})
}

func (TextPart) isPart()           {}
func (CustomPart) isPart()         {}
func (ReasoningPart) isPart()      {}
func (ToolPart) isPart()           {}
func (SourceURLPart) isPart()      {}
func (SourceDocumentPart) isPart() {}
func (FilePart) isPart()           {}
func (ReasoningFilePart) isPart()  {}
func (DataPart) isPart()           {}
func (StepStartPart) isPart()      {}

// Type prefixes of the parts whose type embeds a name.
const (
	dataTypePrefix = "data-"
	toolTypePrefix = "tool-"
)

// dynamicToolType is the type of dynamic tool parts.
const dynamicToolType = "dynamic-tool"

// UnmarshalPart decodes one message part, selected by its type. It rejects
// unknown types.
func UnmarshalPart(data []byte) (Part, error) {
	partType, err := decodeType(data)
	if err != nil {
		return nil, fmt.Errorf("decode part type: %w", err)
	}

	switch {
	case partType == "text":
		return decodePart[TextPart](data)
	case partType == "custom":
		return decodePart[CustomPart](data)
	case partType == "reasoning":
		return decodePart[ReasoningPart](data)
	case partType == "source-url":
		return decodePart[SourceURLPart](data)
	case partType == "source-document":
		return decodePart[SourceDocumentPart](data)
	case partType == "file":
		return decodePart[FilePart](data)
	case partType == "reasoning-file":
		return decodePart[ReasoningFilePart](data)
	case partType == "step-start":
		return StepStartPart{}, nil
	case partType == dynamicToolType:
		return decodeToolPart(data, "", true)
	case strings.HasPrefix(partType, toolTypePrefix):
		return decodeToolPart(data, strings.TrimPrefix(partType, toolTypePrefix), false)
	case strings.HasPrefix(partType, dataTypePrefix):
		return decodeDataPart(data, strings.TrimPrefix(partType, dataTypePrefix))
	default:
		return nil, fmt.Errorf("unknown part type %q", partType)
	}
}

func decodePart[Variant Part](data []byte) (Part, error) {
	var part Variant
	if err := json.Unmarshal(data, &part); err != nil {
		return nil, err
	}

	return part, nil
}

func decodeDataPart(data []byte, name string) (Part, error) {
	var fields dataPartFields
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}

	return DataPart{
		Name: name,
		ID:   fields.ID,
		Data: fields.Data,
	}, nil
}

// Citation is a reference supporting a text part.
type Citation struct {
	Source CitationSource `json:"source"`

	// StartIndex and EndIndex are optional character offsets into the text.
	StartIndex *int `json:"startIndex,omitempty"`
	EndIndex   *int `json:"endIndex,omitempty"`

	CitedText string `json:"citedText,omitempty"`
}

// CitationSource is the source of a citation. URL sources set URL; document
// sources set MediaType, Title and optionally Filename.
type CitationSource struct {
	SourceType       model.SourceType       `json:"sourceType"`
	ID               string                 `json:"id"`
	URL              string                 `json:"url,omitempty"`
	Title            string                 `json:"title,omitempty"`
	MediaType        string                 `json:"mediaType,omitempty"`
	Filename         string                 `json:"filename,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// MarshalJSON encodes the source with the type "source".
func (s CitationSource) MarshalJSON() ([]byte, error) {
	type fields CitationSource

	return marshalWithType("source", fields(s))
}

// citationsFromModel converts model citations to their wire form.
func citationsFromModel(citations []model.Citation) []Citation {
	if citations == nil {
		return nil
	}

	converted := make([]Citation, 0, len(citations))
	for _, citation := range citations {
		converted = append(converted, Citation{
			Source: CitationSource{
				SourceType:       citation.Source.SourceType,
				ID:               citation.Source.ID,
				URL:              citation.Source.URL,
				Title:            citation.Source.Title,
				MediaType:        citation.Source.MediaType,
				Filename:         citation.Source.Filename,
				ProviderMetadata: citation.Source.ProviderMetadata,
			},
			StartIndex: citation.StartIndex,
			EndIndex:   citation.EndIndex,
			CitedText:  citation.CitedText,
		})
	}

	return converted
}
