package wirejson

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// StreamPart is one encoded part of a language model stream.
type StreamPart interface {
	isStreamPart()
}

// StreamStartPart is the encoded first part of a stream.
type StreamStartPart struct {
	Type     string    `json:"type"`
	Warnings []Warning `json:"warnings"`
}

// BlockPart is an encoded start or end of a text, reasoning or tool input
// block. Only text ends have citations.
type BlockPart struct {
	Type             string                 `json:"type"`
	ID               string                 `json:"id"`
	Citations        []Citation             `json:"citations,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// DeltaPart is an encoded text, reasoning or tool input delta.
type DeltaPart struct {
	Type             string                 `json:"type"`
	ID               string                 `json:"id"`
	Delta            string                 `json:"delta"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// ToolInputStartPart is the encoded start of a streamed tool input.
type ToolInputStartPart struct {
	Type             string                  `json:"type"`
	ID               string                  `json:"id"`
	ToolName         string                  `json:"toolName"`
	InputEncoding    model.ToolInputEncoding `json:"inputEncoding,omitempty"`
	ProviderExecuted bool                    `json:"providerExecuted,omitempty"`
	Dynamic          bool                    `json:"dynamic,omitempty"`
	Title            string                  `json:"title,omitempty"`
	ProviderMetadata model.ProviderMetadata  `json:"providerMetadata,omitempty"`
}

// ResponseMetadataPart is encoded response metadata.
type ResponseMetadataPart struct {
	Type string `json:"type"`
	ResponseMetadata
}

// FinishPart is the encoded last part of a stream.
type FinishPart struct {
	Type             string                 `json:"type"`
	Usage            Usage                  `json:"usage"`
	FinishReason     FinishReason           `json:"finishReason"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// RawPart is an encoded raw provider chunk.
type RawPart struct {
	Type     string          `json:"type"`
	RawValue json.RawMessage `json:"rawValue"`
}

// ErrorPart is an encoded stream error, reported by its message.
type ErrorPart struct {
	Type  string `json:"type"`
	Error string `json:"error"`
}

func (StreamStartPart) isStreamPart()      {}
func (BlockPart) isStreamPart()            {}
func (DeltaPart) isStreamPart()            {}
func (ToolInputStartPart) isStreamPart()   {}
func (ResponseMetadataPart) isStreamPart() {}
func (FinishPart) isStreamPart()           {}
func (RawPart) isStreamPart()              {}
func (ErrorPart) isStreamPart()            {}

func (TextContent) isStreamPart()                {}
func (CustomContent) isStreamPart()              {}
func (FileContent) isStreamPart()                {}
func (ToolApprovalRequestContent) isStreamPart() {}
func (ToolCallContent) isStreamPart()            {}
func (ToolResultContent) isStreamPart()          {}
func (Source) isStreamPart()                     {}

// StreamParts encodes stream parts, skipping unknown parts.
func (e Encoder) StreamParts(parts []model.StreamPart) []StreamPart {
	encoded := make([]StreamPart, 0, len(parts))

	for _, part := range parts {
		if wire, ok := e.StreamPart(part); ok {
			encoded = append(encoded, wire)
		}
	}

	return encoded
}

// StreamPart encodes one stream part. It reports false for unknown parts.
func (e Encoder) StreamPart(part model.StreamPart) (StreamPart, bool) {
	if wire, ok := streamControlPart(part); ok {
		return wire, true
	}

	if wire, ok := streamBlockPart(part); ok {
		return wire, true
	}

	content, ok := e.content(part)
	if !ok {
		return nil, false
	}

	wire, ok := content.(StreamPart)

	return wire, ok
}

// streamControlPart encodes the parts that frame a stream: its start,
// response metadata, finish, raw chunks and errors.
func streamControlPart(part model.StreamPart) (StreamPart, bool) {
	switch typed := part.(type) {
	case model.StreamStart:
		return StreamStartPart{
			Type:     "stream-start",
			Warnings: EncodeWarnings(typed.Warnings),
		}, true

	case model.ResponseMetadata:
		return ResponseMetadataPart{
			Type:             "response-metadata",
			ResponseMetadata: EncodeResponseMetadata(typed),
		}, true

	case model.Finish:
		return FinishPart{
			Type:             "finish",
			Usage:            EncodeUsage(typed.Usage),
			FinishReason:     EncodeFinishReason(typed.FinishReason),
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.Raw:
		return RawPart{
			Type:     "raw",
			RawValue: RawValue(typed.Value),
		}, true

	case model.StreamError:
		return ErrorPart{
			Type:  "error",
			Error: errorMessage(typed.Err),
		}, true

	default:
		return nil, false
	}
}

// streamBlockPart encodes the starts, deltas and ends of text, reasoning
// and tool input blocks.
func streamBlockPart(part model.StreamPart) (StreamPart, bool) {
	switch typed := part.(type) {
	case model.TextStart:
		return BlockPart{
			Type:             "text-start",
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.TextDelta:
		return DeltaPart{
			Type:             "text-delta",
			ID:               typed.ID,
			Delta:            typed.Delta,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.TextEnd:
		return BlockPart{
			Type:             "text-end",
			ID:               typed.ID,
			Citations:        EncodeCitations(typed.Citations),
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.ReasoningStart:
		return BlockPart{
			Type:             "reasoning-start",
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.ReasoningDelta:
		return DeltaPart{
			Type:             "reasoning-delta",
			ID:               typed.ID,
			Delta:            typed.Delta,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.ReasoningEnd:
		return BlockPart{
			Type:             "reasoning-end",
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	default:
		return streamToolInputPart(part)
	}
}

// streamToolInputPart encodes the start, deltas and end of a streamed tool
// input.
func streamToolInputPart(part model.StreamPart) (StreamPart, bool) {
	switch typed := part.(type) {
	case model.ToolInputStart:
		return ToolInputStartPart{
			Type:             "tool-input-start",
			ID:               typed.ID,
			ToolName:         typed.ToolName,
			InputEncoding:    typed.InputEncoding,
			ProviderExecuted: typed.ProviderExecuted,
			Dynamic:          typed.Dynamic,
			Title:            typed.Title,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.ToolInputDelta:
		return DeltaPart{
			Type:             "tool-input-delta",
			ID:               typed.ID,
			Delta:            typed.Delta,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case model.ToolInputEnd:
		return BlockPart{
			Type:             "tool-input-end",
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	default:
		return nil, false
	}
}

// errorMessage returns the message of a stream error, or an empty string
// when there is none.
func errorMessage(err error) string {
	if err == nil {
		return ""
	}

	return err.Error()
}
