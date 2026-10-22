package ui

import (
	"encoding/json"
	"fmt"
	"strings"
)

// dataChunkFields are the encoded fields of a DataChunk besides its type.
type dataChunkFields struct {
	ID        string          `json:"id,omitempty"`
	Data      json.RawMessage `json:"data,omitempty"`
	Transient bool            `json:"transient,omitempty"`
}

// MarshalJSON encodes the chunk with the type "text-start".
func (c TextStartChunk) MarshalJSON() ([]byte, error) {
	type fields TextStartChunk

	return marshalWithType("text-start", fields(c))
}

// MarshalJSON encodes the chunk with the type "text-delta".
func (c TextDeltaChunk) MarshalJSON() ([]byte, error) {
	type fields TextDeltaChunk

	return marshalWithType("text-delta", fields(c))
}

// MarshalJSON encodes the chunk with the type "text-end".
func (c TextEndChunk) MarshalJSON() ([]byte, error) {
	type fields TextEndChunk

	return marshalWithType("text-end", fields(c))
}

// MarshalJSON encodes the chunk with the type "reasoning-start".
func (c ReasoningStartChunk) MarshalJSON() ([]byte, error) {
	type fields ReasoningStartChunk

	return marshalWithType("reasoning-start", fields(c))
}

// MarshalJSON encodes the chunk with the type "reasoning-delta".
func (c ReasoningDeltaChunk) MarshalJSON() ([]byte, error) {
	type fields ReasoningDeltaChunk

	return marshalWithType("reasoning-delta", fields(c))
}

// MarshalJSON encodes the chunk with the type "reasoning-end".
func (c ReasoningEndChunk) MarshalJSON() ([]byte, error) {
	type fields ReasoningEndChunk

	return marshalWithType("reasoning-end", fields(c))
}

// MarshalJSON encodes the chunk with the type "custom".
func (c CustomChunk) MarshalJSON() ([]byte, error) {
	type fields CustomChunk

	return marshalWithType("custom", fields(c))
}

// MarshalJSON encodes the chunk with the type "error".
func (c ErrorChunk) MarshalJSON() ([]byte, error) {
	type fields ErrorChunk

	return marshalWithType("error", fields(c))
}

// MarshalJSON encodes the chunk with the type "tool-input-start".
func (c ToolInputStartChunk) MarshalJSON() ([]byte, error) {
	type fields ToolInputStartChunk

	return marshalWithType("tool-input-start", fields(c))
}

// MarshalJSON encodes the chunk with the type "tool-input-delta".
func (c ToolInputDeltaChunk) MarshalJSON() ([]byte, error) {
	type fields ToolInputDeltaChunk

	return marshalWithType("tool-input-delta", fields(c))
}

// MarshalJSON encodes the chunk with the type "tool-input-available".
func (c ToolInputAvailableChunk) MarshalJSON() ([]byte, error) {
	type fields ToolInputAvailableChunk

	return marshalWithType("tool-input-available", fields(c))
}

// MarshalJSON encodes the chunk with the type "tool-input-error".
func (c ToolInputErrorChunk) MarshalJSON() ([]byte, error) {
	type fields ToolInputErrorChunk

	return marshalWithType("tool-input-error", fields(c))
}

// MarshalJSON encodes the chunk with the type "tool-approval-request".
func (c ToolApprovalRequestChunk) MarshalJSON() ([]byte, error) {
	type fields ToolApprovalRequestChunk

	return marshalWithType("tool-approval-request", fields(c))
}

// MarshalJSON encodes the chunk with the type "tool-approval-response".
func (c ToolApprovalResponseChunk) MarshalJSON() ([]byte, error) {
	type fields ToolApprovalResponseChunk

	return marshalWithType("tool-approval-response", fields(c))
}

// MarshalJSON encodes the chunk with the type "tool-output-available".
func (c ToolOutputAvailableChunk) MarshalJSON() ([]byte, error) {
	type fields ToolOutputAvailableChunk

	return marshalWithType("tool-output-available", fields(c))
}

// MarshalJSON encodes the chunk with the type "tool-output-error".
func (c ToolOutputErrorChunk) MarshalJSON() ([]byte, error) {
	type fields ToolOutputErrorChunk

	return marshalWithType("tool-output-error", fields(c))
}

// MarshalJSON encodes the chunk with the type "tool-output-denied".
func (c ToolOutputDeniedChunk) MarshalJSON() ([]byte, error) {
	type fields ToolOutputDeniedChunk

	return marshalWithType("tool-output-denied", fields(c))
}

// MarshalJSON encodes the chunk with the type "source-url".
func (c SourceURLChunk) MarshalJSON() ([]byte, error) {
	type fields SourceURLChunk

	return marshalWithType("source-url", fields(c))
}

// MarshalJSON encodes the chunk with the type "source-document".
func (c SourceDocumentChunk) MarshalJSON() ([]byte, error) {
	type fields SourceDocumentChunk

	return marshalWithType("source-document", fields(c))
}

// MarshalJSON encodes the chunk with the type "file".
func (c FileChunk) MarshalJSON() ([]byte, error) {
	type fields FileChunk

	return marshalWithType("file", fields(c))
}

// MarshalJSON encodes the chunk with the type "reasoning-file".
func (c ReasoningFileChunk) MarshalJSON() ([]byte, error) {
	type fields ReasoningFileChunk

	return marshalWithType("reasoning-file", fields(c))
}

// MarshalJSON encodes the chunk with the type "data-" + Name.
func (c DataChunk) MarshalJSON() ([]byte, error) {
	return marshalWithType(dataTypePrefix+c.Name, dataChunkFields{
		ID:        c.ID,
		Data:      c.Data,
		Transient: c.Transient,
	})
}

// MarshalJSON encodes the chunk with the type "start-step".
func (StartStepChunk) MarshalJSON() ([]byte, error) {
	return marshalWithType("start-step", struct{}{})
}

// MarshalJSON encodes the chunk with the type "finish-step".
func (FinishStepChunk) MarshalJSON() ([]byte, error) {
	return marshalWithType("finish-step", struct{}{})
}

// MarshalJSON encodes the chunk with the type "reset-step".
func (ResetStepChunk) MarshalJSON() ([]byte, error) {
	return marshalWithType("reset-step", struct{}{})
}

// MarshalJSON encodes the chunk with the type "start".
func (c StartChunk) MarshalJSON() ([]byte, error) {
	type fields StartChunk

	return marshalWithType("start", fields(c))
}

// MarshalJSON encodes the chunk with the type "finish".
func (c FinishChunk) MarshalJSON() ([]byte, error) {
	type fields FinishChunk

	return marshalWithType("finish", fields(c))
}

// MarshalJSON encodes the chunk with the type "abort".
func (c AbortChunk) MarshalJSON() ([]byte, error) {
	type fields AbortChunk

	return marshalWithType("abort", fields(c))
}

// MarshalJSON encodes the chunk with the type "message-metadata".
func (c MessageMetadataChunk) MarshalJSON() ([]byte, error) {
	type fields MessageMetadataChunk

	return marshalWithType("message-metadata", fields(c))
}

// chunkDecoders decode the chunk types with a fixed type discriminator.
var chunkDecoders = map[string]func(data []byte) (Chunk, error){
	"text-start":             decodeChunk[TextStartChunk],
	"text-delta":             decodeChunk[TextDeltaChunk],
	"text-end":               decodeChunk[TextEndChunk],
	"reasoning-start":        decodeChunk[ReasoningStartChunk],
	"reasoning-delta":        decodeChunk[ReasoningDeltaChunk],
	"reasoning-end":          decodeChunk[ReasoningEndChunk],
	"custom":                 decodeChunk[CustomChunk],
	"error":                  decodeChunk[ErrorChunk],
	"tool-input-start":       decodeChunk[ToolInputStartChunk],
	"tool-input-delta":       decodeChunk[ToolInputDeltaChunk],
	"tool-input-available":   decodeChunk[ToolInputAvailableChunk],
	"tool-input-error":       decodeChunk[ToolInputErrorChunk],
	"tool-approval-request":  decodeChunk[ToolApprovalRequestChunk],
	"tool-approval-response": decodeChunk[ToolApprovalResponseChunk],
	"tool-output-available":  decodeChunk[ToolOutputAvailableChunk],
	"tool-output-error":      decodeChunk[ToolOutputErrorChunk],
	"tool-output-denied":     decodeChunk[ToolOutputDeniedChunk],
	"source-url":             decodeChunk[SourceURLChunk],
	"source-document":        decodeChunk[SourceDocumentChunk],
	"file":                   decodeChunk[FileChunk],
	"reasoning-file":         decodeChunk[ReasoningFileChunk],
	"start-step":             decodeChunk[StartStepChunk],
	"finish-step":            decodeChunk[FinishStepChunk],
	"reset-step":             decodeChunk[ResetStepChunk],
	"start":                  decodeChunk[StartChunk],
	"finish":                 decodeChunk[FinishChunk],
	"abort":                  decodeChunk[AbortChunk],
	"message-metadata":       decodeChunk[MessageMetadataChunk],
}

// UnmarshalChunk decodes one chunk, selected by its type. It rejects unknown
// types.
func UnmarshalChunk(data []byte) (Chunk, error) {
	chunkType, err := decodeType(data)
	if err != nil {
		return nil, fmt.Errorf("decode chunk type: %w", err)
	}

	if name, isData := strings.CutPrefix(chunkType, dataTypePrefix); isData {
		return decodeDataChunk(data, name)
	}

	decode, known := chunkDecoders[chunkType]
	if !known {
		return nil, fmt.Errorf("unknown chunk type %q", chunkType)
	}

	chunk, err := decode(data)
	if err != nil {
		return nil, fmt.Errorf("decode %s chunk: %w", chunkType, err)
	}

	return chunk, nil
}

func decodeChunk[Variant Chunk](data []byte) (Chunk, error) {
	var chunk Variant
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil, err
	}

	return chunk, nil
}

func decodeDataChunk(data []byte, name string) (Chunk, error) {
	var fields dataChunkFields
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, fmt.Errorf("decode data chunk: %w", err)
	}

	return DataChunk{
		Name:      name,
		ID:        fields.ID,
		Data:      fields.Data,
		Transient: fields.Transient,
	}, nil
}
