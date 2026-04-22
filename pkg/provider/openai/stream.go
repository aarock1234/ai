package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"strings"

	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// toolCallState tracks the ID and name for a streamed tool call so that
// subsequent deltas (which carry only an index) can emit the full identity.
type toolCallState struct {
	id   string
	name string
}

// audioStreamState accumulates streamed audio chunks so a single
// FileChunk can be emitted when the stream ends. Each event's base64
// payload is decoded independently (OpenAI sends self-contained
// base64 units, not a single stream split at arbitrary byte offsets).
type audioStreamState struct {
	id         string
	format     string
	data       []byte
	transcript strings.Builder
}

// DecodeStream parses an SSE stream of OpenAI chat completion chunks
// and yields normalized model.Chunk values. The caller ranges over
// the returned iterator to consume the stream.
func DecodeStream(r io.Reader) iter.Seq[model.Chunk] {
	return func(yield func(model.Chunk) bool) {
		var metadata model.ProviderMetadata
		toolCalls := make(map[int]toolCallState)
		var audio audioStreamState

		for event := range sse.Read(r) {
			if event.Data == "" {
				continue
			}

			if event.Data == "[DONE]" {
				if file, ok := finalizeAudioState(&audio); ok {
					if !yield(file) {
						return
					}
				}

				return
			}

			var wire chatStreamChunk
			if err := json.Unmarshal([]byte(event.Data), &wire); err != nil {
				// Skip malformed chunks rather than halting the stream.
				continue
			}

			if !yieldMetadataChanges(yield, &metadata, &wire) {
				return
			}

			// Yield usage if present (typically the final chunk).
			if wire.Usage != nil {
				if !yield(model.UsageChunk{Usage: decodeUsage(wire.Usage)}) {
					return
				}
			}

			if len(wire.Choices) == 0 {
				continue
			}

			choice := wire.Choices[0]

			if !yieldReasoningDeltas(yield, choice.Delta) {
				return
			}

			if choice.Delta.Content != nil && *choice.Delta.Content != "" {
				if !yield(model.TextDelta{Text: *choice.Delta.Content}) {
					return
				}
			}

			if !yieldFileDeltas(yield, choice.Delta.Images) {
				return
			}

			accumulateAudio(&audio, choice.Delta.Audio)

			if !yieldSourceDeltas(yield, choice.Delta.Annotations) {
				return
			}

			if !yieldToolCallDeltas(yield, toolCalls, choice.Delta.ToolCalls) {
				return
			}

			if choice.Delta.Refusal != "" {
				if !yield(model.ErrorChunk{
					Err: fmt.Errorf("refusal: %s", choice.Delta.Refusal),
				}) {
					return
				}
			}

			if choice.FinishReason != nil && *choice.FinishReason != "" {
				if file, ok := finalizeAudioState(&audio); ok {
					if !yield(file) {
						return
					}
				}

				if !yield(model.FinishChunk{
					FinishReason:    mapFinishReason(*choice.FinishReason),
					RawFinishReason: *choice.FinishReason,
				}) {
					return
				}
			}
		}
	}
}

func yieldMetadataChanges(yield func(model.Chunk) bool, metadata *model.ProviderMetadata, wire *chatStreamChunk) bool {
	changed := false

	if wire.ID != "" && metadata.RequestID != wire.ID {
		metadata.RequestID = wire.ID
		changed = true
	}
	if wire.Model != "" && metadata.ModelID != wire.Model {
		metadata.ModelID = wire.Model
		changed = true
	}
	if len(wire.Metadata) > 0 {
		if metadata.Provider == nil {
			metadata.Provider = make(map[string]string, len(wire.Metadata))
		}
		for key, value := range wire.Metadata {
			if metadata.Provider[key] == value {
				continue
			}
			metadata.Provider[key] = value
			changed = true
		}
	}

	if !changed {
		return true
	}

	return yield(model.ProviderMetadataChunk{Metadata: metadata.Clone()})
}

func yieldReasoningDeltas(yield func(model.Chunk) bool, delta chatDelta) bool {
	// Prefer the structured reasoning_details array when present; the
	// flat reasoning/reasoning_content strings are a compatibility
	// shim providers emit alongside it.
	if len(delta.ReasoningDetails) > 0 {
		for _, part := range delta.ReasoningDetails {
			rd, ok := reasoningDeltaFromPart(part)
			if !ok {
				continue
			}
			if !yield(rd) {
				return false
			}
		}

		return true
	}

	if delta.Reasoning != "" {
		if !yield(model.ReasoningDelta{Text: delta.Reasoning}) {
			return false
		}
	}

	if delta.ReasoningContent != "" {
		if !yield(model.ReasoningDelta{Text: delta.ReasoningContent}) {
			return false
		}
	}

	return true
}

func reasoningDeltaFromPart(part wireReasoningPart) (model.ReasoningDelta, bool) {
	delta := model.ReasoningDelta{
		ID:        part.ID,
		Signature: part.Signature,
		Format:    part.Format,
	}

	switch part.Type {
	case "reasoning.encrypted":
		delta.Redacted = true
	case "reasoning.summary":
		delta.Text = part.Summary
	default:
		delta.Text = part.Text
	}

	if delta.Text == "" && !delta.Redacted && delta.Signature == "" && delta.ID == "" {
		return model.ReasoningDelta{}, false
	}

	return delta, true
}

func yieldFileDeltas(yield func(model.Chunk) bool, images []wireMessageImage) bool {
	for _, image := range images {
		chunk, ok := fileChunkFromImage(image)
		if !ok {
			continue
		}
		if !yield(chunk) {
			return false
		}
	}

	return true
}

func fileChunkFromImage(image wireMessageImage) (model.FileChunk, bool) {
	if image.ImageURL == nil {
		return model.FileChunk{}, false
	}

	file, ok := fileContentFromDataURL(image.ImageURL.URL)
	if !ok {
		return model.FileChunk{}, false
	}

	return model.FileChunk(file), true
}

func yieldSourceDeltas(yield func(model.Chunk) bool, annotations []wireAnnotation) bool {
	for _, annotation := range annotations {
		chunk, ok := sourceChunkFromAnnotation(annotation)
		if !ok {
			continue
		}
		if !yield(chunk) {
			return false
		}
	}

	return true
}

func sourceChunkFromAnnotation(a wireAnnotation) (model.SourceChunk, bool) {
	source, ok := sourceContentFromAnnotation(a)
	if !ok {
		return model.SourceChunk{}, false
	}

	return model.SourceChunk(source), true
}

// yieldToolCallDeltas emits ToolCallDelta chunks, patching ID and name
// into every delta because OpenAI only sends them on the first chunk
// for each tool call.
func yieldToolCallDeltas(yield func(model.Chunk) bool, toolCalls map[int]toolCallState, deltas []wireToolCall) bool {
	for _, tc := range deltas {
		if tc.ID != "" {
			toolCalls[tc.Index] = toolCallState{
				id:   tc.ID,
				name: tc.Function.Name,
			}
		}

		state := toolCalls[tc.Index]
		if !yield(model.ToolCallDelta{
			ToolCallID: state.id,
			ToolName:   state.name,
			ArgsJSON:   tc.Function.Arguments,
		}) {
			return false
		}
	}

	return true
}

func accumulateAudio(state *audioStreamState, delta *wireMessageAudio) {
	if delta == nil {
		return
	}
	if delta.ID != "" {
		state.id = delta.ID
	}
	if delta.Format != "" {
		state.format = delta.Format
	}
	if delta.Data != "" {
		if decoded, err := base64.StdEncoding.DecodeString(delta.Data); err == nil {
			state.data = append(state.data, decoded...)
		}
	}
	if delta.Transcript != "" {
		state.transcript.WriteString(delta.Transcript)
	}
}

func finalizeAudioState(state *audioStreamState) (model.FileChunk, bool) {
	if len(state.data) == 0 {
		*state = audioStreamState{}

		return model.FileChunk{}, false
	}

	chunk := model.FileChunk{
		Data:     state.data,
		MIMEType: mimeTypeForAudioFormat(state.format),
	}
	if state.id != "" {
		chunk.Metadata.Provider = map[string]string{
			"audio_id": state.id,
		}
	}

	*state = audioStreamState{}

	return chunk, true
}
