package openai

import (
	"encoding/json"
	"io"
	"iter"

	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// DecodeStream parses an SSE stream of OpenAI chat completion chunks
// and yields normalized model.Chunk values. The caller ranges over
// the returned iterator to consume the stream.
func DecodeStream(r io.Reader) iter.Seq[model.Chunk] {
	return func(yield func(model.Chunk) bool) {
		var metadata model.ProviderMetadata

		for event := range sse.Read(r) {
			if event.Data == "" {
				continue
			}

			if event.Data == "[DONE]" {
				return
			}

			var chunk chatStreamChunk
			if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
				// Skip malformed chunks rather than halting the stream.
				continue
			}

			metadataChanged := false
			if chunk.ID != "" && metadata.RequestID != chunk.ID {
				metadata.RequestID = chunk.ID
				metadataChanged = true
			}
			if chunk.Model != "" && metadata.ModelID != chunk.Model {
				metadata.ModelID = chunk.Model
				metadataChanged = true
			}
			if len(chunk.Metadata) > 0 {
				if metadata.Provider == nil {
					metadata.Provider = make(map[string]string, len(chunk.Metadata))
				}
				for key, value := range chunk.Metadata {
					if metadata.Provider[key] == value {
						continue
					}
					metadata.Provider[key] = value
					metadataChanged = true
				}
			}
			if metadataChanged {
				if !yield(model.ProviderMetadataChunk{Metadata: metadata.Clone()}) {
					return
				}
			}

			// Yield usage if present (typically the final chunk).
			if chunk.Usage != nil {
				if !yield(model.UsageChunk{
					Usage: model.Usage{
						InputTokens:  chunk.Usage.PromptTokens,
						OutputTokens: chunk.Usage.CompletionTokens,
					},
				}) {
					return
				}
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]

			// Yield text delta.
			if choice.Delta.Content != nil && *choice.Delta.Content != "" {
				if !yield(model.TextDelta{Text: *choice.Delta.Content}) {
					return
				}
			}

			// Yield tool call deltas.
			for _, tc := range choice.Delta.ToolCalls {
				if !yield(model.ToolCallDelta{
					ToolCallID: tc.ID,
					ToolName:   tc.Function.Name,
					ArgsJSON:   tc.Function.Arguments,
				}) {
					return
				}
			}

			// Yield finish reason.
			if choice.FinishReason != nil && *choice.FinishReason != "" {
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
