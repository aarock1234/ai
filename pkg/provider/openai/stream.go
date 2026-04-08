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
		for event := range sse.Read(r) {
			if event.Data == "" {
				continue
			}

			var chunk chatStreamChunk
			if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
				// Skip malformed chunks rather than halting the stream.
				continue
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
					FinishReason: mapFinishReason(*choice.FinishReason),
				}) {
					return
				}
			}
		}
	}
}
