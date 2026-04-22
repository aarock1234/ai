package anthropic

import (
	"bytes"
	"encoding/json"
	"io"
	"iter"

	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

type toolCallState struct {
	id   string
	name string
}

// thinkingBlockState tracks a streamed thinking block so that signature
// and text deltas carry the same block identity to downstream consumers.
type thinkingBlockState struct {
	format string
}

// DecodeStream parses an SSE stream of Anthropic Messages API events and
// yields normalized model.Chunk values. The caller ranges over the returned
// iterator to consume the stream.
func DecodeStream(r io.Reader) iter.Seq[model.Chunk] {
	return func(yield func(model.Chunk) bool) {
		toolCalls := make(map[int]toolCallState)
		thinking := make(map[int]thinkingBlockState)

		for event := range sse.Read(r) {
			if event.Data == "" {
				continue
			}

			switch event.Type {
			case "message_start":
				if !handleMessageStart(yield, event.Data) {
					return
				}
			case "content_block_start":
				if !handleContentBlockStart(yield, event.Data, toolCalls, thinking) {
					return
				}
			case "content_block_delta":
				if !handleContentBlockDelta(yield, event.Data, toolCalls, thinking) {
					return
				}
			case "message_delta":
				if !handleMessageDelta(yield, event.Data) {
					return
				}
			case "content_block_stop":
				var blockStop streamContentBlockStopEvent
				if err := json.Unmarshal([]byte(event.Data), &blockStop); err != nil {
					continue
				}
				delete(toolCalls, blockStop.Index)
				delete(thinking, blockStop.Index)
			case "error":
				return
			case "message_stop", "ping":
			default:
				continue
			}
		}
	}
}

func handleMessageStart(yield func(model.Chunk) bool, data string) bool {
	var msgStart streamMessageStartEvent
	if err := json.Unmarshal([]byte(data), &msgStart); err != nil {
		return true
	}

	if !yield(model.ProviderMetadataChunk{Metadata: responseMetadata(&msgStart.Message)}) {
		return false
	}

	if msgStart.Message.Usage.InputTokens == 0 && msgStart.Message.Usage.OutputTokens == 0 {
		return true
	}

	return yield(model.UsageChunk{Usage: toUsage(msgStart.Message.Usage)})
}

func handleContentBlockStart(yield func(model.Chunk) bool, data string, toolCalls map[int]toolCallState, thinking map[int]thinkingBlockState) bool {
	var blockStart streamContentBlockStartEvent
	if err := json.Unmarshal([]byte(data), &blockStart); err != nil {
		return true
	}

	switch blockStart.ContentBlock.Type {
	case "thinking":
		thinking[blockStart.Index] = thinkingBlockState{format: "anthropic-thinking"}
		if blockStart.ContentBlock.Thinking == "" && blockStart.ContentBlock.Signature == "" {
			return true
		}

		return yield(model.ReasoningDelta{
			Text:      blockStart.ContentBlock.Thinking,
			Signature: blockStart.ContentBlock.Signature,
			Format:    "anthropic-thinking",
		})
	case "redacted_thinking":
		thinking[blockStart.Index] = thinkingBlockState{format: "anthropic-redacted-thinking"}

		return yield(model.ReasoningDelta{
			Text:     blockStart.ContentBlock.Data,
			Format:   "anthropic-redacted-thinking",
			Redacted: true,
		})
	case "tool_use":
		toolCalls[blockStart.Index] = toolCallState{
			id:   blockStart.ContentBlock.ID,
			name: blockStart.ContentBlock.Name,
		}

		args := trimEmptyJSON(blockStart.ContentBlock.Input)
		if blockStart.ContentBlock.ID == "" && blockStart.ContentBlock.Name == "" && len(args) == 0 {
			return true
		}

		return yield(model.ToolCallDelta{
			ToolCallID: blockStart.ContentBlock.ID,
			ToolName:   blockStart.ContentBlock.Name,
			ArgsJSON:   string(args),
		})
	default:
		return true
	}
}

func handleContentBlockDelta(yield func(model.Chunk) bool, data string, toolCalls map[int]toolCallState, thinking map[int]thinkingBlockState) bool {
	var blockDelta streamContentBlockDeltaEvent
	if err := json.Unmarshal([]byte(data), &blockDelta); err != nil {
		return true
	}

	switch blockDelta.Delta.Type {
	case "text_delta":
		if blockDelta.Delta.Text == "" {
			return true
		}

		return yield(model.TextDelta{Text: blockDelta.Delta.Text})
	case "thinking_delta":
		if blockDelta.Delta.Thinking == "" {
			return true
		}
		state := thinking[blockDelta.Index]

		return yield(model.ReasoningDelta{
			Text:   blockDelta.Delta.Thinking,
			Format: state.format,
		})
	case "signature_delta":
		if blockDelta.Delta.Signature == "" {
			return true
		}
		state := thinking[blockDelta.Index]

		return yield(model.ReasoningDelta{
			Signature: blockDelta.Delta.Signature,
			Format:    state.format,
		})
	case "input_json_delta":
		state, ok := toolCalls[blockDelta.Index]
		if !ok {
			return true
		}

		return yield(model.ToolCallDelta{
			ToolCallID: state.id,
			ToolName:   state.name,
			ArgsJSON:   blockDelta.Delta.PartialJSON,
		})
	default:
		return true
	}
}

func handleMessageDelta(yield func(model.Chunk) bool, data string) bool {
	var msgDelta streamMessageDeltaEvent
	if err := json.Unmarshal([]byte(data), &msgDelta); err != nil {
		return true
	}

	if msgDelta.Usage != nil {
		if !yield(model.UsageChunk{Usage: toUsage(*msgDelta.Usage)}) {
			return false
		}
	}

	if msgDelta.Delta.StopReason == "" {
		return true
	}

	return yield(model.FinishChunk{
		FinishReason:    mapFinishReason(msgDelta.Delta.StopReason),
		RawFinishReason: msgDelta.Delta.StopReason,
	})
}

func trimEmptyJSON(raw []byte) []byte {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("{}")) || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}

	return trimmed
}

func toUsage(usage wireUsage) model.Usage {
	return model.Usage{
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
	}
}
