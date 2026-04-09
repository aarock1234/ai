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

// DecodeStream parses an SSE stream of Anthropic Messages API events and
// yields normalized model.Chunk values. The caller ranges over the returned
// iterator to consume the stream.
func DecodeStream(r io.Reader) iter.Seq[model.Chunk] {
	return func(yield func(model.Chunk) bool) {
		toolCalls := make(map[int]toolCallState)

		for event := range sse.Read(r) {
			if event.Data == "" {
				continue
			}

			switch event.Type {
			case "message_start":
				var msgStart streamMessageStartEvent
				if err := json.Unmarshal([]byte(event.Data), &msgStart); err != nil {
					continue
				}

				if !yield(model.ProviderMetadataChunk{Metadata: responseMetadata(&msgStart.Message)}) {
					return
				}

				if msgStart.Message.Usage.InputTokens != 0 || msgStart.Message.Usage.OutputTokens != 0 {
					if !yield(model.UsageChunk{Usage: toUsage(msgStart.Message.Usage)}) {
						return
					}
				}
			case "content_block_start":
				var blockStart streamContentBlockStartEvent
				if err := json.Unmarshal([]byte(event.Data), &blockStart); err != nil {
					continue
				}

				if blockStart.ContentBlock.Type != "tool_use" {
					continue
				}

				toolCalls[blockStart.Index] = toolCallState{
					id:   blockStart.ContentBlock.ID,
					name: blockStart.ContentBlock.Name,
				}

				argsJSON := bytes.TrimSpace(blockStart.ContentBlock.Input)
				if len(argsJSON) == 0 || bytes.Equal(argsJSON, []byte("{}")) || bytes.Equal(argsJSON, []byte("null")) {
					argsJSON = nil
				}

				if blockStart.ContentBlock.ID == "" && blockStart.ContentBlock.Name == "" && len(argsJSON) == 0 {
					continue
				}

				if !yield(model.ToolCallDelta{
					ToolCallID: blockStart.ContentBlock.ID,
					ToolName:   blockStart.ContentBlock.Name,
					ArgsJSON:   string(argsJSON),
				}) {
					return
				}
			case "content_block_delta":
				var blockDelta streamContentBlockDeltaEvent
				if err := json.Unmarshal([]byte(event.Data), &blockDelta); err != nil {
					continue
				}

				switch blockDelta.Delta.Type {
				case "text_delta":
					if blockDelta.Delta.Text == "" {
						continue
					}
					if !yield(model.TextDelta{Text: blockDelta.Delta.Text}) {
						return
					}
				case "input_json_delta":
					state, ok := toolCalls[blockDelta.Index]
					if !ok {
						continue
					}
					if !yield(model.ToolCallDelta{
						ToolCallID: state.id,
						ToolName:   state.name,
						ArgsJSON:   blockDelta.Delta.PartialJSON,
					}) {
						return
					}
				}
			case "message_delta":
				var msgDelta streamMessageDeltaEvent
				if err := json.Unmarshal([]byte(event.Data), &msgDelta); err != nil {
					continue
				}

				if msgDelta.Usage != nil {
					if !yield(model.UsageChunk{Usage: toUsage(*msgDelta.Usage)}) {
						return
					}
				}

				if msgDelta.Delta.StopReason != "" {
					if !yield(model.FinishChunk{FinishReason: mapFinishReason(msgDelta.Delta.StopReason)}) {
						return
					}
				}
			case "error":
				return
			case "content_block_stop":
				var blockStop streamContentBlockStopEvent
				if err := json.Unmarshal([]byte(event.Data), &blockStop); err != nil {
					continue
				}
				delete(toolCalls, blockStop.Index)
			case "message_stop", "ping":
			default:
				continue
			}
		}
	}
}

func toUsage(usage wireUsage) model.Usage {
	return model.Usage{
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
	}
}
