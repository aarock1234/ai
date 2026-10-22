package harness

import (
	"encoding/json"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// Reserved tool names of the synthetic tool calls that project harness
// events without an ai stream part equivalent.
const (
	fileChangeToolName = "fileChange"
	compactionToolName = "compaction"
)

// toolCallLookup returns the parsed tool call with an ID, reporting false
// when the call arrived in an earlier slice of the turn.
type toolCallLookup func(toolCallID string) (ai.ToolCall, bool)

// translateStreamPart translates one harness event into the ai stream parts
// consumers see. Tool calls and approval requests are validated and emitted
// by the turn driver, and step boundaries are emitted by the result, so they
// translate to nothing here. File changes and compactions have no ai
// equivalent and become synthetic dynamic, provider-executed tool call and
// result pairs named fileChange and compaction.
func translateStreamPart(part StreamPart, lookup toolCallLookup) []ai.StreamPart {
	switch typed := part.(type) {
	case TextStartPart:
		return single(ai.TextStartPart{
			ID:               typed.ID,
			ProviderMetadata: providerMetadata(typed.HarnessMetadata),
		})
	case TextDeltaPart:
		return single(ai.TextDeltaPart{
			ID:               typed.ID,
			Text:             typed.Delta,
			ProviderMetadata: providerMetadata(typed.HarnessMetadata),
		})
	case TextEndPart:
		return single(ai.TextEndPart{
			ID:               typed.ID,
			ProviderMetadata: providerMetadata(typed.HarnessMetadata),
		})
	case ReasoningStartPart:
		return single(ai.ReasoningStartPart{
			ID:               typed.ID,
			ProviderMetadata: providerMetadata(typed.HarnessMetadata),
		})
	case ReasoningDeltaPart:
		return single(ai.ReasoningDeltaPart{
			ID:               typed.ID,
			Text:             typed.Delta,
			ProviderMetadata: providerMetadata(typed.HarnessMetadata),
		})
	case ReasoningEndPart:
		return single(ai.ReasoningEndPart{
			ID:               typed.ID,
			ProviderMetadata: providerMetadata(typed.HarnessMetadata),
		})
	case ToolInputStartPart:
		return single(ai.ToolInputStartPart{
			ID:               typed.ID,
			ToolName:         typed.ToolName,
			InputEncoding:    typed.InputEncoding,
			ProviderExecuted: typed.ProviderExecuted,
			Dynamic:          typed.Dynamic,
			Title:            typed.Title,
			ProviderMetadata: typed.ProviderMetadata,
		})
	case ToolInputDeltaPart:
		return single(ai.ToolInputDeltaPart{
			ID:               typed.ID,
			Delta:            typed.Delta,
			ProviderMetadata: typed.ProviderMetadata,
		})
	case ToolInputEndPart:
		return single(ai.ToolInputEndPart{
			ID:               typed.ID,
			ProviderMetadata: typed.ProviderMetadata,
		})
	case ToolResultPart:
		return single(translateToolResult(typed, lookup))
	case FileChangePart:
		return translateFileChange(typed)
	case CompactionPart:
		return translateCompaction(typed)
	case ErrorPart:
		return single(ai.ErrorPart{
			Err: typed.Err,
		})
	case RawPart:
		return single(ai.RawPart{
			Value: typed.Value,
		})
	default:
		return nil
	}
}

// translateToolResult translates a tool result. Failed results of
// provider-executed tools become tool errors carrying the runtime's JSON;
// whether a tool was provider-executed comes from its originating call, and
// calls from earlier slices count as provider-executed.
func translateToolResult(part ToolResultPart, lookup toolCallLookup) ai.StreamPart {
	call, found := lookup(part.ToolCallID)
	providerExecuted := !found || call.ProviderExecuted

	if part.IsError && providerExecuted {
		return ai.ToolError{
			ToolCallID: part.ToolCallID,
			ToolName:   part.ToolName,
			Input:      call.Input,
			Error: &ai.ProviderToolError{
				Result: part.Result,
			},
			ProviderExecuted: true,
			Dynamic:          part.Dynamic,
			ProviderMetadata: part.ProviderMetadata,
		}
	}

	return ai.ToolResult{
		ToolCallID:       part.ToolCallID,
		ToolName:         part.ToolName,
		Input:            call.Input,
		Output:           part.Result,
		ProviderExecuted: found && call.ProviderExecuted,
		Dynamic:          part.Dynamic,
		Preliminary:      part.Preliminary,
		ProviderMetadata: part.ProviderMetadata,
	}
}

// fileChangePayload is the input and output of a fileChange tool call.
type fileChangePayload struct {
	Event FileChangeEvent `json:"event"`
	Path  string          `json:"path"`
}

// compactionPayload is the output of a compaction tool call.
type compactionPayload struct {
	Trigger      CompactionTrigger `json:"trigger"`
	Summary      string            `json:"summary"`
	TokensBefore *int              `json:"tokensBefore,omitempty"`
	TokensAfter  *int              `json:"tokensAfter,omitempty"`
}

// translateFileChange projects a file change as a synthetic tool call and
// result.
func translateFileChange(part FileChangePart) []ai.StreamPart {
	payload := mustMarshal(fileChangePayload{
		Event: part.Event,
		Path:  part.Path,
	})

	return syntheticToolCall(syntheticToolCallInput{
		toolCallID: id.New("harness-file-change"),
		toolName:   fileChangeToolName,
		input:      payload,
		output:     payload,
		metadata:   part.HarnessMetadata,
	})
}

// translateCompaction projects a compaction as a synthetic tool call and
// result. The call takes no input; the details are its output.
func translateCompaction(part CompactionPart) []ai.StreamPart {
	output := mustMarshal(compactionPayload{
		Trigger:      part.Trigger,
		Summary:      part.Summary,
		TokensBefore: part.TokensBefore,
		TokensAfter:  part.TokensAfter,
	})

	return syntheticToolCall(syntheticToolCallInput{
		toolCallID: id.New("harness-compaction"),
		toolName:   compactionToolName,
		input:      json.RawMessage(`{}`),
		output:     output,
		metadata:   part.HarnessMetadata,
	})
}

// syntheticToolCallInput describes a synthetic tool call and result.
type syntheticToolCallInput struct {
	toolCallID string
	toolName   string
	input      json.RawMessage
	output     json.RawMessage
	metadata   Metadata
}

// syntheticToolCall returns a dynamic, provider-executed tool call and its
// result.
func syntheticToolCall(input syntheticToolCallInput) []ai.StreamPart {
	return []ai.StreamPart{
		ai.ToolCall{
			ToolCallID:       input.toolCallID,
			ToolName:         input.toolName,
			Input:            input.input,
			ProviderExecuted: true,
			Dynamic:          true,
			ProviderMetadata: providerMetadata(input.metadata),
		},
		ai.ToolResult{
			ToolCallID:       input.toolCallID,
			ToolName:         input.toolName,
			Input:            input.input,
			Output:           input.output,
			ProviderExecuted: true,
			Dynamic:          true,
			ProviderMetadata: providerMetadata(input.metadata),
		},
	}
}

// providerMetadata rebinds harness metadata as provider metadata for ai
// consumers.
func providerMetadata(metadata Metadata) model.ProviderMetadata {
	if metadata == nil {
		return nil
	}

	return model.ProviderMetadata(metadata)
}

// single returns a one-part slice.
func single(part ai.StreamPart) []ai.StreamPart {
	return []ai.StreamPart{
		part,
	}
}

// mustMarshal encodes a value whose types always encode.
func mustMarshal(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic("encode harness payload: " + err.Error())
	}

	return encoded
}
