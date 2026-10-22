package google

import (
	"iter"

	"github.com/aarock1234/ai/pkg/model"
)

// jsonResponseToolFilter turns the streamed input of the JSON response tool
// into text, so the caller can parse it as structured output.
// Conversational text is dropped and application tool calls pass through. A
// response that holds only the JSON tool finishes with stop.
type jsonResponseToolFilter struct {
	toolName string

	responseToolIDs     map[string]bool
	responseToolsDeltas map[string]bool
	hasResponseTool     bool
	hasApplicationTool  bool
}

func convertJSONResponseToolStream(parts iter.Seq[model.StreamPart], toolName string) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		filter := &jsonResponseToolFilter{
			toolName:            toolName,
			responseToolIDs:     make(map[string]bool),
			responseToolsDeltas: make(map[string]bool),
		}

		for part := range parts {
			converted, ok := filter.convert(part)
			if !ok {
				continue
			}

			for _, convertedPart := range converted {
				if !yield(convertedPart) {
					return
				}
			}
		}
	}
}

// convert returns the parts that replace part, or false to drop it.
func (f *jsonResponseToolFilter) convert(part model.StreamPart) ([]model.StreamPart, bool) {
	switch part := part.(type) {
	case model.TextStart, model.TextDelta, model.TextEnd:
		return nil, false

	case model.ToolInputStart:
		if part.ToolName != f.toolName {
			break
		}

		f.responseToolIDs[part.ID] = true

		return []model.StreamPart{
			model.TextStart{
				ID:               part.ID,
				ProviderMetadata: part.ProviderMetadata,
			},
		}, true

	case model.ToolInputDelta:
		if !f.responseToolIDs[part.ID] {
			break
		}

		if part.Delta != "" {
			f.responseToolsDeltas[part.ID] = true
		}

		//nolint:staticcheck // tool input and text deltas are distinct contracts that happen to share fields
		textDelta := model.TextDelta{
			ID:               part.ID,
			Delta:            part.Delta,
			ProviderMetadata: part.ProviderMetadata,
		}

		return []model.StreamPart{
			textDelta,
		}, true

	case model.ToolInputEnd:
		if f.responseToolIDs[part.ID] {
			return nil, false
		}

	case model.ToolCall:
		if part.ToolName == f.toolName {
			return f.convertResponseToolCall(part), true
		}

		if !part.ProviderExecuted {
			f.hasApplicationTool = true
		}

	case model.Finish:
		onlyResponseTool := f.hasResponseTool && !f.hasApplicationTool
		if onlyResponseTool && part.FinishReason.Unified == model.FinishToolCalls {
			part.FinishReason.Unified = model.FinishStop
		}

		return []model.StreamPart{
			part,
		}, true
	}

	return []model.StreamPart{
		part,
	}, true
}

// convertResponseToolCall ends the text of the JSON response tool. Calls
// that streamed no input deltas emit their input as one text delta.
func (f *jsonResponseToolFilter) convertResponseToolCall(call model.ToolCall) []model.StreamPart {
	f.hasResponseTool = true

	var parts []model.StreamPart
	if !f.responseToolsDeltas[call.ToolCallID] {
		parts = append(parts, model.TextDelta{
			ID:               call.ToolCallID,
			Delta:            call.Input,
			ProviderMetadata: call.ProviderMetadata,
		})
	}

	return append(parts, model.TextEnd{
		ID:               call.ToolCallID,
		ProviderMetadata: call.ProviderMetadata,
	})
}
