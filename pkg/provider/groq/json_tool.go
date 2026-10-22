package groq

import (
	"fmt"
	"iter"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// jsonToolBaseName is the name of the JSON response tool before suffixes
// that avoid collisions with the call's tools.
const jsonToolBaseName = "json"

// jsonResponseTool returns a function tool that carries the JSON output, for
// calls that combine a JSON schema response format with function tools,
// which Groq rejects. It returns nil for other calls.
func jsonResponseTool(format *model.ResponseFormat, tools []model.Tool) *model.FunctionTool {
	if !hasSchema(format) || !hasFunctionTool(tools) {
		return nil
	}

	name := jsonToolBaseName
	for suffix := 1; hasToolNamed(tools, name); suffix++ {
		name = fmt.Sprintf("%s_%d", jsonToolBaseName, suffix)
	}

	description := format.Description
	if description == "" {
		description = jsonToolDescription
	}

	return &model.FunctionTool{
		Name:        name,
		Description: description,
		InputSchema: format.Schema,
	}
}

// jsonToolChoice keeps an explicit tool choice, forces the JSON tool instead
// of no tool, and otherwise requires a tool call.
func jsonToolChoice(choice *model.ToolChoice, jsonToolName string) *model.ToolChoice {
	if choice != nil && choice.Type == model.ToolChoiceTool {
		return choice
	}

	if choice != nil && choice.Type == model.ToolChoiceNone {
		return &model.ToolChoice{
			Type:     model.ToolChoiceTool,
			ToolName: jsonToolName,
		}
	}

	return &model.ToolChoice{
		Type: model.ToolChoiceRequired,
	}
}

func hasFunctionTool(tools []model.Tool) bool {
	return slices.ContainsFunc(tools, func(definition model.Tool) bool {
		_, ok := definition.(model.FunctionTool)

		return ok
	})
}

func hasToolNamed(tools []model.Tool, name string) bool {
	return slices.ContainsFunc(tools, func(definition model.Tool) bool {
		return definition.ToolName() == name
	})
}

// jsonToolStream converts the streamed input of the JSON response tool into
// text, so that the output can be parsed as structured output. Other text is
// dropped and application tool calls pass through. A stream that called only
// the JSON tool finishes with a stop reason.
type jsonToolStream struct {
	toolName string

	responseToolIDs        map[string]bool
	responseToolsWithDelta map[string]bool
	hasResponseTool        bool
	hasApplicationTool     bool
}

// jsonToolParts applies a jsonToolStream to parts.
func jsonToolParts(parts iter.Seq[model.StreamPart], toolName string) iter.Seq[model.StreamPart] {
	converter := &jsonToolStream{
		toolName:               toolName,
		responseToolIDs:        make(map[string]bool),
		responseToolsWithDelta: make(map[string]bool),
	}

	return func(yield func(model.StreamPart) bool) {
		for part := range parts {
			for _, converted := range converter.convert(part) {
				if !yield(converted) {
					return
				}
			}
		}
	}
}

// convert returns the parts to emit for part.
func (s *jsonToolStream) convert(part model.StreamPart) []model.StreamPart {
	switch typed := part.(type) {
	case model.TextStart, model.TextDelta, model.TextEnd:
		return nil

	case model.ToolInputStart:
		if typed.ToolName == s.toolName {
			s.responseToolIDs[typed.ID] = true

			return []model.StreamPart{
				model.TextStart{
					ID:               typed.ID,
					ProviderMetadata: typed.ProviderMetadata,
				},
			}
		}

	case model.ToolInputDelta:
		if s.responseToolIDs[typed.ID] {
			if typed.Delta != "" {
				s.responseToolsWithDelta[typed.ID] = true
			}

			return []model.StreamPart{
				model.TextDelta(typed),
			}
		}

	case model.ToolInputEnd:
		if s.responseToolIDs[typed.ID] {
			return nil
		}

	case model.ToolCall:
		if typed.ToolName == s.toolName {
			return s.responseToolCall(typed)
		}

		if !typed.ProviderExecuted {
			s.hasApplicationTool = true
		}

	case model.Finish:
		if s.hasResponseTool && !s.hasApplicationTool && typed.FinishReason.Unified == model.FinishToolCalls {
			typed.FinishReason.Unified = model.FinishStop

			return []model.StreamPart{
				typed,
			}
		}
	}

	return []model.StreamPart{
		part,
	}
}

// responseToolCall closes the text of the JSON tool, sending the whole input
// when no delta carried it.
func (s *jsonToolStream) responseToolCall(call model.ToolCall) []model.StreamPart {
	s.hasResponseTool = true

	var parts []model.StreamPart
	if !s.responseToolsWithDelta[call.ToolCallID] {
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
