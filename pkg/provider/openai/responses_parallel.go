package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// OpenAI models can emit an internal "parallel" function call that wraps
// several calls of declared functions. The provider expands it into the
// wrapped calls and, when server-side state is used, sends their results
// back as the wrapper's single output.
const (
	responsesParallelToolName    = "parallel"
	responsesRecipientNamePrefix = "functions."
)

// responsesParallelInput is the input of a parallel wrapper call.
type responsesParallelInput struct {
	ToolUses []responsesParallelToolUse `json:"tool_uses"`
}

// responsesParallelToolUse is one wrapped call.
type responsesParallelToolUse struct {
	RecipientName string          `json:"recipient_name"`
	Parameters    json.RawMessage `json:"parameters"`
}

// isUndeclaredParallelCall reports whether toolName is the parallel
// wrapper, which applies only when no declared function has that name.
func (c responsesCallContext) isUndeclaredParallelCall(toolName string) bool {
	return toolName == responsesParallelToolName && !c.functionToolNames[responsesParallelToolName]
}

// expandParallelCall expands a parallel wrapper call. It reports false when
// the call is not a wrapper of declared function calls.
func (c responsesCallContext) expandParallelCall(call responsesFunctionCallItem) ([]model.ToolCall, bool) {
	if !c.isUndeclaredParallelCall(call.Name) {
		return nil, false
	}

	var input responsesParallelInput
	if err := json.Unmarshal([]byte(call.Arguments), &input); err != nil || len(input.ToolUses) == 0 {
		return nil, false
	}

	calls := make([]model.ToolCall, 0, len(input.ToolUses))

	for index, use := range input.ToolUses {
		toolName, ok := strings.CutPrefix(use.RecipientName, responsesRecipientNamePrefix)
		if !ok || toolName == "" || !c.functionToolNames[toolName] {
			return nil, false
		}

		parameters, ok := responsesCompactObject(use.Parameters)
		if !ok {
			return nil, false
		}

		calls = append(calls, model.ToolCall{
			ToolCallID: fmt.Sprintf("%s_%d", call.CallID, index),
			ToolName:   toolName,
			Input:      parameters,
			ProviderMetadata: c.metadata(ResponsesToolCallMetadata{
				ParallelToolCall: &ResponsesParallelToolCall{
					ItemID:     call.ID,
					ToolCallID: call.CallID,
					ToolName:   call.Name,
					Input:      call.Arguments,
					Index:      index,
					Count:      len(input.ToolUses),
				},
			}),
		})
	}

	return calls, true
}

// responsesCompactObject returns a JSON object in compact form. It reports
// false for other JSON values.
func responsesCompactObject(value json.RawMessage) (string, bool) {
	trimmed := bytes.TrimSpace(value)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return "", false
	}

	var buffer bytes.Buffer
	if err := json.Compact(&buffer, trimmed); err != nil {
		return "", false
	}

	return buffer.String(), true
}

// responsesIsParallelPosition reports whether metadata describes a position
// in a wrapper call.
func responsesIsParallelPosition(metadata *ResponsesParallelToolCall) bool {
	return metadata != nil && metadata.Index >= 0 && metadata.Count > metadata.Index
}

// responsesSameWrapper reports whether two metadata values refer to the
// same wrapper call.
func responsesSameWrapper(first, second ResponsesParallelToolCall) bool {
	return first.ItemID == second.ItemID &&
		first.ToolCallID == second.ToolCallID &&
		first.ToolName == second.ToolName &&
		first.Input == second.Input &&
		first.Count == second.Count
}

// responsesParallelResultGroup holds the results of every call expanded
// from one wrapper, in their original order.
type responsesParallelResultGroup struct {
	metadata ResponsesParallelToolCall
	results  []model.ToolResultPart
}

// responsesPendingParallelGroup collects results while scanning the prompt.
type responsesPendingParallelGroup struct {
	metadata ResponsesParallelToolCall
	results  map[int]model.ToolResultPart
	invalid  bool
}

// responsesCompleteParallelGroups finds wrappers whose expanded calls all
// have exactly one result.
func responsesCompleteParallelGroups(prompt []model.Message, namespace string) (map[string]responsesParallelResultGroup, error) {
	pending := make(map[string]*responsesPendingParallelGroup)

	for _, message := range prompt {
		toolMessage, ok := message.(model.ToolMessage)
		if !ok {
			continue
		}

		for _, part := range toolMessage.Content {
			result, ok := part.(model.ToolResultPart)
			if !ok {
				continue
			}

			options, err := responsesProviderValue[ResponsesToolResultOptions](result.ProviderOptions, namespace)
			if err != nil {
				return nil, err
			}

			metadata := options.ParallelToolCall
			if !responsesIsParallelPosition(metadata) {
				continue
			}

			responsesAddParallelResult(pending, *metadata, result)
		}
	}

	complete := make(map[string]responsesParallelResultGroup)

	for toolCallID, group := range pending {
		if group.invalid || len(group.results) != group.metadata.Count {
			continue
		}

		results := make([]model.ToolResultPart, 0, group.metadata.Count)
		for index := range group.metadata.Count {
			result, ok := group.results[index]
			if !ok {
				break
			}

			results = append(results, result)
		}

		if len(results) == group.metadata.Count {
			complete[toolCallID] = responsesParallelResultGroup{
				metadata: group.metadata,
				results:  results,
			}
		}
	}

	return complete, nil
}

// responsesAddParallelResult adds a result to its wrapper's group, marking
// the group invalid on conflicting metadata or duplicate positions.
func responsesAddParallelResult(
	pending map[string]*responsesPendingParallelGroup,
	metadata ResponsesParallelToolCall,
	result model.ToolResultPart,
) {
	group, exists := pending[metadata.ToolCallID]
	if !exists {
		pending[metadata.ToolCallID] = &responsesPendingParallelGroup{
			metadata: metadata,
			results: map[int]model.ToolResultPart{
				metadata.Index: result,
			},
		}

		return
	}

	_, duplicate := group.results[metadata.Index]
	if duplicate || !responsesSameWrapper(group.metadata, metadata) {
		group.invalid = true

		return
	}

	group.results[metadata.Index] = result
}

// parallelCall sends the wrapper call once in place of its expanded calls.
// It reports whether the call belonged to a complete wrapper.
func (c *responsesInputConverter) parallelCall(metadata *ResponsesParallelToolCall) bool {
	group, ok := c.parallelGroup(metadata)
	if !ok {
		return false
	}

	wrapperID := group.metadata.ToolCallID
	if c.emittedParallelCalls[wrapperID] {
		return true
	}

	c.emittedParallelCalls[wrapperID] = true

	// Conversations already contain the wrapper item, while previous
	// response chains need the client function call rebuilt in full.
	if !c.options.hasConversation {
		c.add(responsesFunctionCallInput{
			Type:      responsesFunctionCall,
			CallID:    wrapperID,
			Name:      group.metadata.ToolName,
			Arguments: group.metadata.Input,
		})
	}

	return true
}

// parallelResult sends the results of a complete wrapper as one output, in
// the order of the original calls. It reports whether the result belonged
// to a complete wrapper.
func (c *responsesInputConverter) parallelResult(metadata *ResponsesParallelToolCall) (bool, error) {
	group, ok := c.parallelGroup(metadata)
	if !ok {
		return false, nil
	}

	wrapperID := group.metadata.ToolCallID
	if c.emittedParallelResults[wrapperID] {
		return true, nil
	}

	c.emittedParallelResults[wrapperID] = true

	output, err := c.parallelOutput(group)
	if err != nil {
		return true, err
	}

	c.add(responsesFunctionCallOutputInput{
		Type:   responsesFunctionCallOutput,
		CallID: wrapperID,
		Output: output,
	})

	return true, nil
}

// parallelOutput joins the serialized results with newlines, as parts when
// any result carries a cache breakpoint.
func (c *responsesInputConverter) parallelOutput(group responsesParallelResultGroup) (responsesContent, error) {
	texts := make([]string, 0, len(group.results))
	breakpoints := make([]*ResponsesPromptCacheBreakpoint, 0, len(group.results))
	hasBreakpoint := false

	for _, result := range group.results {
		breakpoint, err := c.scalarBreakpoint(result.Output, result.ProviderOptions)
		if err != nil {
			return responsesContent{}, err
		}

		text, err := c.serializedResult(result)
		if err != nil {
			return responsesContent{}, err
		}

		texts = append(texts, text)
		breakpoints = append(breakpoints, breakpoint)
		hasBreakpoint = hasBreakpoint || breakpoint != nil
	}

	if !hasBreakpoint {
		return responsesContent{
			Text: strings.Join(texts, "\n"),
		}, nil
	}

	parts := make([]responsesContentPart, 0, len(texts))
	for index, text := range texts {
		if index > 0 {
			text = "\n" + text
		}

		parts = append(parts, responsesInputText{
			Type:                  "input_text",
			Text:                  text,
			PromptCacheBreakpoint: breakpoints[index],
		})
	}

	return responsesContent{
		Parts: parts,
	}, nil
}

// serializedResult returns one wrapped result as text. Multipart results
// are encoded as their JSON parts.
func (c *responsesInputConverter) serializedResult(result model.ToolResultPart) (string, error) {
	content, ok := result.Output.(model.ContentOutput)
	if !ok {
		quoteText := c.options.outputSchemaToolNames[result.ToolName]

		return responsesScalarOutput(result.Output, quoteText)
	}

	parts, err := c.toolContentParts(content.Value, true, "tool")
	if err != nil {
		return "", err
	}

	return string(responsesEncodeJSON(parts)), nil
}

// parallelGroup returns the complete group that metadata belongs to.
func (c *responsesInputConverter) parallelGroup(metadata *ResponsesParallelToolCall) (responsesParallelResultGroup, bool) {
	if !responsesIsParallelPosition(metadata) {
		return responsesParallelResultGroup{}, false
	}

	group, ok := c.parallelGroups[metadata.ToolCallID]
	if !ok || !responsesSameWrapper(group.metadata, *metadata) {
		return responsesParallelResultGroup{}, false
	}

	return group, true
}
