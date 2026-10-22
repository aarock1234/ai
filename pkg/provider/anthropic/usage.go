package anthropic

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// convertUsage converts Anthropic usage into token counts. raw is the native
// usage object reported with the result.
//
// When iterations are present because of compaction or the advisor, the
// executor iterations are summed, since the top-level counts exclude
// compaction and advisor iterations bill at a different rate. A turn served
// by a server-side fallback keeps the top-level counts, which already
// describe the served answer.
func convertUsage(native usage, raw json.RawMessage) model.Usage {
	cacheWrite := valueOrZero(native.CacheCreationInputTokens)
	cacheRead := valueOrZero(native.CacheReadInputTokens)

	inputTokens, outputTokens := executorTokens(native)

	converted := model.Usage{
		InputTokens: model.InputTokens{
			Total:      new(inputTokens + cacheWrite + cacheRead),
			NoCache:    new(inputTokens),
			CacheRead:  new(cacheRead),
			CacheWrite: new(cacheWrite),
		},
		OutputTokens: model.OutputTokens{
			Total: new(outputTokens),
		},
		Raw: raw,
	}

	if reasoningTokens, reported := thinkingTokens(native); reported {
		converted.OutputTokens.Reasoning = new(reasoningTokens)
		converted.OutputTokens.Text = new(outputTokens - reasoningTokens)
	}

	return converted
}

func thinkingTokens(native usage) (int, bool) {
	details := native.OutputTokensDetails
	if details == nil || details.ThinkingTokens == nil {
		return 0, false
	}

	return *details.ThinkingTokens, true
}

func executorTokens(native usage) (int, int) {
	if len(native.Iterations) == 0 || servedByFallback(native.Iterations) {
		return native.InputTokens, native.OutputTokens
	}

	inputTokens := 0
	outputTokens := 0
	executorIterations := 0

	for _, iteration := range native.Iterations {
		if !isExecutorIteration(iteration.Type) {
			continue
		}

		inputTokens += iteration.InputTokens
		outputTokens += iteration.OutputTokens
		executorIterations++
	}

	if executorIterations == 0 {
		return native.InputTokens, native.OutputTokens
	}

	return inputTokens, outputTokens
}

func servedByFallback(iterations []wireUsageIteration) bool {
	for _, iteration := range iterations {
		if iteration.Type == IterationFallbackMessage {
			return true
		}
	}

	return false
}

func isExecutorIteration(iterationType UsageIterationType) bool {
	return iterationType == IterationCompaction || iterationType == IterationMessage
}

// convertIterations converts the usage iterations for MessagesMetadata. It
// returns nil when the API reported none.
func convertIterations(iterations []wireUsageIteration) []UsageIteration {
	if iterations == nil {
		return nil
	}

	converted := make([]UsageIteration, 0, len(iterations))
	for _, iteration := range iterations {
		converted = append(converted, UsageIteration{
			Type:                     iteration.Type,
			Model:                    valueOrZero(iteration.Model),
			InputTokens:              iteration.InputTokens,
			OutputTokens:             iteration.OutputTokens,
			CacheCreationInputTokens: valueOrZero(iteration.CacheCreationInputTokens),
			CacheReadInputTokens:     valueOrZero(iteration.CacheReadInputTokens),
		})
	}

	return converted
}

// mapStopReason maps an Anthropic stop reason to a unified finish reason. A
// tool_use stop is a normal stop when the tool call carried JSON output.
func mapStopReason(stopReason string, jsonResponseFromTool bool) model.UnifiedFinishReason {
	switch stopReason {
	case "pause_turn", "end_turn", "stop_sequence":
		return model.FinishStop
	case "refusal":
		return model.FinishContentFilter
	case "tool_use":
		if jsonResponseFromTool {
			return model.FinishStop
		}

		return model.FinishToolCalls
	case "max_tokens", "model_context_window_exceeded":
		return model.FinishLength
	default:
		return model.FinishOther
	}
}

func finishReason(stopReason *string, jsonResponseFromTool bool) model.FinishReason {
	raw := valueOrZero(stopReason)

	return model.FinishReason{
		Unified: mapStopReason(raw, jsonResponseFromTool),
		Raw:     raw,
	}
}

func valueOrZero[T any](value *T) T {
	if value == nil {
		var zero T

		return zero
	}

	return *value
}
