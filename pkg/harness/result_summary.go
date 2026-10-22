package harness

import (
	"github.com/aarock1234/ai/pkg/ai"
)

// summarizeSteps aggregates steps into a generation result, as GenerateText
// does: content, files, sources, tool calls, tool results, warnings and
// response messages of every step, and the text, reasoning, finish reason,
// request, response and metadata of the final step.
func summarizeSteps(steps []ai.StepResult, finalStep ai.StepResult, usage ai.LanguageModelUsage) *ai.GenerateTextResult {
	reasoningText, _ := finalStep.ReasoningText()

	result := &ai.GenerateTextResult{
		Text:             finalStep.Text(),
		Reasoning:        finalStep.Reasoning(),
		ReasoningText:    reasoningText,
		FinishReason:     finalStep.FinishReason,
		RawFinishReason:  finalStep.RawFinishReason,
		Usage:            usage,
		TotalUsage:       usage,
		Request:          finalStep.Request,
		Response:         finalStep.Response,
		ProviderMetadata: finalStep.ProviderMetadata,
		Steps:            steps,
		FinalStep:        finalStep,
	}

	for _, step := range steps {
		result.Content = append(result.Content, step.Content...)
		result.Files = append(result.Files, step.Files()...)
		result.Sources = append(result.Sources, step.Sources()...)
		result.ToolCalls = append(result.ToolCalls, step.ToolCalls()...)
		result.StaticToolCalls = append(result.StaticToolCalls, step.StaticToolCalls()...)
		result.DynamicToolCalls = append(result.DynamicToolCalls, step.DynamicToolCalls()...)
		result.ToolResults = append(result.ToolResults, step.ToolResults()...)
		result.StaticToolResults = append(result.StaticToolResults, step.StaticToolResults()...)
		result.DynamicToolResults = append(result.DynamicToolResults, step.DynamicToolResults()...)
		result.Warnings = append(result.Warnings, step.Warnings...)
		result.ResponseMessages = append(result.ResponseMessages, step.Response.Messages...)
	}

	return result
}
