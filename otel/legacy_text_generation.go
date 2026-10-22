package otel

import (
	"context"
	"encoding/json"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// legacyPrompt is the ai.prompt attribute of a generation.
type legacyPrompt struct {
	System   string             `json:"system,omitempty"`
	Messages []wirejson.Message `json:"messages"`
}

// legacyToolCall is one entry of the ai.response.toolCalls attribute.
type legacyToolCall struct {
	ToolCallID string          `json:"toolCallId"`
	ToolName   string          `json:"toolName"`
	Input      json.RawMessage `json:"input"`
}

// legacyFile is one entry of the ai.response.files attribute.
type legacyFile struct {
	Type      string `json:"type"`
	MediaType string `json:"mediaType"`
	Data      string `json:"data"`
}

// OnGenerateTextStart starts the root span of GenerateText or StreamText.
func (l *Legacy) OnGenerateTextStart(ctx context.Context, event ai.TelemetryEvent[ai.GenerateTextStartEvent]) {
	start := event.Event

	settings := start.CallSettings
	maxRetries := maxRetriesOrDefault(settings.MaxRetries)
	settings.MaxRetries = &maxRetries

	baseAttributes := legacyBaseAttributes(
		event.TelemetryMetadata,
		legacyModel{
			provider: start.Provider,
			modelID:  start.ModelID,
		},
		settings,
		settings.Headers,
	)

	attributes := newSpanAttributes(event.TelemetryMetadata)
	setOperationName(attributes, start.OperationID)
	attributes.add(baseAttributes...)
	attributes.setInput("ai.prompt", func() attribute.Value {
		return jsonValue(legacyPrompt{
			System:   start.System,
			Messages: telemetryJSON.Messages(start.Messages),
		})
	})

	l.mu.Lock()
	defer l.mu.Unlock()

	l.calls[start.CallID] = newCallState(callStart{
		operationID:    start.OperationID,
		metadata:       event.TelemetryMetadata,
		provider:       start.Provider,
		modelID:        start.ModelID,
		settings:       settings,
		baseAttributes: baseAttributes,
		rootSpan:       l.startSpan(ctx, start.OperationID, attributes),
	})
}

// OnGenerateTextStepStart starts the doGenerate or doStream span of a step.
func (l *Legacy) OnGenerateTextStepStart(ctx context.Context, event ai.TelemetryEvent[ai.GenerateTextStepStartEvent]) {
	step := event.Event

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[step.CallID]
	if state == nil {
		return
	}

	stepOperationID := operationGenerateText + ".doGenerate"
	if state.operationID == operationStreamText {
		stepOperationID = operationStreamText + ".doStream"
	}

	settings := state.settings

	attributes := newSpanAttributes(state.metadata)
	setOperationName(attributes, stepOperationID)
	attributes.add(state.baseAttributes...)
	attributes.setString("ai.model.provider", step.Provider)
	attributes.setString("ai.model.id", step.ModelID)
	setLegacyStepPrompt(attributes, step)
	attributes.setString("gen_ai.system", step.Provider)
	attributes.setString("gen_ai.request.model", step.ModelID)
	setRequestSettings(attributes, settings)

	state.stepSpan = l.startSpan(trace.ContextWithSpan(ctx, state.rootSpan), stepOperationID, attributes)
}

// setLegacyStepPrompt records the prompt messages, tools and tool choice
// sent to the model in a step.
func setLegacyStepPrompt(attributes *spanAttributes, step ai.GenerateTextStepStartEvent) {
	if step.PromptMessages != nil {
		attributes.setInput("ai.prompt.messages", func() attribute.Value {
			return jsonValue(telemetryJSON.Messages(step.PromptMessages))
		})
	}

	if step.StepTools != nil {
		attributes.setInput("ai.prompt.tools", func() attribute.Value {
			return jsonListValue(wirejson.EncodeTools(step.StepTools))
		})
	}

	if step.StepToolChoice != nil {
		choice := wirejson.EncodeToolChoice(*step.StepToolChoice)

		attributes.setInput("ai.prompt.toolChoice", func() attribute.Value {
			return jsonValue(choice)
		})
	}
}

// OnToolExecutionStart starts the ai.toolCall span of a tool call below the
// step span.
func (l *Legacy) OnToolExecutionStart(ctx context.Context, event ai.TelemetryEvent[ai.ToolExecutionStartEvent]) {
	start := event.Event
	call := start.ToolCall

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[start.CallID]
	if state == nil || state.stepSpan == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	setOperationName(attributes, "ai.toolCall")
	attributes.setString("ai.toolCall.name", call.ToolName)
	attributes.setString("ai.toolCall.id", call.ToolCallID)

	// Upstream records the arguments as an output.
	attributes.setOutput("ai.toolCall.args", func() attribute.Value {
		return jsonValue(wirejson.RawValue(call.Input))
	})

	parent := trace.ContextWithSpan(ctx, state.stepSpan)
	state.toolSpans[call.ToolCallID] = l.startSpan(parent, "ai.toolCall", attributes)
}

// OnToolExecutionEnd records the result or error of a tool call and ends
// its span.
func (l *Legacy) OnToolExecutionEnd(_ context.Context, event ai.TelemetryEvent[ai.ToolExecutionEndEvent]) {
	end := event.Event

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[end.CallID]
	if state == nil {
		return
	}

	span := state.toolSpans[end.ToolCall.ToolCallID]
	if span == nil {
		return
	}

	switch output := end.Output.(type) {
	case ai.ToolResult:
		attributes := newSpanAttributes(state.metadata)
		attributes.setOutput("ai.toolCall.result", func() attribute.Value {
			return jsonValue(output.Output)
		})
		span.SetAttributes(attributes.list()...)

	case ai.ToolError:
		recordLegacyError(span, output.Error)
	}

	span.End()
	delete(state.toolSpans, end.ToolCall.ToolCallID)
}

// OnGenerateTextStepEnd records the response of a step and ends its span. Streamed
// steps also record their timing as span events.
func (l *Legacy) OnGenerateTextStepEnd(_ context.Context, event ai.TelemetryEvent[ai.StepResult]) {
	step := event.Event

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[step.CallID]
	if state == nil || state.stepSpan == nil {
		return
	}

	span := state.stepSpan
	isStreamText := state.operationID == operationStreamText

	markErrorFinish(span, step.FinishReason)

	attributes := newSpanAttributes(state.metadata)
	attributes.setString("ai.response.finishReason", string(step.FinishReason))
	setLegacyResponseContent(attributes, step, step.Text())
	setLegacyResponseInfo(attributes, step.Response.ResponseInfo)
	setLegacyProviderMetadata(attributes, step.ProviderMetadata)

	if isStreamText {
		setLegacyStreamTiming(attributes, step.Performance.LanguageModelCallPerformance)
	}

	setLegacyUsage(attributes, step.Usage)
	attributes.setStrings("gen_ai.response.finish_reasons", []string{
		string(step.FinishReason),
	})
	attributes.setString("gen_ai.response.id", step.Response.ID)
	attributes.setString("gen_ai.response.model", step.Response.ModelID)
	attributes.setInt("gen_ai.usage.input_tokens", step.Usage.InputTokens)
	attributes.setInt("gen_ai.usage.output_tokens", step.Usage.OutputTokens)
	span.SetAttributes(attributes.list()...)

	if isStreamText {
		addLegacyStreamEvents(span, step.Performance.LanguageModelCallPerformance)
	}

	span.End()
	state.stepSpan = nil
}

// setLegacyStreamTiming records the timing of a streamed step.
func setLegacyStreamTiming(attributes *spanAttributes, performance ai.LanguageModelCallPerformance) {
	if performance.TimeToFirstOutput != nil {
		attributes.add(attribute.Float64("ai.response.msToFirstChunk", milliseconds(*performance.TimeToFirstOutput)))
	}

	attributes.add(
		attribute.Float64("ai.response.msToFinish", milliseconds(performance.ResponseTime)),
		attribute.Float64("ai.response.avgOutputTokensPerSecond", performance.EffectiveOutputTokensPerSecond),
	)
}

// addLegacyStreamEvents adds the first chunk and finish events of a
// streamed step.
func addLegacyStreamEvents(span trace.Span, performance ai.LanguageModelCallPerformance) {
	if performance.TimeToFirstOutput != nil {
		span.AddEvent("ai.stream.firstChunk", trace.WithAttributes(
			attribute.Float64("ai.response.msToFirstChunk", milliseconds(*performance.TimeToFirstOutput)),
		))
	}

	finish := sanitizeAttributes([]attribute.KeyValue{
		attribute.Float64("ai.response.msToFinish", milliseconds(performance.ResponseTime)),
		attribute.Float64("ai.response.avgOutputTokensPerSecond", performance.EffectiveOutputTokensPerSecond),
	})

	span.AddEvent("ai.stream.finish", trace.WithAttributes(finish...))
}

// OnGenerateTextEnd records the outcome of the generation and ends its span.
func (l *Legacy) OnGenerateTextEnd(_ context.Context, event ai.TelemetryEvent[ai.GenerateTextEndEvent]) {
	finish := event.Event

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[finish.CallID]
	if state == nil {
		return
	}

	markErrorFinish(state.rootSpan, finish.FinishReason)

	attributes := newSpanAttributes(state.metadata)
	attributes.setString("ai.response.finishReason", string(finish.FinishReason))
	setLegacyResponseContent(attributes, finish.FinalStep, finish.Text)
	setLegacyProviderMetadata(attributes, finish.FinalStep.ProviderMetadata)
	setLegacyUsage(attributes, finish.Usage)
	state.rootSpan.SetAttributes(attributes.list()...)

	state.rootSpan.End()
	delete(l.calls, finish.CallID)
}

// setLegacyResponseContent records the text, reasoning, tool calls and
// files of a step.
func setLegacyResponseContent(attributes *spanAttributes, step ai.StepResult, text string) {
	attributes.setOutput("ai.response.text", func() attribute.Value {
		return attribute.StringValue(text)
	})

	if reasoning := step.Reasoning(); len(reasoning) > 0 {
		attributes.setOutput("ai.response.reasoning", func() attribute.Value {
			return attribute.StringValue(joinReasoningText(reasoning))
		})
	}

	if calls := step.ToolCalls(); len(calls) > 0 {
		attributes.setOutput("ai.response.toolCalls", func() attribute.Value {
			return jsonValue(legacyToolCalls(calls))
		})
	}

	if files := step.Files(); len(files) > 0 {
		attributes.setOutput("ai.response.files", func() attribute.Value {
			return jsonValue(legacyFiles(files))
		})
	}
}

// joinReasoningText joins the text of the reasoning parts with newlines.
func joinReasoningText(reasoning []ai.ContentPart) string {
	texts := make([]string, 0, len(reasoning))

	for _, part := range reasoning {
		if typed, ok := part.(ai.ReasoningContent); ok {
			texts = append(texts, typed.Text)
		}
	}

	return strings.Join(texts, "\n")
}

func legacyToolCalls(calls []ai.ToolCall) []legacyToolCall {
	converted := make([]legacyToolCall, 0, len(calls))

	for _, call := range calls {
		converted = append(converted, legacyToolCall{
			ToolCallID: call.ToolCallID,
			ToolName:   call.ToolName,
			Input:      wirejson.RawValue(call.Input),
		})
	}

	return converted
}

func legacyFiles(files []ai.GeneratedFile) []legacyFile {
	converted := make([]legacyFile, 0, len(files))

	for _, file := range files {
		converted = append(converted, legacyFile{
			Type:      "file",
			MediaType: file.MediaType,
			Data:      file.Base64(),
		})
	}

	return converted
}

// setLegacyResponseInfo records the response ID, model and timestamp.
func setLegacyResponseInfo(attributes *spanAttributes, response model.ResponseInfo) {
	attributes.setString("ai.response.id", response.ID)
	attributes.setString("ai.response.model", response.ModelID)

	if !response.Timestamp.IsZero() {
		attributes.setString("ai.response.timestamp", response.Timestamp.UTC().Format(isoTimestamp))
	}
}

// isoTimestamp formats timestamps like JavaScript's Date.toISOString.
const isoTimestamp = "2006-01-02T15:04:05.000Z"

// setLegacyProviderMetadata records provider metadata as JSON.
func setLegacyProviderMetadata(attributes *spanAttributes, metadata model.ProviderMetadata) {
	if len(metadata) > 0 {
		attributes.setJSON("ai.response.providerMetadata", metadata)
	}
}

// setLegacyUsage records the token usage of a step or a generation.
func setLegacyUsage(attributes *spanAttributes, usage ai.LanguageModelUsage) {
	attributes.setInt("ai.usage.inputTokens", usage.InputTokens)
	attributes.setInt("ai.usage.outputTokens", usage.OutputTokens)
	attributes.setInt("ai.usage.totalTokens", usage.TotalTokens)
	attributes.setInt("ai.usage.reasoningTokens", usage.OutputTokenDetails.ReasoningTokens)
	attributes.setInt("ai.usage.cachedInputTokens", usage.InputTokenDetails.CacheReadTokens)
	attributes.setInt("ai.usage.inputTokenDetails.noCacheTokens", usage.InputTokenDetails.NoCacheTokens)
	attributes.setInt("ai.usage.inputTokenDetails.cacheReadTokens", usage.InputTokenDetails.CacheReadTokens)
	attributes.setInt("ai.usage.inputTokenDetails.cacheWriteTokens", usage.InputTokenDetails.CacheWriteTokens)
	attributes.setInt("ai.usage.outputTokenDetails.textTokens", usage.OutputTokenDetails.TextTokens)
	attributes.setInt("ai.usage.outputTokenDetails.reasoningTokens", usage.OutputTokenDetails.ReasoningTokens)
}
