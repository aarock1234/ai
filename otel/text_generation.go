package otel

import (
	"context"
	"fmt"
	"net/http"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// OnGenerateTextStart starts the invoke_agent span of GenerateText or StreamText.
func (t *OpenTelemetry) OnGenerateTextStart(ctx context.Context, event ai.TelemetryEvent[ai.GenerateTextStartEvent]) {
	start := event.Event
	settings := start.CallSettings
	operation := operationName(start.OperationID)
	baseAttributes := t.baseAttributes(event.TelemetryMetadata, settings.Headers)

	attributes := newSpanAttributes(event.TelemetryMetadata)
	attributes.setString("gen_ai.operation.name", operation)
	attributes.setString("gen_ai.provider.name", providerName(start.Provider))
	attributes.setString("gen_ai.request.model", start.ModelID)
	attributes.setString("gen_ai.agent.name", event.FunctionID)
	setRequestSettings(attributes, settings)
	attributes.setInt("gen_ai.request.seed", settings.Seed)
	setSystemInstructions(attributes, start.System)
	attributes.setInput("gen_ai.input.messages", func() attribute.Value {
		return jsonValue(formatMessages(start.Messages))
	})
	attributes.add(baseAttributes...)

	t.mu.Lock()
	defer t.mu.Unlock()

	rootSpan := t.startSpan(ctx, spanStart{
		name:       operation + " " + start.ModelID,
		kind:       trace.SpanKindInternal,
		spanType:   SpanTypeOperation,
		callID:     start.CallID,
		operation:  start.OperationID,
		attributes: attributes.list(),
	})

	t.calls[start.CallID] = newCallState(callStart{
		operationID:    start.OperationID,
		metadata:       event.TelemetryMetadata,
		provider:       start.Provider,
		modelID:        start.ModelID,
		settings:       settings,
		baseAttributes: baseAttributes,
		rootSpan:       rootSpan,
	})
}

// OnGenerateTextStepStart starts the agent_step span of a step.
func (t *OpenTelemetry) OnGenerateTextStepStart(ctx context.Context, event ai.TelemetryEvent[ai.GenerateTextStepStartEvent]) {
	step := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[step.CallID]
	if state == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	attributes.setString("gen_ai.operation.name", "agent_step")
	attributes.add(state.baseAttributes...)

	if t.options.ToolChoice && step.StepToolChoice != nil {
		choice := wirejson.EncodeToolChoice(*step.StepToolChoice)

		attributes.setInput("ai.prompt.toolChoice", func() attribute.Value {
			return jsonValue(choice)
		})
	}

	state.stepSpan = t.startSpan(trace.ContextWithSpan(ctx, state.rootSpan), spanStart{
		name:       fmt.Sprintf("step %d", len(step.Steps)+1),
		kind:       trace.SpanKindInternal,
		spanType:   SpanTypeStep,
		callID:     step.CallID,
		operation:  state.operationID,
		attributes: attributes.list(),
	})
}

// OnLanguageModelCallStart starts the chat span of a model call.
func (t *OpenTelemetry) OnLanguageModelCallStart(ctx context.Context, event ai.TelemetryEvent[ai.LanguageModelCallStartEvent]) {
	call := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[call.CallID]
	if state == nil || state.stepSpan == nil {
		return
	}

	definitions := wirejson.EncodeTools(call.Tools)
	state.inferenceTools = definitions

	attributes := newSpanAttributes(state.metadata)
	attributes.setString("gen_ai.operation.name", "chat")
	attributes.setString("gen_ai.provider.name", providerName(call.Provider))
	attributes.setString("gen_ai.request.model", call.ModelID)
	setRequestSettings(attributes, state.settings)
	setSystemInstructions(attributes, call.System)
	setInputMessages(attributes, call.Messages)

	if len(definitions) > 0 {
		attributes.setInput("gen_ai.tool.definitions", func() attribute.Value {
			return jsonValue(definitions)
		})
	}

	state.inferenceSpan = t.startSpan(trace.ContextWithSpan(ctx, state.stepSpan), spanStart{
		name:       "chat " + call.ModelID,
		kind:       trace.SpanKindClient,
		spanType:   SpanTypeLanguageModel,
		callID:     call.CallID,
		operation:  state.operationID,
		attributes: attributes.list(),
	})
}

// OnLanguageModelCallEnd records the response on the chat span, records a
// tool span for every provider-executed tool call, and ends the chat span.
func (t *OpenTelemetry) OnLanguageModelCallEnd(ctx context.Context, event ai.TelemetryEvent[ai.LanguageModelCallEndEvent]) {
	end := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[end.CallID]
	if state == nil || state.inferenceSpan == nil {
		return
	}

	span := state.inferenceSpan
	markErrorFinish(span, end.FinishReason)

	definitions, observedProviderTools := withProviderExecutedTools(state.inferenceTools, end.Content)

	attributes := newSpanAttributes(state.metadata)
	setPerformance(attributes, end.Performance)
	attributes.setString("gen_ai.provider.name", providerName(end.Provider))
	attributes.setStrings("gen_ai.response.finish_reasons", []string{
		string(end.FinishReason),
	})
	attributes.setString("gen_ai.response.id", end.ResponseID)
	attributes.setString("gen_ai.response.model", end.ModelID)
	setTokenUsage(attributes, end.Usage)
	attributes.setOutput("gen_ai.output.messages", func() attribute.Value {
		return jsonValue(formatOutputMessages(responseContent(end.Content, end.FinishReason)))
	})

	if observedProviderTools {
		attributes.setInput("gen_ai.tool.definitions", func() attribute.Value {
			return jsonValue(definitions)
		})
	}

	t.setResponseDetails(attributes, end.ProviderMetadata, end.Usage)
	span.SetAttributes(attributes.list()...)

	t.recordProviderExecutedTools(ctx, state, end)

	span.End()
	state.inferenceSpan = nil
	state.inferenceTools = nil
}

// withProviderExecutedTools adds an extension tool definition for every
// provider-executed tool call whose tool was not declared. It reports
// whether it added one.
func withProviderExecutedTools(declared []wirejson.ToolDefinition, content []ai.ContentPart) ([]wirejson.ToolDefinition, bool) {
	definitions := append([]wirejson.ToolDefinition(nil), declared...)

	names := make(map[string]bool, len(definitions))
	for _, definition := range definitions {
		names[definition.Name] = true
	}

	added := false

	for _, part := range content {
		call, ok := part.(ai.ToolCall)
		if !ok || !call.ProviderExecuted || names[call.ToolName] {
			continue
		}

		definitions = append(definitions, wirejson.ToolDefinition{
			Type: "extension",
			Name: call.ToolName,
		})
		names[call.ToolName] = true
		added = true
	}

	return definitions, added
}

// recordProviderExecutedTools records an execute_tool span below the chat
// span for every provider-executed tool call of a response, with its final
// result or error.
func (t *OpenTelemetry) recordProviderExecutedTools(ctx context.Context, state *callState, end ai.LanguageModelCallEndEvent) {
	outputs := finalToolOutputs(end.Content)
	recorded := make(map[string]bool)
	parent := trace.ContextWithSpan(ctx, state.inferenceSpan)

	for _, part := range end.Content {
		call, ok := part.(ai.ToolCall)
		if !ok || !call.ProviderExecuted || recorded[call.ToolCallID] {
			continue
		}

		recorded[call.ToolCallID] = true

		toolSpan := t.startToolSpan(parent, state, toolSpanStart{
			callID:   end.CallID,
			call:     call,
			toolType: "extension",
		})

		recordToolOutput(toolSpan, state.metadata, outputs[call.ToolCallID])
		toolSpan.End()
	}
}

// finalToolOutputs returns the last non-preliminary result or error of each
// tool call, keyed by tool call ID.
func finalToolOutputs(content []ai.ContentPart) map[string]ai.ToolOutput {
	outputs := make(map[string]ai.ToolOutput)

	for _, part := range content {
		switch typed := part.(type) {
		case ai.ToolResult:
			if !typed.Preliminary {
				outputs[typed.ToolCallID] = typed
			}

		case ai.ToolError:
			outputs[typed.ToolCallID] = typed
		}
	}

	return outputs
}

// OnToolExecutionStart starts the execute_tool span of a client-side tool
// call below the step span.
func (t *OpenTelemetry) OnToolExecutionStart(ctx context.Context, event ai.TelemetryEvent[ai.ToolExecutionStartEvent]) {
	start := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[start.CallID]
	if state == nil {
		return
	}

	toolType := "function"
	if start.ToolCall.ProviderExecuted {
		toolType = "extension"
	}

	state.toolSpans[start.ToolCall.ToolCallID] = t.startToolSpan(
		trace.ContextWithSpan(ctx, state.parentSpan()),
		state,
		toolSpanStart{
			callID:   start.CallID,
			call:     start.ToolCall,
			toolType: toolType,
		},
	)
}

// toolSpanStart describes an execute_tool span.
type toolSpanStart struct {
	callID   string
	call     ai.ToolCall
	toolType string
}

func (t *OpenTelemetry) startToolSpan(parent context.Context, state *callState, start toolSpanStart) trace.Span {
	attributes := newSpanAttributes(state.metadata)
	attributes.setString("gen_ai.operation.name", "execute_tool")
	attributes.setString("gen_ai.tool.name", start.call.ToolName)
	attributes.setString("gen_ai.tool.call.id", start.call.ToolCallID)
	attributes.setString("gen_ai.tool.type", start.toolType)
	attributes.setInput("gen_ai.tool.call.arguments", func() attribute.Value {
		return jsonValue(wirejson.RawValue(start.call.Input))
	})

	return t.startSpan(parent, spanStart{
		name:       "execute_tool " + start.call.ToolName,
		kind:       trace.SpanKindInternal,
		spanType:   SpanTypeTool,
		callID:     start.callID,
		operation:  state.operationID,
		attributes: attributes.list(),
	})
}

// OnToolExecutionEnd records the result or error of a tool call and ends
// its span.
func (t *OpenTelemetry) OnToolExecutionEnd(_ context.Context, event ai.TelemetryEvent[ai.ToolExecutionEndEvent]) {
	end := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[end.CallID]
	if state == nil {
		return
	}

	span := state.toolSpans[end.ToolCall.ToolCallID]
	if span == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	attributes.setSeconds("gen_ai.execute_tool.duration", end.ExecutionTime)
	span.SetAttributes(attributes.list()...)

	recordToolOutput(span, state.metadata, end.Output)

	span.End()
	delete(state.toolSpans, end.ToolCall.ToolCallID)
}

// recordToolOutput records the result of a tool call on its span, or its
// error. A nil output records nothing.
func recordToolOutput(span trace.Span, metadata ai.TelemetryMetadata, output ai.ToolOutput) {
	switch typed := output.(type) {
	case ai.ToolResult:
		attributes := newSpanAttributes(metadata)
		attributes.setOutput("gen_ai.tool.call.result", func() attribute.Value {
			return jsonValue(typed.Output)
		})
		span.SetAttributes(attributes.list()...)

	case ai.ToolError:
		recordError(span, typed.Error)
	}
}

// OnGenerateTextStepEnd ends the step span.
func (t *OpenTelemetry) OnGenerateTextStepEnd(_ context.Context, event ai.TelemetryEvent[ai.StepResult]) {
	step := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[step.CallID]
	if state == nil || state.stepSpan == nil {
		return
	}

	markErrorFinish(state.stepSpan, step.FinishReason)

	attributes := newSpanAttributes(state.metadata)
	t.setResponseDetails(attributes, step.ProviderMetadata, step.Usage)
	state.stepSpan.SetAttributes(attributes.list()...)

	state.stepSpan.End()
	state.stepSpan = nil
}

// OnGenerateTextEnd records the outcome of the generation and ends its span.
func (t *OpenTelemetry) OnGenerateTextEnd(_ context.Context, event ai.TelemetryEvent[ai.GenerateTextEndEvent]) {
	finish := event.Event
	finalStep := finish.FinalStep

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[finish.CallID]
	if state == nil {
		return
	}

	markErrorFinish(state.rootSpan, finish.FinishReason)

	attributes := newSpanAttributes(state.metadata)
	attributes.setStrings("gen_ai.response.finish_reasons", []string{
		string(finish.FinishReason),
	})
	setTokenUsage(attributes, finish.Usage)
	attributes.setOutput("gen_ai.output.messages", func() attribute.Value {
		content := stepContent(finalStep)
		content.text = finish.Text
		content.finishReason = finish.FinishReason

		return jsonValue(formatOutputMessages(content))
	})
	t.setResponseDetails(attributes, finalStep.ProviderMetadata, finish.Usage)
	state.rootSpan.SetAttributes(attributes.list()...)

	state.rootSpan.End()
	delete(t.calls, finish.CallID)
}

// OnGenerateTextAbort ends every open span of an aborted stream.
func (t *OpenTelemetry) OnGenerateTextAbort(_ context.Context, event ai.TelemetryEvent[ai.StreamTextAbortEvent]) {
	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[event.Event.CallID]
	if state == nil {
		return
	}

	state.endOpenSpans(nil, recordError)
	delete(t.calls, event.Event.CallID)
}

// OnError records the error that ended an operation on every open span of
// the call and ends them.
func (t *OpenTelemetry) OnError(_ context.Context, event ai.TelemetryEvent[ai.TelemetryErrorEvent]) {
	failure := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[failure.CallID]
	if state == nil {
		return
	}

	state.endOpenSpans(failure.Error, recordError)
	delete(t.calls, failure.CallID)
}

// baseAttributes returns the supplemental attributes added to every span
// of a call.
func (t *OpenTelemetry) baseAttributes(metadata ai.TelemetryMetadata, headers http.Header) []attribute.KeyValue {
	if !t.options.Headers {
		return nil
	}

	attributes := newSpanAttributes(metadata)
	attributes.add(headerAttributes(headers)...)

	return attributes.list()
}

// setResponseDetails records the supplemental provider metadata and usage
// details of a response.
func (t *OpenTelemetry) setResponseDetails(
	attributes *spanAttributes,
	metadata model.ProviderMetadata,
	usage ai.LanguageModelUsage,
) {
	if t.options.ProviderMetadata && len(metadata) > 0 {
		attributes.setJSON("ai.response.providerMetadata", metadata)
	}

	if t.options.Usage {
		setDetailedUsage(attributes, usage)
	}
}

// setRequestSettings records the generation settings of a request.
func setRequestSettings(attributes *spanAttributes, settings ai.CallSettings) {
	attributes.setFloat("gen_ai.request.frequency_penalty", settings.FrequencyPenalty)
	attributes.setInt("gen_ai.request.max_tokens", settings.MaxOutputTokens)
	attributes.setFloat("gen_ai.request.presence_penalty", settings.PresencePenalty)
	attributes.setStrings("gen_ai.request.stop_sequences", settings.StopSequences)
	attributes.setFloat("gen_ai.request.temperature", settings.Temperature)
	attributes.setInt("gen_ai.request.top_k", settings.TopK)
	attributes.setFloat("gen_ai.request.top_p", settings.TopP)
}

// setSystemInstructions records a system prompt as an input.
func setSystemInstructions(attributes *spanAttributes, system string) {
	if system == "" {
		return
	}

	attributes.setInput("gen_ai.system_instructions", func() attribute.Value {
		return jsonValue(systemInstructions(system))
	})
}

// setInputMessages records the prompt messages of a model call as an
// input, skipping an empty prompt.
func setInputMessages(attributes *spanAttributes, messages []model.Message) {
	if len(messages) == 0 {
		return
	}

	attributes.setInput("gen_ai.input.messages", func() attribute.Value {
		return jsonValue(formatMessages(messages))
	})
}

// setPerformance records the client operation timings of a model call.
func setPerformance(attributes *spanAttributes, performance ai.LanguageModelCallPerformance) {
	attributes.setSeconds("gen_ai.client.operation.duration", performance.ResponseTime)

	if performance.TimeToFirstOutput != nil {
		attributes.setSeconds("gen_ai.client.operation.time_to_first_chunk", *performance.TimeToFirstOutput)
	}

	if performance.TimeBetweenOutputChunks != nil {
		attributes.setSeconds("gen_ai.client.operation.time_per_output_chunk", performance.TimeBetweenOutputChunks.Avg)
	}
}
