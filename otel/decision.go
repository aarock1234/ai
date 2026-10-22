package otel

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/aarock1234/ai/internal/decisionwire"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// decideOperationName is the gen_ai.operation.name of decision spans.
const decideOperationName = "decide"

var _ ai.DecisionTelemetry = (*OpenTelemetry)(nil)

// OnDecideStart starts the operation span of Decide.
func (t *OpenTelemetry) OnDecideStart(ctx context.Context, event ai.TelemetryEvent[ai.DecideStartEvent]) {
	start := event.Event
	baseAttributes := t.baseAttributes(event.TelemetryMetadata, start.Headers)

	attributes := newSpanAttributes(event.TelemetryMetadata)
	attributes.setString("gen_ai.operation.name", decideOperationName)
	attributes.setString("gen_ai.provider.name", providerName(start.Provider))
	attributes.setString("gen_ai.request.model", start.ModelID)
	attributes.add(baseAttributes...)

	if t.options.ExperimentalDecision {
		setDecisionInputs(attributes, start.State, start.Questions)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	rootSpan := t.startSpan(ctx, spanStart{
		name:       decideOperationName + " " + start.ModelID,
		kind:       trace.SpanKindClient,
		spanType:   SpanTypeOperation,
		callID:     start.CallID,
		operation:  start.OperationID,
		attributes: attributes.list(),
	})

	maxRetries := start.MaxRetries

	t.calls[start.CallID] = newCallState(callStart{
		operationID: start.OperationID,
		metadata:    event.TelemetryMetadata,
		provider:    start.Provider,
		modelID:     start.ModelID,
		settings: ai.CallSettings{
			MaxRetries: &maxRetries,
		},
		baseAttributes: baseAttributes,
		rootSpan:       rootSpan,
	})
}

// OnDecideEnd records the answers and ends the operation span of Decide.
func (t *OpenTelemetry) OnDecideEnd(_ context.Context, event ai.TelemetryEvent[ai.DecideEndEvent]) {
	end := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[end.CallID]
	if state == nil {
		return
	}

	if t.options.ExperimentalDecision {
		attributes := newSpanAttributes(state.metadata)
		setDecisionAnswers(attributes, end.Result.Answers, end.Questions)
		state.rootSpan.SetAttributes(attributes.list()...)
	}

	state.rootSpan.End()
	delete(t.calls, end.CallID)
}

// OnDecisionModelCallStart starts the span of the decision model call.
func (t *OpenTelemetry) OnDecisionModelCallStart(ctx context.Context, event ai.TelemetryEvent[ai.DecisionModelCallStartEvent]) {
	call := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[call.CallID]
	if state == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	attributes.setString("gen_ai.operation.name", decideOperationName)
	attributes.setString("gen_ai.provider.name", providerName(state.provider))
	attributes.setString("gen_ai.request.model", state.modelID)
	attributes.add(state.baseAttributes...)

	if t.options.ExperimentalDecision {
		setDecisionInputs(attributes, call.State, call.Questions)
	}

	state.decisionSpan = t.startSpan(trace.ContextWithSpan(ctx, state.rootSpan), spanStart{
		name:       decideOperationName + " " + state.modelID,
		kind:       trace.SpanKindClient,
		spanType:   SpanTypeDecision,
		callID:     call.CallID,
		operation:  call.OperationID,
		attributes: attributes.list(),
	})
}

// OnDecisionModelCallEnd records the response, usage and answers of the
// decision model call and ends its span.
func (t *OpenTelemetry) OnDecisionModelCallEnd(_ context.Context, event ai.TelemetryEvent[ai.DecisionModelCallEndEvent]) {
	end := event.Event
	result := end.Result

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[end.CallID]
	if state == nil || state.decisionSpan == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	attributes.setString("gen_ai.response.id", result.Response.ID)
	attributes.setString("gen_ai.response.model", result.Response.ModelID)
	setDecisionTokens(attributes, "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", result.Usage)

	if t.options.ExperimentalDecision {
		setDecisionAnswers(attributes, result.Answers, end.Questions)
	}

	if t.options.ProviderMetadata && len(result.ProviderMetadata) > 0 {
		attributes.setJSON("ai.response.providerMetadata", result.ProviderMetadata)
	}

	state.decisionSpan.SetAttributes(attributes.list()...)
	state.decisionSpan.End()
	state.decisionSpan = nil
}

// setDecisionInputs records the decision state, with file contents
// flattened as in prompts, and the questions.
func setDecisionInputs(attributes *spanAttributes, state []model.DecisionStatePart, questions []model.DecisionQuestion) {
	attributes.setInput("ai.decision.state", func() attribute.Value {
		return jsonValue(telemetryJSON.DecisionState(state))
	})
	attributes.setInput("ai.decision.questions", func() attribute.Value {
		encoded, err := decisionwire.EncodeQuestions(questions)
		if err != nil {
			return attribute.Value{}
		}

		return jsonValue(encoded)
	})
}

// setDecisionAnswers records the answers keyed by question ID.
func setDecisionAnswers(
	attributes *spanAttributes,
	answers map[string]model.DecisionAnswer,
	questions []model.DecisionQuestion,
) {
	attributes.setOutput("ai.decision.answers", func() attribute.Value {
		encoded, err := wirejson.EncodeDecisionAnswers(answers, questions)
		if err != nil {
			return attribute.Value{}
		}

		return jsonValue(encoded)
	})
}

// setDecisionTokens records the token counts of a decision usage, if any.
func setDecisionTokens(attributes *spanAttributes, inputKey, outputKey string, usage *model.DecisionUsage) {
	if usage == nil {
		return
	}

	attributes.setInt(inputKey, usage.InputTokens)
	attributes.setInt(outputKey, usage.OutputTokens)
}
