package otel

import (
	"context"

	"go.opentelemetry.io/otel/trace"

	"github.com/aarock1234/ai/pkg/ai"
)

var _ ai.DecisionTelemetry = (*Legacy)(nil)

// OnDecideStart starts the root span of Decide.
func (l *Legacy) OnDecideStart(ctx context.Context, event ai.TelemetryEvent[ai.DecideStartEvent]) {
	start := event.Event
	maxRetries := start.MaxRetries

	settings := ai.CallSettings{
		MaxRetries: &maxRetries,
	}

	baseAttributes := legacyBaseAttributes(
		event.TelemetryMetadata,
		legacyModel{
			provider: start.Provider,
			modelID:  start.ModelID,
		},
		settings,
		start.Headers,
	)

	attributes := newSpanAttributes(event.TelemetryMetadata)
	setOperationName(attributes, start.OperationID)
	attributes.add(baseAttributes...)
	setDecisionInputs(attributes, start.State, start.Questions)

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

// OnDecideEnd records the answers and ends the root span of Decide.
func (l *Legacy) OnDecideEnd(_ context.Context, event ai.TelemetryEvent[ai.DecideEndEvent]) {
	end := event.Event

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[end.CallID]
	if state == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	setDecisionAnswers(attributes, end.Result.Answers, end.Questions)
	state.rootSpan.SetAttributes(attributes.list()...)

	state.rootSpan.End()
	delete(l.calls, end.CallID)
}

// OnDecisionModelCallStart starts the doDecide span of the decision model
// call.
func (l *Legacy) OnDecisionModelCallStart(ctx context.Context, event ai.TelemetryEvent[ai.DecisionModelCallStartEvent]) {
	call := event.Event

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[call.CallID]
	if state == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	setOperationName(attributes, call.OperationID)
	attributes.add(state.baseAttributes...)
	setDecisionInputs(attributes, call.State, call.Questions)

	parent := trace.ContextWithSpan(ctx, state.rootSpan)
	state.decisionSpan = l.startSpan(parent, call.OperationID, attributes)
}

// OnDecisionModelCallEnd records the answers, usage and provider metadata
// of the decision model call and ends the doDecide span.
func (l *Legacy) OnDecisionModelCallEnd(_ context.Context, event ai.TelemetryEvent[ai.DecisionModelCallEndEvent]) {
	end := event.Event
	result := end.Result

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[end.CallID]
	if state == nil || state.decisionSpan == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	setDecisionAnswers(attributes, result.Answers, end.Questions)
	setDecisionTokens(attributes, "ai.usage.inputTokens", "ai.usage.outputTokens", result.Usage)

	if len(result.ProviderMetadata) > 0 {
		attributes.setJSON("ai.response.providerMetadata", result.ProviderMetadata)
	}

	state.decisionSpan.SetAttributes(attributes.list()...)
	state.decisionSpan.End()
	state.decisionSpan = nil
}
