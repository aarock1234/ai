package ai

import (
	"context"
	"net/http"

	"github.com/aarock1234/ai/pkg/model"
)

// DecideStartEvent describes a Decide call before the model is called.
type DecideStartEvent struct {
	CallID string

	// OperationID is "ai.decide".
	OperationID string

	Provider string
	ModelID  string

	// State is the state as given, before files were resolved.
	State     []model.DecisionStatePart
	Questions []model.DecisionQuestion

	MaxOutputTokens *int
	Reasoning       model.ReasoningEffort
	MaxRetries      int

	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// DecideEndEvent describes a Decide call that completed with valid answers.
type DecideEndEvent struct {
	DecideStartEvent

	Result DecideResult
}

// DecisionModelCallStartEvent describes the decision model call, including
// its retries, before it starts.
type DecisionModelCallStartEvent struct {
	CallID string

	// OperationID is "ai.decide.doDecide".
	OperationID string

	Provider string
	ModelID  string

	// State is the state sent to the model, with files resolved.
	State     []model.DecisionStatePart
	Questions []model.DecisionQuestion
}

// DecisionModelCallEndEvent describes the decision model call after its
// answers were validated.
type DecisionModelCallEndEvent struct {
	DecisionModelCallStartEvent

	Result model.DecisionResult
}

// DecisionTelemetry receives the lifecycle events of Decide.
type DecisionTelemetry interface {
	OnDecideStart(ctx context.Context, event TelemetryEvent[DecideStartEvent])
	OnDecideEnd(ctx context.Context, event TelemetryEvent[DecideEndEvent])
	OnDecisionModelCallStart(ctx context.Context, event TelemetryEvent[DecisionModelCallStartEvent])
	OnDecisionModelCallEnd(ctx context.Context, event TelemetryEvent[DecisionModelCallEndEvent])
}

// OnDecideStart does nothing.
func (NopTelemetry) OnDecideStart(context.Context, TelemetryEvent[DecideStartEvent]) {}

// OnDecideEnd does nothing.
func (NopTelemetry) OnDecideEnd(context.Context, TelemetryEvent[DecideEndEvent]) {}

// OnDecisionModelCallStart does nothing.
func (NopTelemetry) OnDecisionModelCallStart(context.Context, TelemetryEvent[DecisionModelCallStartEvent]) {
}

// OnDecisionModelCallEnd does nothing.
func (NopTelemetry) OnDecisionModelCallEnd(context.Context, TelemetryEvent[DecisionModelCallEndEvent]) {
}

func (d telemetryDispatcher) onDecideStart(ctx context.Context, event DecideStartEvent) {
	notifyTelemetry(ctx, d, event, DecisionTelemetry.OnDecideStart)
}

func (d telemetryDispatcher) onDecideEnd(ctx context.Context, event DecideEndEvent) {
	notifyTelemetry(ctx, d, event, DecisionTelemetry.OnDecideEnd)
}

func (d telemetryDispatcher) onDecisionModelCallStart(ctx context.Context, event DecisionModelCallStartEvent) {
	notifyTelemetry(ctx, d, event, DecisionTelemetry.OnDecisionModelCallStart)
}

func (d telemetryDispatcher) onDecisionModelCallEnd(ctx context.Context, event DecisionModelCallEndEvent) {
	notifyTelemetry(ctx, d, event, DecisionTelemetry.OnDecisionModelCallEnd)
}

func (d decision) startEvent() DecideStartEvent {
	return DecideStartEvent{
		CallID:          d.callID,
		OperationID:     decideOperationID,
		Provider:        d.options.Model.Provider(),
		ModelID:         d.options.Model.ModelID(),
		State:           d.options.State,
		Questions:       d.options.Questions,
		MaxOutputTokens: d.options.MaxOutputTokens,
		Reasoning:       d.options.Reasoning,
		MaxRetries:      d.retry.maxRetries,
		Headers:         d.options.Headers,
		ProviderOptions: d.options.ProviderOptions,
	}
}

func (d decision) modelCallStartEvent(state []model.DecisionStatePart) DecisionModelCallStartEvent {
	return DecisionModelCallStartEvent{
		CallID:      d.callID,
		OperationID: decisionModelCallOperationID,
		Provider:    d.options.Model.Provider(),
		ModelID:     d.options.Model.ModelID(),
		State:       state,
		Questions:   d.options.Questions,
	}
}
