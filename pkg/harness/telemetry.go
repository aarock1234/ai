package harness

import (
	"context"
	"slices"

	"github.com/aarock1234/ai/pkg/ai"
)

// telemetryDispatcher sends turn events to the telemetry integrations, in
// order, with the call's resolved telemetry settings. Turn events go to the
// integrations that implement ai.TextTelemetry; errors go to every
// integration.
type telemetryDispatcher struct {
	integrations []ai.Telemetry
	metadata     ai.TelemetryMetadata
}

// newTelemetryDispatcher resolves the telemetry options: integrations
// receive events unless IsEnabled is false, and inputs and outputs are
// recorded unless disabled.
func newTelemetryDispatcher(options ai.TelemetryOptions) telemetryDispatcher {
	if options.IsEnabled != nil && !*options.IsEnabled {
		return telemetryDispatcher{}
	}

	return telemetryDispatcher{
		integrations: options.Integrations,
		metadata: ai.TelemetryMetadata{
			RecordInputs:  options.RecordInputs == nil || *options.RecordInputs,
			RecordOutputs: options.RecordOutputs == nil || *options.RecordOutputs,
			FunctionID:    options.FunctionID,
		},
	}
}

func (d telemetryDispatcher) onStart(ctx context.Context, event ai.GenerateTextStartEvent) {
	notifyText(ctx, d, event, ai.TextTelemetry.OnGenerateTextStart)
}

func (d telemetryDispatcher) onStepStart(ctx context.Context, event ai.GenerateTextStepStartEvent) {
	notifyText(ctx, d, event, ai.TextTelemetry.OnGenerateTextStepStart)
}

func (d telemetryDispatcher) onLanguageModelCallStart(ctx context.Context, event ai.LanguageModelCallStartEvent) {
	notifyText(ctx, d, event, ai.TextTelemetry.OnLanguageModelCallStart)
}

func (d telemetryDispatcher) onLanguageModelCallEnd(ctx context.Context, event ai.LanguageModelCallEndEvent) {
	notifyText(ctx, d, event, ai.TextTelemetry.OnLanguageModelCallEnd)
}

func (d telemetryDispatcher) onToolExecutionStart(ctx context.Context, event ai.ToolExecutionStartEvent) {
	notifyText(ctx, d, event, ai.TextTelemetry.OnToolExecutionStart)
}

func (d telemetryDispatcher) onToolExecutionEnd(ctx context.Context, event ai.ToolExecutionEndEvent) {
	notifyText(ctx, d, event, ai.TextTelemetry.OnToolExecutionEnd)
}

func (d telemetryDispatcher) onStepEnd(ctx context.Context, step ai.StepResult) {
	notifyText(ctx, d, step, ai.TextTelemetry.OnGenerateTextStepEnd)
}

func (d telemetryDispatcher) onEnd(ctx context.Context, event ai.GenerateTextEndEvent) {
	notifyText(ctx, d, event, ai.TextTelemetry.OnGenerateTextEnd)
}

func (d telemetryDispatcher) onAbort(ctx context.Context, event ai.StreamTextAbortEvent) {
	notifyText(ctx, d, event, ai.TextTelemetry.OnGenerateTextAbort)
}

func (d telemetryDispatcher) onError(ctx context.Context, event ai.TelemetryErrorEvent) {
	for _, integration := range d.integrations {
		integration.OnError(ctx, telemetryEvent(d.metadata, event))
	}
}

// executeTool runs execute inside the ExecuteTool of every text
// integration, the first integration outermost.
func (d telemetryDispatcher) executeTool(ctx context.Context, event ai.ToolExecutionStartEvent, execute func(ctx context.Context) error) error {
	wrapped := execute

	for _, registered := range slices.Backward(d.integrations) {
		integration, ok := registered.(ai.TextTelemetry)
		if !ok {
			continue
		}

		inner := wrapped

		wrapped = func(ctx context.Context) error {
			return integration.ExecuteTool(ctx, telemetryEvent(d.metadata, event), inner)
		}
	}

	return wrapped(ctx)
}

// notifyText passes an event, in order, to the integrations that implement
// ai.TextTelemetry.
func notifyText[Event any](
	ctx context.Context,
	dispatcher telemetryDispatcher,
	event Event,
	notify func(integration ai.TextTelemetry, ctx context.Context, event ai.TelemetryEvent[Event]),
) {
	for _, integration := range dispatcher.integrations {
		textIntegration, ok := integration.(ai.TextTelemetry)
		if !ok {
			continue
		}

		notify(textIntegration, ctx, telemetryEvent(dispatcher.metadata, event))
	}
}

// telemetryEvent wraps an event with the call's telemetry settings.
func telemetryEvent[Event any](metadata ai.TelemetryMetadata, event Event) ai.TelemetryEvent[Event] {
	return ai.TelemetryEvent[Event]{
		TelemetryMetadata: metadata,
		Event:             event,
	}
}
