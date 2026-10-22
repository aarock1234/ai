package otel

import (
	"context"
	"net/http"
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/aarock1234/ai/pkg/ai"
)

// legacyTracerName names the tracer taken from the global tracer provider
// when LegacyOptions.Tracer is nil.
const legacyTracerName = "ai"

// LegacyOptions configure a Legacy integration.
type LegacyOptions struct {
	// Tracer creates the spans. Nil uses the global tracer provider's "ai"
	// tracer.
	Tracer trace.Tracer
}

// Legacy records telemetry events in the span layout of earlier AI SDK
// versions: spans named after operation IDs, such as "ai.generateText" and
// "ai.generateText.doGenerate", with ai.* attributes and a few gen_ai.*
// attributes. Use it for observability backends that read that layout. It
// is safe for concurrent use by many operations.
type Legacy struct {
	ai.NopTelemetry

	tracer trace.Tracer

	mu    sync.Mutex
	calls map[string]*callState
}

var (
	_ ai.Telemetry              = (*Legacy)(nil)
	_ ai.TextTelemetry          = (*Legacy)(nil)
	_ ai.EmbedTelemetry         = (*Legacy)(nil)
	_ ai.RerankTelemetry        = (*Legacy)(nil)
	_ ai.SpeechTelemetry        = (*Legacy)(nil)
	_ ai.TranscriptionTelemetry = (*Legacy)(nil)
	_ ai.TranslationTelemetry   = (*Legacy)(nil)
)

// NewLegacy returns a Legacy integration.
func NewLegacy(options LegacyOptions) *Legacy {
	return &Legacy{
		tracer: tracerOrDefault(options.Tracer, legacyTracerName),
		calls:  make(map[string]*callState),
	}
}

// ExecuteLanguageModelCall runs the model call with the step span as the
// parent of spans created inside it.
func (l *Legacy) ExecuteLanguageModelCall(
	ctx context.Context,
	event ai.TelemetryEvent[ai.LanguageModelCallStartEvent],
	execute func(ctx context.Context) error,
) error {
	span := l.stepSpan(event.Event.CallID)
	if span == nil {
		return execute(ctx)
	}

	return execute(trace.ContextWithSpan(ctx, span))
}

// ExecuteTool runs the tool with its tool span as the parent of spans
// created inside it.
func (l *Legacy) ExecuteTool(
	ctx context.Context,
	event ai.TelemetryEvent[ai.ToolExecutionStartEvent],
	execute func(ctx context.Context) error,
) error {
	span := l.toolSpan(event.Event.CallID, event.Event.ToolCall.ToolCallID)
	if span == nil {
		return execute(ctx)
	}

	return execute(trace.ContextWithSpan(ctx, span))
}

func (l *Legacy) stepSpan(callID string) trace.Span {
	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[callID]
	if state == nil {
		return nil
	}

	return state.stepSpan
}

func (l *Legacy) toolSpan(callID, toolCallID string) trace.Span {
	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[callID]
	if state == nil {
		return nil
	}

	return state.toolSpans[toolCallID]
}

// startSpan starts a span named after its operation below the span in
// parent.
func (l *Legacy) startSpan(parent context.Context, name string, attributes *spanAttributes) trace.Span {
	_, span := l.tracer.Start(
		parent,
		name,
		trace.WithAttributes(attributes.list()...),
	)

	return span
}

// OnGenerateTextAbort ends every open span of an aborted stream.
func (l *Legacy) OnGenerateTextAbort(_ context.Context, event ai.TelemetryEvent[ai.StreamTextAbortEvent]) {
	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[event.Event.CallID]
	if state == nil {
		return
	}

	state.endOpenSpans(nil, recordLegacyError)
	delete(l.calls, event.Event.CallID)
}

// OnError records the error that ended an operation on its open spans and
// ends them.
func (l *Legacy) OnError(_ context.Context, event ai.TelemetryEvent[ai.TelemetryErrorEvent]) {
	failure := event.Event

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[failure.CallID]
	if state == nil {
		return
	}

	state.endOpenSpans(failure.Error, recordLegacyError)
	delete(l.calls, failure.CallID)
}

// setOperationName records the operation of a span and the function ID of
// the call.
func setOperationName(attributes *spanAttributes, operationID string) {
	functionID := attributes.metadata.FunctionID

	name := operationID
	if functionID != "" {
		name += " " + functionID
	}

	attributes.setString("operation.name", name)
	attributes.setString("resource.name", functionID)
	attributes.setString("ai.operationId", operationID)
	attributes.setString("ai.telemetry.functionId", functionID)
}

// legacyModel identifies the model of a call.
type legacyModel struct {
	provider string
	modelID  string
}

// legacyBaseAttributes returns the attributes added to every span of a
// call: the model, the call settings and the request headers.
func legacyBaseAttributes(
	metadata ai.TelemetryMetadata,
	model legacyModel,
	settings ai.CallSettings,
	headers http.Header,
) []attribute.KeyValue {
	attributes := newSpanAttributes(metadata)
	attributes.setString("ai.model.provider", model.provider)
	attributes.setString("ai.model.id", model.modelID)
	attributes.setInt("ai.settings.maxOutputTokens", settings.MaxOutputTokens)
	attributes.setFloat("ai.settings.temperature", settings.Temperature)
	attributes.setFloat("ai.settings.topP", settings.TopP)
	attributes.setInt("ai.settings.topK", settings.TopK)
	attributes.setFloat("ai.settings.presencePenalty", settings.PresencePenalty)
	attributes.setFloat("ai.settings.frequencyPenalty", settings.FrequencyPenalty)
	attributes.setStrings("ai.settings.stopSequences", settings.StopSequences)
	attributes.setInt("ai.settings.seed", settings.Seed)
	attributes.setInt("ai.settings.maxRetries", settings.MaxRetries)
	attributes.add(headerAttributes(headers)...)

	return attributes.list()
}

// milliseconds converts a duration to fractional milliseconds.
func milliseconds(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}
