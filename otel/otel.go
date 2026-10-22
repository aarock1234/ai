// Package otel records the telemetry events of package ai as OpenTelemetry
// spans, a port of the Vercel AI SDK @ai-sdk/otel package.
//
// OpenTelemetry follows the OpenTelemetry semantic conventions for
// generative AI: operation, step, chat, tool, embedding, reranking and
// decision spans with gen_ai.* attributes. Legacy records the earlier AI SDK span layout
// with ai.* attributes, for backends that still read it.
//
// Pass an integration in ai.TelemetryOptions.Integrations:
//
//	telemetry := otel.New(otel.Options{
//		Tracer: tracerProvider.Tracer("my-app"),
//	})
//
//	result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
//		Model:  languageModel,
//		Prompt: ai.Prompt{Prompt: "hello"},
//		Telemetry: ai.TelemetryOptions{
//			Integrations: []ai.Telemetry{telemetry},
//		},
//	})
//
// Go adaptations: spans become children of the span in the context passed
// to the operation, instead of the active OpenTelemetry context. Upstream's
// runtime context is the Go context.Context, so EnrichSpan receives it and
// there is no runtime context attribute option. The deprecated object
// generation operations are not ported, so their spans and the schema
// option are absent.
package otel

import (
	"context"
	"sync"

	otelglobal "go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/aarock1234/ai/pkg/ai"
)

// defaultTracerName names the tracer taken from the global tracer provider
// when Options.Tracer is nil.
const defaultTracerName = "gen_ai"

// SpanType identifies the kind of span passed to EnrichSpan.
type SpanType string

// Span types.
const (
	SpanTypeOperation     SpanType = "operation"
	SpanTypeStep          SpanType = "step"
	SpanTypeLanguageModel SpanType = "languageModel"
	SpanTypeTool          SpanType = "tool"
	SpanTypeEmbedding     SpanType = "embedding"
	SpanTypeReranking     SpanType = "reranking"
	SpanTypeDecision      SpanType = "experimental_decision"
)

// SpanInfo describes a span that is about to start.
type SpanInfo struct {
	Type SpanType

	// OperationID identifies the operation, such as "ai.generateText".
	OperationID string

	// CallID identifies the operation call the span belongs to.
	CallID string
}

// EnrichSpan returns custom attributes for a span when it starts. The
// context is the one passed to the telemetry event, so it carries the
// application's request-scoped values. Attributes recorded by the
// integration take precedence over custom attributes with the same key.
// EnrichSpan must not call the integration.
type EnrichSpan func(ctx context.Context, span SpanInfo) []attribute.KeyValue

// Options configure an OpenTelemetry integration. The boolean fields enable
// supplemental AI SDK attributes that the generative AI semantic
// conventions do not define.
type Options struct {
	// Tracer creates the spans. Nil uses the global tracer provider's
	// "gen_ai" tracer.
	Tracer trace.Tracer

	// EnrichSpan adds custom attributes to spans when they start.
	EnrichSpan EnrichSpan

	// Usage records usage details: tokens without cache, text tokens and
	// reasoning tokens, and the provider's native usage of audio operations.
	Usage bool

	// ProviderMetadata records provider metadata on response spans.
	ProviderMetadata bool

	// Embedding records embedding input values and output embeddings.
	Embedding bool

	// Reranking records reranking input documents and the output ranking.
	Reranking bool

	// ExperimentalDecision records the decision state, questions and
	// answers of Decide.
	ExperimentalDecision bool

	// Headers records request headers.
	Headers bool

	// ToolChoice records the tool choice of each step.
	ToolChoice bool
}

// OpenTelemetry records telemetry events as spans that follow the
// OpenTelemetry semantic conventions for generative AI. It is safe for
// concurrent use by many operations.
type OpenTelemetry struct {
	tracer  trace.Tracer
	options Options

	mu    sync.Mutex
	calls map[string]*callState
}

var (
	_ ai.Telemetry              = (*OpenTelemetry)(nil)
	_ ai.TextTelemetry          = (*OpenTelemetry)(nil)
	_ ai.EmbedTelemetry         = (*OpenTelemetry)(nil)
	_ ai.RerankTelemetry        = (*OpenTelemetry)(nil)
	_ ai.SpeechTelemetry        = (*OpenTelemetry)(nil)
	_ ai.TranscriptionTelemetry = (*OpenTelemetry)(nil)
	_ ai.TranslationTelemetry   = (*OpenTelemetry)(nil)
)

// New returns an OpenTelemetry integration.
func New(options Options) *OpenTelemetry {
	return &OpenTelemetry{
		tracer:  tracerOrDefault(options.Tracer, defaultTracerName),
		options: options,
		calls:   make(map[string]*callState),
	}
}

// tracerOrDefault returns tracer, or the named tracer of the global tracer
// provider when tracer is nil.
func tracerOrDefault(tracer trace.Tracer, name string) trace.Tracer {
	if tracer != nil {
		return tracer
	}

	return otelglobal.GetTracerProvider().Tracer(name)
}

// ExecuteLanguageModelCall runs the model call with the chat span, or the
// step span when there is none, as the parent of spans created inside it.
func (t *OpenTelemetry) ExecuteLanguageModelCall(
	ctx context.Context,
	event ai.TelemetryEvent[ai.LanguageModelCallStartEvent],
	execute func(ctx context.Context) error,
) error {
	span := t.languageModelCallSpan(event.Event.CallID)
	if span == nil {
		return execute(ctx)
	}

	return execute(trace.ContextWithSpan(ctx, span))
}

// ExecuteTool runs the tool with its tool span as the parent of spans
// created inside it, such as those of nested generations.
func (t *OpenTelemetry) ExecuteTool(
	ctx context.Context,
	event ai.TelemetryEvent[ai.ToolExecutionStartEvent],
	execute func(ctx context.Context) error,
) error {
	span := t.toolSpan(event.Event.CallID, event.Event.ToolCall.ToolCallID)
	if span == nil {
		return execute(ctx)
	}

	return execute(trace.ContextWithSpan(ctx, span))
}

func (t *OpenTelemetry) languageModelCallSpan(callID string) trace.Span {
	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[callID]
	if state == nil {
		return nil
	}

	if state.inferenceSpan != nil {
		return state.inferenceSpan
	}

	return state.stepSpan
}

func (t *OpenTelemetry) toolSpan(callID, toolCallID string) trace.Span {
	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[callID]
	if state == nil {
		return nil
	}

	return state.toolSpans[toolCallID]
}

// spanStart describes a span to start.
type spanStart struct {
	name       string
	kind       trace.SpanKind
	spanType   SpanType
	callID     string
	operation  string
	attributes []attribute.KeyValue
}

// startSpan starts a span below the span in parent, adding the custom
// attributes of EnrichSpan before the integration's own attributes.
func (t *OpenTelemetry) startSpan(parent context.Context, start spanStart) trace.Span {
	attributes := t.customAttributes(parent, start)
	attributes = append(attributes, start.attributes...)

	_, span := t.tracer.Start(
		parent,
		start.name,
		trace.WithSpanKind(start.kind),
		trace.WithAttributes(attributes...),
	)

	return span
}

func (t *OpenTelemetry) customAttributes(ctx context.Context, start spanStart) []attribute.KeyValue {
	if t.options.EnrichSpan == nil {
		return nil
	}

	custom := t.options.EnrichSpan(ctx, SpanInfo{
		Type:        start.spanType,
		OperationID: start.operation,
		CallID:      start.callID,
	})

	return sanitizeAttributes(custom)
}
