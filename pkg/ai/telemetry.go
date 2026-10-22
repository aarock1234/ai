package ai

import (
	"context"
)

// TelemetryOptions configure the telemetry of one call, such as a text
// generation, an embedding or a transcription.
//
// Go adaptation: upstream also reads integrations from a global registry.
// This SDK has no global state, so integrations are passed per call.
type TelemetryOptions struct {
	// IsEnabled turns telemetry off when false. Nil enables it whenever
	// Integrations is not empty.
	IsEnabled *bool

	// RecordInputs and RecordOutputs tell integrations whether to record the
	// inputs and outputs of the call. Nil records them.
	RecordInputs  *bool
	RecordOutputs *bool

	// FunctionID identifies the calling function, so that telemetry can be
	// grouped by it.
	FunctionID string

	// Integrations receive the lifecycle events of the call, in order.
	Integrations []Telemetry
}

// Telemetry is a telemetry integration, for example one that records
// OpenTelemetry spans. Every integration receives the errors that end an
// operation. It also implements the optional interface of each operation
// it records: TextTelemetry, EmbedTelemetry, RerankTelemetry,
// SpeechTelemetry, TranscriptionTelemetry, TranslationTelemetry and
// DecisionTelemetry. Each operation finds its interface with a type
// assertion and skips integrations that do not implement it. Embed
// NopTelemetry to implement every interface and override only some
// methods.
//
// Go adaptation: upstream sends the start and end of every operation to one
// onStart and onEnd callback with a union of events. Here each operation has
// its own interface with typed methods, and upstream's onEmbedStart,
// onEmbedEnd, onRerankStart and onRerankEnd model call events are the
// EmbeddingModelCall and RerankingModelCall methods.
type Telemetry interface {
	// OnError is called when an error ends an operation.
	OnError(ctx context.Context, event TelemetryEvent[TelemetryErrorEvent])
}

// TextTelemetry receives the lifecycle events of GenerateText, StreamText
// and agents.
//
// The Execute methods wrap the model call and each tool execution. They must
// call execute exactly once, passing a context derived from ctx (for example
// one that carries a span), and return its error.
type TextTelemetry interface {
	// OnGenerateTextStart is called when a generation begins.
	OnGenerateTextStart(ctx context.Context, event TelemetryEvent[GenerateTextStartEvent])

	// OnGenerateTextStepStart is called before each step's model call.
	OnGenerateTextStepStart(ctx context.Context, event TelemetryEvent[GenerateTextStepStartEvent])

	// OnLanguageModelCallStart is called immediately before each model call.
	OnLanguageModelCallStart(ctx context.Context, event TelemetryEvent[LanguageModelCallStartEvent])

	// OnLanguageModelCallEnd is called after each model response was parsed,
	// before client-side tools run.
	OnLanguageModelCallEnd(ctx context.Context, event TelemetryEvent[LanguageModelCallEndEvent])

	// OnToolExecutionStart is called before a tool's Execute function runs.
	OnToolExecutionStart(ctx context.Context, event TelemetryEvent[ToolExecutionStartEvent])

	// OnToolExecutionEnd is called after a tool execution succeeded or
	// failed.
	OnToolExecutionEnd(ctx context.Context, event TelemetryEvent[ToolExecutionEndEvent])

	// OnGenerateTextStepEnd is called after each step.
	OnGenerateTextStepEnd(ctx context.Context, event TelemetryEvent[StepResult])

	// OnGenerateTextEnd is called when a generation completes.
	OnGenerateTextEnd(ctx context.Context, event TelemetryEvent[GenerateTextEndEvent])

	// OnGenerateTextAbort is called when a stream is aborted before it
	// completes.
	OnGenerateTextAbort(ctx context.Context, event TelemetryEvent[StreamTextAbortEvent])

	// ExecuteLanguageModelCall runs one model call attempt.
	ExecuteLanguageModelCall(ctx context.Context, event TelemetryEvent[LanguageModelCallStartEvent], execute func(ctx context.Context) error) error

	// ExecuteTool runs one tool execution, so that nested generations inside
	// the tool become children of the tool's telemetry context.
	ExecuteTool(ctx context.Context, event TelemetryEvent[ToolExecutionStartEvent], execute func(ctx context.Context) error) error
}

// TelemetryEvent is a lifecycle event together with the call's telemetry
// settings.
type TelemetryEvent[Event any] struct {
	TelemetryMetadata

	Event Event
}

// TelemetryMetadata are the telemetry settings of a call, resolved to their
// defaults.
type TelemetryMetadata struct {
	RecordInputs  bool
	RecordOutputs bool
	FunctionID    string
}

// TelemetryErrorEvent describes an error that ended a generation or another
// operation.
type TelemetryErrorEvent struct {
	CallID string
	Error  error
}

// NopTelemetry implements Telemetry and every optional telemetry interface
// with methods that do nothing and Execute methods that only call execute.
// Embed it in an integration to override the events it handles.
type NopTelemetry struct{}

// OnError does nothing.
func (NopTelemetry) OnError(context.Context, TelemetryEvent[TelemetryErrorEvent]) {}

// OnGenerateTextStart does nothing.
func (NopTelemetry) OnGenerateTextStart(context.Context, TelemetryEvent[GenerateTextStartEvent]) {}

// OnGenerateTextStepStart does nothing.
func (NopTelemetry) OnGenerateTextStepStart(context.Context, TelemetryEvent[GenerateTextStepStartEvent]) {
}

// OnLanguageModelCallStart does nothing.
func (NopTelemetry) OnLanguageModelCallStart(context.Context, TelemetryEvent[LanguageModelCallStartEvent]) {
}

// OnLanguageModelCallEnd does nothing.
func (NopTelemetry) OnLanguageModelCallEnd(context.Context, TelemetryEvent[LanguageModelCallEndEvent]) {
}

// OnToolExecutionStart does nothing.
func (NopTelemetry) OnToolExecutionStart(context.Context, TelemetryEvent[ToolExecutionStartEvent]) {}

// OnToolExecutionEnd does nothing.
func (NopTelemetry) OnToolExecutionEnd(context.Context, TelemetryEvent[ToolExecutionEndEvent]) {}

// OnGenerateTextStepEnd does nothing.
func (NopTelemetry) OnGenerateTextStepEnd(context.Context, TelemetryEvent[StepResult]) {}

// OnGenerateTextEnd does nothing.
func (NopTelemetry) OnGenerateTextEnd(context.Context, TelemetryEvent[GenerateTextEndEvent]) {}

// OnGenerateTextAbort does nothing.
func (NopTelemetry) OnGenerateTextAbort(context.Context, TelemetryEvent[StreamTextAbortEvent]) {}

// ExecuteLanguageModelCall calls execute with ctx.
func (NopTelemetry) ExecuteLanguageModelCall(ctx context.Context, _ TelemetryEvent[LanguageModelCallStartEvent], execute func(ctx context.Context) error) error {
	return execute(ctx)
}

// ExecuteTool calls execute with ctx.
func (NopTelemetry) ExecuteTool(ctx context.Context, _ TelemetryEvent[ToolExecutionStartEvent], execute func(ctx context.Context) error) error {
	return execute(ctx)
}

// telemetryDispatcher sends the events of one call to its integrations. The
// zero value dispatches nothing.
type telemetryDispatcher struct {
	integrations []Telemetry
	metadata     TelemetryMetadata
}

func newTelemetryDispatcher(options TelemetryOptions) telemetryDispatcher {
	if options.IsEnabled != nil && !*options.IsEnabled {
		return telemetryDispatcher{}
	}

	return telemetryDispatcher{
		integrations: options.Integrations,
		metadata: TelemetryMetadata{
			RecordInputs:  enabledByDefault(options.RecordInputs),
			RecordOutputs: enabledByDefault(options.RecordOutputs),
			FunctionID:    options.FunctionID,
		},
	}
}

// enabled reports whether any integration receives events.
func (d telemetryDispatcher) enabled() bool {
	return len(d.integrations) > 0
}

func (d telemetryDispatcher) onGenerateTextStart(ctx context.Context, event GenerateTextStartEvent) {
	notifyTelemetry(ctx, d, event, TextTelemetry.OnGenerateTextStart)
}

func (d telemetryDispatcher) onGenerateTextStepStart(ctx context.Context, event GenerateTextStepStartEvent) {
	notifyTelemetry(ctx, d, event, TextTelemetry.OnGenerateTextStepStart)
}

func (d telemetryDispatcher) onLanguageModelCallStart(ctx context.Context, event LanguageModelCallStartEvent) {
	notifyTelemetry(ctx, d, event, TextTelemetry.OnLanguageModelCallStart)
}

func (d telemetryDispatcher) onLanguageModelCallEnd(ctx context.Context, event LanguageModelCallEndEvent) {
	notifyTelemetry(ctx, d, event, TextTelemetry.OnLanguageModelCallEnd)
}

func (d telemetryDispatcher) onToolExecutionStart(ctx context.Context, event ToolExecutionStartEvent) {
	notifyTelemetry(ctx, d, event, TextTelemetry.OnToolExecutionStart)
}

func (d telemetryDispatcher) onToolExecutionEnd(ctx context.Context, event ToolExecutionEndEvent) {
	notifyTelemetry(ctx, d, event, TextTelemetry.OnToolExecutionEnd)
}

func (d telemetryDispatcher) onGenerateTextStepEnd(ctx context.Context, step StepResult) {
	notifyTelemetry(ctx, d, step, TextTelemetry.OnGenerateTextStepEnd)
}

func (d telemetryDispatcher) onGenerateTextEnd(ctx context.Context, event GenerateTextEndEvent) {
	notifyTelemetry(ctx, d, event, TextTelemetry.OnGenerateTextEnd)
}

func (d telemetryDispatcher) onGenerateTextAbort(ctx context.Context, event StreamTextAbortEvent) {
	notifyTelemetry(ctx, d, event, TextTelemetry.OnGenerateTextAbort)
}

func (d telemetryDispatcher) onError(ctx context.Context, event TelemetryErrorEvent) {
	notifyTelemetry(ctx, d, event, Telemetry.OnError)
}

// onOperationError reports the error that ended an operation and returns
// it, so that callers can report and return in one statement.
func (d telemetryDispatcher) onOperationError(ctx context.Context, callID string, err error) error {
	d.onError(ctx, TelemetryErrorEvent{
		CallID: callID,
		Error:  err,
	})

	return err
}

// executeLanguageModelCall runs one model call attempt inside the wrappers
// of every text integration.
func (d telemetryDispatcher) executeLanguageModelCall(ctx context.Context, event LanguageModelCallStartEvent, execute func(ctx context.Context) error) error {
	return wrapWithTextTelemetry(d, event, TextTelemetry.ExecuteLanguageModelCall, execute)(ctx)
}

// executeTool runs one tool execution inside the wrappers of every text
// integration.
func (d telemetryDispatcher) executeTool(ctx context.Context, event ToolExecutionStartEvent, execute func(ctx context.Context) error) error {
	return wrapWithTextTelemetry(d, event, TextTelemetry.ExecuteTool, execute)(ctx)
}

// notifyTelemetry passes an event, in order, to the integrations that
// implement the operation's telemetry interface.
func notifyTelemetry[Integration any, Event any](
	ctx context.Context,
	dispatcher telemetryDispatcher,
	event Event,
	notify func(integration Integration, ctx context.Context, event TelemetryEvent[Event]),
) {
	telemetryEvent := telemetryEventOf(dispatcher.metadata, event)

	for _, integration := range dispatcher.integrations {
		operationIntegration, ok := integration.(Integration)
		if !ok {
			continue
		}

		notify(operationIntegration, ctx, telemetryEvent)
	}
}

// wrapWithTextTelemetry nests execute inside the wrapper of every
// integration that implements TextTelemetry. The last integration is the
// outermost, as upstream composes them.
func wrapWithTextTelemetry[Event any](
	dispatcher telemetryDispatcher,
	event Event,
	wrap func(integration TextTelemetry, ctx context.Context, event TelemetryEvent[Event], execute func(ctx context.Context) error) error,
	execute func(ctx context.Context) error,
) func(ctx context.Context) error {
	telemetryEvent := telemetryEventOf(dispatcher.metadata, event)

	wrapped := execute
	for _, integration := range dispatcher.integrations {
		textIntegration, ok := integration.(TextTelemetry)
		if !ok {
			continue
		}

		inner := wrapped
		wrapped = func(ctx context.Context) error {
			return wrap(textIntegration, ctx, telemetryEvent, inner)
		}
	}

	return wrapped
}

// telemetryEventOf wraps an event with the call's telemetry settings.
func telemetryEventOf[Event any](metadata TelemetryMetadata, event Event) TelemetryEvent[Event] {
	return TelemetryEvent[Event]{
		TelemetryMetadata: metadata,
		Event:             event,
	}
}

func enabledByDefault(option *bool) bool {
	return option == nil || *option
}
