package otel

import (
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/ai"
)

// Operation IDs of package ai, as reported by its telemetry events.
const (
	operationGenerateText     = "ai.generateText"
	operationStreamText       = "ai.streamText"
	operationEmbed            = "ai.embed"
	operationEmbedMany        = "ai.embedMany"
	operationRerank           = "ai.rerank"
	operationGenerateSpeech   = "ai.generateSpeech"
	operationTranscribe       = "ai.transcribe"
	operationStreamTranscribe = "ai.streamTranscribe"
	operationStreamTranslate  = "ai.streamTranslate"
)

// defaultMaxRetries is upstream's retry count when an operation does not
// set one.
const defaultMaxRetries = 2

// callState holds the open spans of one operation call. The integration's
// mutex guards it.
type callState struct {
	operationID string
	metadata    ai.TelemetryMetadata
	provider    string
	modelID     string
	settings    ai.CallSettings

	// baseAttributes are added to every span of the call.
	baseAttributes []attribute.KeyValue

	rootSpan      trace.Span
	stepSpan      trace.Span
	inferenceSpan trace.Span
	rerankSpan    trace.Span
	decisionSpan  trace.Span

	// inferenceTools are the tool definitions of the open chat span.
	inferenceTools []wirejson.ToolDefinition

	// embedSpans are keyed by embedding model call ID, and toolSpans by tool
	// call ID.
	embedSpans map[string]trace.Span
	toolSpans  map[string]trace.Span
}

// callStart describes the root span of an operation call.
type callStart struct {
	operationID    string
	metadata       ai.TelemetryMetadata
	provider       string
	modelID        string
	settings       ai.CallSettings
	baseAttributes []attribute.KeyValue
	rootSpan       trace.Span
}

func newCallState(start callStart) *callState {
	return &callState{
		operationID:    start.operationID,
		metadata:       start.metadata,
		provider:       start.provider,
		modelID:        start.modelID,
		settings:       start.settings,
		baseAttributes: start.baseAttributes,
		rootSpan:       start.rootSpan,
		embedSpans:     make(map[string]trace.Span),
		toolSpans:      make(map[string]trace.Span),
	}
}

// parentSpan returns the step span, or the root span outside a step.
func (s *callState) parentSpan() trace.Span {
	if s.stepSpan != nil {
		return s.stepSpan
	}

	return s.rootSpan
}

// endOpenSpans ends every span of the call. When err is not nil, record
// first records it on each span.
func (s *callState) endOpenSpans(err error, record errorRecorder) {
	for _, span := range s.toolSpans {
		endSpan(span, err, record)
	}

	clear(s.toolSpans)

	if s.inferenceSpan != nil {
		endSpan(s.inferenceSpan, err, record)
		s.inferenceSpan = nil
		s.inferenceTools = nil
	}

	if s.stepSpan != nil {
		endSpan(s.stepSpan, err, record)
		s.stepSpan = nil
	}

	for _, span := range s.embedSpans {
		endSpan(span, err, record)
	}

	clear(s.embedSpans)

	if s.rerankSpan != nil {
		endSpan(s.rerankSpan, err, record)
		s.rerankSpan = nil
	}

	if s.decisionSpan != nil {
		endSpan(s.decisionSpan, err, record)
		s.decisionSpan = nil
	}

	endSpan(s.rootSpan, err, record)
}

// errorRecorder records an error on a span.
type errorRecorder func(span trace.Span, err error)

// endSpan ends span, recording err first when it is not nil.
func endSpan(span trace.Span, err error, record errorRecorder) {
	if err != nil {
		record(span, err)
	}

	span.End()
}

// maxRetriesOrDefault resolves an optional retry count.
func maxRetriesOrDefault(maxRetries *int) int {
	if maxRetries == nil {
		return defaultMaxRetries
	}

	return *maxRetries
}
