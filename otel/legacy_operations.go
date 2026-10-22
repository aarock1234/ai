package otel

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/aarock1234/ai/pkg/ai"
)

// OnEmbedStart starts the root span of Embed or EmbedMany.
func (l *Legacy) OnEmbedStart(ctx context.Context, event ai.TelemetryEvent[ai.EmbedStartEvent]) {
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
	setEmbeddingValues(attributes, start.OperationID, start.Values)

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

// OnEmbedEnd records the embeddings and usage and ends the root span.
func (l *Legacy) OnEmbedEnd(_ context.Context, event ai.TelemetryEvent[ai.EmbedEndEvent]) {
	end := event.Event

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[end.CallID]
	if state == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	setEmbeddingOutputs(attributes, state.operationID, end.Embeddings)
	setEmbeddingTokens(attributes, "ai.usage.tokens", end.Usage)
	state.rootSpan.SetAttributes(attributes.list()...)

	state.rootSpan.End()
	delete(l.calls, end.CallID)
}

// OnEmbeddingModelCallStart starts the doEmbed span of one embedding model
// call.
func (l *Legacy) OnEmbeddingModelCallStart(ctx context.Context, event ai.TelemetryEvent[ai.EmbeddingModelCallStartEvent]) {
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
	attributes.setInput("ai.values", func() attribute.Value {
		return jsonListValue(call.Values)
	})

	parent := trace.ContextWithSpan(ctx, state.rootSpan)
	state.embedSpans[call.EmbedCallID] = l.startSpan(parent, call.OperationID, attributes)
}

// OnEmbeddingModelCallEnd records the embeddings and usage of one
// embedding model call and ends its span.
func (l *Legacy) OnEmbeddingModelCallEnd(_ context.Context, event ai.TelemetryEvent[ai.EmbeddingModelCallEndEvent]) {
	end := event.Event

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[end.CallID]
	if state == nil {
		return
	}

	span := state.embedSpans[end.EmbedCallID]
	if span == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	attributes.setOutput("ai.embeddings", func() attribute.Value {
		return jsonListValue(end.Embeddings)
	})
	setEmbeddingTokens(attributes, "ai.usage.tokens", end.Usage)
	span.SetAttributes(attributes.list()...)

	span.End()
	delete(state.embedSpans, end.EmbedCallID)
}

// OnRerankStart starts the root span of Rerank.
func (l *Legacy) OnRerankStart(ctx context.Context, event ai.TelemetryEvent[ai.RerankStartEvent[any]]) {
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
	setRerankDocuments(attributes, start.Documents)

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

// OnRerankEnd ends the root span of Rerank.
func (l *Legacy) OnRerankEnd(_ context.Context, event ai.TelemetryEvent[ai.RerankEndEvent[any]]) {
	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[event.Event.CallID]
	if state == nil {
		return
	}

	state.rootSpan.End()
	delete(l.calls, event.Event.CallID)
}

// OnRerankingModelCallStart starts the doRerank span of the reranking model
// call.
func (l *Legacy) OnRerankingModelCallStart(ctx context.Context, event ai.TelemetryEvent[ai.RerankingModelCallStartEvent]) {
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
	setRerankDocuments(attributes, call.Documents)

	parent := trace.ContextWithSpan(ctx, state.rootSpan)
	state.rerankSpan = l.startSpan(parent, call.OperationID, attributes)
}

// OnRerankingModelCallEnd records the ranking and ends the doRerank span.
func (l *Legacy) OnRerankingModelCallEnd(_ context.Context, event ai.TelemetryEvent[ai.RerankingModelCallEndEvent]) {
	end := event.Event

	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[end.CallID]
	if state == nil || state.rerankSpan == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	setRanking(attributes, end)
	state.rerankSpan.SetAttributes(attributes.list()...)

	state.rerankSpan.End()
	state.rerankSpan = nil
}

// OnGenerateSpeechStart starts the root span of GenerateSpeech.
func (l *Legacy) OnGenerateSpeechStart(ctx context.Context, event ai.TelemetryEvent[ai.GenerateSpeechStartEvent]) {
	start := event.Event

	l.startAudioOperation(ctx, event.TelemetryMetadata, audioStart{
		callID:      start.CallID,
		operationID: start.OperationID,
		provider:    start.Provider,
		modelID:     start.ModelID,
		headers:     start.Headers,
		text:        start.Text,
	})
}

// OnTranscriptionStart starts the root span of Transcribe.
func (l *Legacy) OnTranscriptionStart(ctx context.Context, event ai.TelemetryEvent[ai.TranscriptionStartEvent]) {
	l.startAudioOperation(ctx, event.TelemetryMetadata, transcriptionAudioStart(event.Event))
}

// OnStreamTranscriptionStart starts the root span of StreamTranscribe.
func (l *Legacy) OnStreamTranscriptionStart(ctx context.Context, event ai.TelemetryEvent[ai.TranscriptionStartEvent]) {
	l.startAudioOperation(ctx, event.TelemetryMetadata, transcriptionAudioStart(event.Event))
}

func (l *Legacy) startAudioOperation(ctx context.Context, metadata ai.TelemetryMetadata, start audioStart) {
	attributes := newSpanAttributes(metadata)
	setOperationName(attributes, start.operationID)
	attributes.setString("ai.model.provider", start.provider)
	attributes.setString("ai.model.id", start.modelID)
	attributes.setString("ai.request.targetLanguage", start.targetLanguage)
	attributes.setString("ai.request.sourceLanguage", start.sourceLanguage)
	attributes.add(headerAttributes(start.headers)...)

	if start.operationID == operationGenerateSpeech {
		attributes.setInput("ai.request.text", func() attribute.Value {
			return attribute.StringValue(start.text)
		})
	}

	setInputAudio(attributes, start.audioByteLength, start.audioMediaType, "ai.request.audio.mediaType")

	l.mu.Lock()
	defer l.mu.Unlock()

	l.calls[start.callID] = newCallState(callStart{
		operationID: start.operationID,
		metadata:    metadata,
		provider:    start.provider,
		modelID:     start.modelID,
		rootSpan:    l.startSpan(ctx, start.operationID, attributes),
	})
}

// OnGenerateSpeechEnd records the generated audio and ends the root span.
func (l *Legacy) OnGenerateSpeechEnd(_ context.Context, event ai.TelemetryEvent[ai.GenerateSpeechEndEvent]) {
	end := event.Event

	l.endAudioOperation(audioEnd{
		callID: end.CallID,
		outputAudio: &audioDescription{
			byteLength: end.AudioByteLength,
			mediaType:  end.AudioMediaType,
			format:     end.AudioFormat,
		},
		usage:            end.Usage,
		providerMetadata: end.ProviderMetadata,
	})
}

// OnTranscriptionEnd records the transcript and ends the root span.
func (l *Legacy) OnTranscriptionEnd(_ context.Context, event ai.TelemetryEvent[ai.TranscriptionEndEvent]) {
	l.endAudioOperation(transcriptionAudioEnd(event.Event))
}

// OnStreamTranscriptionEnd records the transcript and ends the root span.
func (l *Legacy) OnStreamTranscriptionEnd(_ context.Context, event ai.TelemetryEvent[ai.TranscriptionEndEvent]) {
	l.endAudioOperation(transcriptionAudioEnd(event.Event))
}

func (l *Legacy) endAudioOperation(end audioEnd) {
	l.mu.Lock()
	defer l.mu.Unlock()

	state := l.calls[end.callID]
	if state == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)

	if end.inputAudio != nil {
		setInputAudio(attributes, &end.inputAudio.byteLength, end.inputAudio.mediaType, "ai.request.audio.mediaType")
	}

	if end.outputText != nil {
		text := *end.outputText

		attributes.setOutput("ai.response.text", func() attribute.Value {
			return attribute.StringValue(text)
		})
	}

	setSourceText(attributes, end.sourceText, "ai.response.sourceText")

	if end.outputAudio != nil {
		setOutputAudio(attributes, *end.outputAudio, "ai.response.audio.mediaType")
	}

	attributes.add(providerUsageAttributes(end.usage, "ai.usage")...)

	if len(end.usage) > 0 {
		attributes.setJSON("ai.response.usage", end.usage)
	}

	setLegacyProviderMetadata(attributes, end.providerMetadata)

	state.rootSpan.SetAttributes(attributes.list()...)
	state.rootSpan.End()
	delete(l.calls, end.callID)
}
