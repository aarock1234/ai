package otel

import (
	"context"
	"encoding/json"
	"net/http"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// OnEmbedStart starts the embeddings span of Embed or EmbedMany.
func (t *OpenTelemetry) OnEmbedStart(ctx context.Context, event ai.TelemetryEvent[ai.EmbedStartEvent]) {
	start := event.Event
	operation := operationName(start.OperationID)
	baseAttributes := t.baseAttributes(event.TelemetryMetadata, start.Headers)

	attributes := newSpanAttributes(event.TelemetryMetadata)
	attributes.setString("gen_ai.operation.name", operation)
	attributes.setString("gen_ai.provider.name", providerName(start.Provider))
	attributes.setString("gen_ai.request.model", start.ModelID)
	attributes.add(baseAttributes...)

	if t.options.Embedding {
		setEmbeddingValues(attributes, start.OperationID, start.Values)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	rootSpan := t.startSpan(ctx, spanStart{
		name:       operation + " " + start.ModelID,
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

// setEmbeddingValues records the embedded values: one JSON string per value
// for EmbedMany, and the JSON of the single value for Embed.
func setEmbeddingValues(attributes *spanAttributes, operationID string, values []string) {
	if operationID == operationEmbedMany {
		attributes.setInput("ai.values", func() attribute.Value {
			return jsonListValue(values)
		})

		return
	}

	if len(values) > 0 {
		attributes.setInput("ai.value", func() attribute.Value {
			return jsonValue(values[0])
		})
	}
}

// setEmbeddingOutputs records the embeddings: one JSON string per
// embedding for EmbedMany, and the JSON of the single embedding for Embed.
func setEmbeddingOutputs(attributes *spanAttributes, operationID string, embeddings [][]float64) {
	if operationID == operationEmbedMany {
		attributes.setOutput("ai.embeddings", func() attribute.Value {
			return jsonListValue(embeddings)
		})

		return
	}

	if len(embeddings) > 0 {
		attributes.setOutput("ai.embedding", func() attribute.Value {
			return jsonValue(embeddings[0])
		})
	}
}

// OnEmbedEnd records the embeddings and ends the operation span.
func (t *OpenTelemetry) OnEmbedEnd(_ context.Context, event ai.TelemetryEvent[ai.EmbedEndEvent]) {
	end := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[end.CallID]
	if state == nil {
		return
	}

	if t.options.Embedding {
		attributes := newSpanAttributes(state.metadata)
		setEmbeddingOutputs(attributes, state.operationID, end.Embeddings)
		state.rootSpan.SetAttributes(attributes.list()...)
	}

	state.rootSpan.End()
	delete(t.calls, end.CallID)
}

// OnEmbeddingModelCallStart starts the span of one embedding model call.
func (t *OpenTelemetry) OnEmbeddingModelCallStart(ctx context.Context, event ai.TelemetryEvent[ai.EmbeddingModelCallStartEvent]) {
	call := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[call.CallID]
	if state == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	attributes.setString("gen_ai.operation.name", "embeddings")
	attributes.setString("gen_ai.provider.name", providerName(state.provider))
	attributes.setString("gen_ai.request.model", state.modelID)
	attributes.add(state.baseAttributes...)

	if t.options.Embedding {
		attributes.setInput("ai.values", func() attribute.Value {
			return jsonListValue(call.Values)
		})
	}

	state.embedSpans[call.EmbedCallID] = t.startSpan(trace.ContextWithSpan(ctx, state.rootSpan), spanStart{
		name:       "embeddings " + state.modelID,
		kind:       trace.SpanKindClient,
		spanType:   SpanTypeEmbedding,
		callID:     call.CallID,
		operation:  state.operationID,
		attributes: attributes.list(),
	})
}

// OnEmbeddingModelCallEnd records the usage and embeddings of one
// embedding model call and ends its span.
func (t *OpenTelemetry) OnEmbeddingModelCallEnd(_ context.Context, event ai.TelemetryEvent[ai.EmbeddingModelCallEndEvent]) {
	end := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[end.CallID]
	if state == nil {
		return
	}

	span := state.embedSpans[end.EmbedCallID]
	if span == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	setEmbeddingTokens(attributes, "gen_ai.usage.input_tokens", end.Usage)

	if t.options.Embedding {
		attributes.setOutput("ai.embeddings", func() attribute.Value {
			return jsonListValue(end.Embeddings)
		})
	}

	span.SetAttributes(attributes.list()...)
	span.End()
	delete(state.embedSpans, end.EmbedCallID)
}

// setEmbeddingTokens records the tokens of an embedding usage, if any.
func setEmbeddingTokens(attributes *spanAttributes, key string, usage *model.EmbeddingUsage) {
	if usage == nil {
		return
	}

	attributes.setInt(key, &usage.Tokens)
}

// OnRerankStart starts the rerank span of Rerank.
func (t *OpenTelemetry) OnRerankStart(ctx context.Context, event ai.TelemetryEvent[ai.RerankStartEvent[any]]) {
	start := event.Event
	baseAttributes := t.baseAttributes(event.TelemetryMetadata, start.Headers)

	attributes := newSpanAttributes(event.TelemetryMetadata)
	attributes.setString("gen_ai.operation.name", "rerank")
	attributes.setString("gen_ai.provider.name", providerName(start.Provider))
	attributes.setString("gen_ai.request.model", start.ModelID)
	attributes.add(baseAttributes...)

	if t.options.Reranking {
		setRerankDocuments(attributes, start.Documents)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	rootSpan := t.startSpan(ctx, spanStart{
		name:       "rerank " + start.ModelID,
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

func setRerankDocuments(attributes *spanAttributes, documents []any) {
	attributes.setInput("ai.documents", func() attribute.Value {
		return jsonListValue(documents)
	})
}

// OnRerankEnd ends the operation span of Rerank.
func (t *OpenTelemetry) OnRerankEnd(_ context.Context, event ai.TelemetryEvent[ai.RerankEndEvent[any]]) {
	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[event.Event.CallID]
	if state == nil {
		return
	}

	state.rootSpan.End()
	delete(t.calls, event.Event.CallID)
}

// OnRerankingModelCallStart starts the span of the reranking model call.
func (t *OpenTelemetry) OnRerankingModelCallStart(ctx context.Context, event ai.TelemetryEvent[ai.RerankingModelCallStartEvent]) {
	call := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[call.CallID]
	if state == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)
	attributes.setString("gen_ai.operation.name", "rerank")
	attributes.setString("gen_ai.provider.name", providerName(state.provider))
	attributes.setString("gen_ai.request.model", state.modelID)
	attributes.add(state.baseAttributes...)

	if t.options.Reranking {
		setRerankDocuments(attributes, call.Documents)
	}

	state.rerankSpan = t.startSpan(trace.ContextWithSpan(ctx, state.rootSpan), spanStart{
		name:       "rerank " + state.modelID,
		kind:       trace.SpanKindClient,
		spanType:   SpanTypeReranking,
		callID:     call.CallID,
		operation:  state.operationID,
		attributes: attributes.list(),
	})
}

// OnRerankingModelCallEnd records the ranking and ends the reranking model
// call span.
func (t *OpenTelemetry) OnRerankingModelCallEnd(_ context.Context, event ai.TelemetryEvent[ai.RerankingModelCallEndEvent]) {
	end := event.Event

	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[end.CallID]
	if state == nil || state.rerankSpan == nil {
		return
	}

	if t.options.Reranking {
		attributes := newSpanAttributes(state.metadata)
		setRanking(attributes, end)
		state.rerankSpan.SetAttributes(attributes.list()...)
	}

	state.rerankSpan.End()
	state.rerankSpan = nil
}

// wireRankedDocument is one entry of a model ranking.
type wireRankedDocument struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevanceScore"`
}

// setRanking records the documents type and the model ranking.
func setRanking(attributes *spanAttributes, end ai.RerankingModelCallEndEvent) {
	attributes.setString("ai.ranking.type", end.DocumentsType)
	attributes.setOutput("ai.ranking", func() attribute.Value {
		ranking := make([]wireRankedDocument, 0, len(end.Ranking))

		for _, ranked := range end.Ranking {
			ranking = append(ranking, wireRankedDocument{
				Index:          ranked.Index,
				RelevanceScore: ranked.RelevanceScore,
			})
		}

		return jsonListValue(ranking)
	})
}

// audioStart describes the start of a speech, transcription or translation
// operation.
type audioStart struct {
	callID      string
	operationID string
	provider    string
	modelID     string
	headers     http.Header

	// text is the text to speak, for speech operations.
	text string

	// audioByteLength and audioMediaType describe the input audio of
	// transcriptions, when known.
	audioByteLength *int
	audioMediaType  string

	// targetLanguage and sourceLanguage are the languages of translations.
	targetLanguage string
	sourceLanguage string
}

// OnGenerateSpeechStart starts the operation span of GenerateSpeech.
func (t *OpenTelemetry) OnGenerateSpeechStart(ctx context.Context, event ai.TelemetryEvent[ai.GenerateSpeechStartEvent]) {
	start := event.Event

	t.startAudioOperation(ctx, event.TelemetryMetadata, audioStart{
		callID:      start.CallID,
		operationID: start.OperationID,
		provider:    start.Provider,
		modelID:     start.ModelID,
		headers:     start.Headers,
		text:        start.Text,
	})
}

// OnTranscriptionStart starts the operation span of Transcribe.
func (t *OpenTelemetry) OnTranscriptionStart(ctx context.Context, event ai.TelemetryEvent[ai.TranscriptionStartEvent]) {
	t.startAudioOperation(ctx, event.TelemetryMetadata, transcriptionAudioStart(event.Event))
}

// OnStreamTranscriptionStart starts the operation span of StreamTranscribe.
func (t *OpenTelemetry) OnStreamTranscriptionStart(ctx context.Context, event ai.TelemetryEvent[ai.TranscriptionStartEvent]) {
	t.startAudioOperation(ctx, event.TelemetryMetadata, transcriptionAudioStart(event.Event))
}

func transcriptionAudioStart(start ai.TranscriptionStartEvent) audioStart {
	return audioStart{
		callID:          start.CallID,
		operationID:     start.OperationID,
		provider:        start.Provider,
		modelID:         start.ModelID,
		headers:         start.Headers,
		audioByteLength: start.AudioByteLength,
		audioMediaType:  start.AudioMediaType,
	}
}

// genAITextMessage is a single-text message, such as the text to speak or
// a transcript.
func genAITextMessage(role model.Role, text string) []genAIMessage {
	return []genAIMessage{
		{
			Role: string(role),
			Parts: []genAIPart{
				textPart(text),
			},
		},
	}
}

func (t *OpenTelemetry) startAudioOperation(ctx context.Context, metadata ai.TelemetryMetadata, start audioStart) {
	operation := operationName(start.operationID)
	isSpeech := start.operationID == operationGenerateSpeech
	isTranslation := start.operationID == operationStreamTranslate
	isStream := start.operationID == operationStreamTranscribe || isTranslation
	baseAttributes := t.baseAttributes(metadata, start.headers)

	outputType := "text"
	if isSpeech || isTranslation {
		outputType = "speech"
	}

	attributes := newSpanAttributes(metadata)
	attributes.setString("gen_ai.operation.name", operation)
	attributes.setString("gen_ai.provider.name", providerName(start.provider))
	attributes.setString("gen_ai.request.model", start.modelID)
	attributes.setString("gen_ai.agent.name", metadata.FunctionID)
	attributes.setString("gen_ai.output.type", outputType)
	attributes.setBool("gen_ai.request.stream", isStream)
	attributes.setString("ai.request.target_language", start.targetLanguage)
	attributes.setString("ai.request.source_language", start.sourceLanguage)

	if isSpeech {
		attributes.setInput("gen_ai.input.messages", func() attribute.Value {
			return jsonValue(genAITextMessage(model.RoleUser, start.text))
		})
		attributes.setInput("ai.request.text", func() attribute.Value {
			return attribute.StringValue(start.text)
		})
	}

	setInputAudio(attributes, start.audioByteLength, start.audioMediaType, "ai.request.audio.media_type")
	attributes.add(baseAttributes...)

	t.mu.Lock()
	defer t.mu.Unlock()

	rootSpan := t.startSpan(ctx, spanStart{
		name:       operation + " " + start.modelID,
		kind:       trace.SpanKindClient,
		spanType:   SpanTypeOperation,
		callID:     start.callID,
		operation:  start.operationID,
		attributes: attributes.list(),
	})

	t.calls[start.callID] = newCallState(callStart{
		operationID:    start.operationID,
		metadata:       metadata,
		provider:       start.provider,
		modelID:        start.modelID,
		baseAttributes: baseAttributes,
		rootSpan:       rootSpan,
	})
}

// setInputAudio records the size and media type of input audio, when
// known. The media type key differs between the integrations.
func setInputAudio(attributes *spanAttributes, byteLength *int, mediaType, mediaTypeKey string) {
	if byteLength != nil {
		size := *byteLength

		attributes.setInput("ai.request.audio.size", func() attribute.Value {
			return attribute.IntValue(size)
		})
	}

	if mediaType != "" {
		attributes.setInput(mediaTypeKey, func() attribute.Value {
			return attribute.StringValue(mediaType)
		})
	}
}

// audioEnd describes the outcome of a speech, transcription or translation
// operation.
type audioEnd struct {
	callID string

	// outputAudio describes generated speech, and inputAudio the transcribed
	// audio.
	outputAudio *audioDescription
	inputAudio  *audioDescription

	// outputText is the transcript of transcriptions and the translated
	// text of translations.
	outputText *string

	// sourceText is the transcript of the source audio, for translations.
	sourceText *string

	usage            json.RawMessage
	providerMetadata model.ProviderMetadata
}

// audioDescription describes audio without its content.
type audioDescription struct {
	byteLength int
	mediaType  string
	format     string
}

// OnGenerateSpeechEnd records the generated audio and ends the operation
// span.
func (t *OpenTelemetry) OnGenerateSpeechEnd(_ context.Context, event ai.TelemetryEvent[ai.GenerateSpeechEndEvent]) {
	end := event.Event

	t.endAudioOperation(audioEnd{
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

// OnTranscriptionEnd records the transcript and ends the operation span.
func (t *OpenTelemetry) OnTranscriptionEnd(_ context.Context, event ai.TelemetryEvent[ai.TranscriptionEndEvent]) {
	t.endAudioOperation(transcriptionAudioEnd(event.Event))
}

// OnStreamTranscriptionEnd records the transcript and ends the operation
// span.
func (t *OpenTelemetry) OnStreamTranscriptionEnd(_ context.Context, event ai.TelemetryEvent[ai.TranscriptionEndEvent]) {
	t.endAudioOperation(transcriptionAudioEnd(event.Event))
}

func transcriptionAudioEnd(end ai.TranscriptionEndEvent) audioEnd {
	transcript := end.Text

	return audioEnd{
		callID: end.CallID,
		inputAudio: &audioDescription{
			byteLength: end.AudioByteLength,
			mediaType:  end.AudioMediaType,
		},
		outputText:       &transcript,
		usage:            end.Usage,
		providerMetadata: end.ProviderMetadata,
	}
}

func (t *OpenTelemetry) endAudioOperation(end audioEnd) {
	t.mu.Lock()
	defer t.mu.Unlock()

	state := t.calls[end.callID]
	if state == nil {
		return
	}

	attributes := newSpanAttributes(state.metadata)

	if end.inputAudio != nil {
		setInputAudio(attributes, &end.inputAudio.byteLength, end.inputAudio.mediaType, "ai.request.audio.media_type")
	}

	if end.outputText != nil {
		text := *end.outputText

		attributes.setOutput("ai.response.text", func() attribute.Value {
			return attribute.StringValue(text)
		})
		attributes.setOutput("gen_ai.output.messages", func() attribute.Value {
			return jsonValue(genAITextMessage(model.RoleAssistant, text))
		})
	}

	setSourceText(attributes, end.sourceText, "ai.response.source_text")

	if end.outputAudio != nil {
		setOutputAudio(attributes, *end.outputAudio, "ai.response.audio.media_type")
	}

	attributes.add(providerUsageAttributes(end.usage, "gen_ai.usage")...)

	if t.options.Usage && len(end.usage) > 0 {
		attributes.setJSON("ai.response.usage", end.usage)
	}

	if t.options.ProviderMetadata && len(end.providerMetadata) > 0 {
		attributes.setJSON("ai.response.providerMetadata", end.providerMetadata)
	}

	state.rootSpan.SetAttributes(attributes.list()...)
	state.rootSpan.End()
	delete(t.calls, end.callID)
}

// setOutputAudio records the size, media type and format of generated
// audio. The media type key differs between the integrations.
func setOutputAudio(attributes *spanAttributes, audio audioDescription, mediaTypeKey string) {
	attributes.setOutput("ai.response.audio.size", func() attribute.Value {
		return attribute.IntValue(audio.byteLength)
	})
	attributes.setOutput(mediaTypeKey, func() attribute.Value {
		return attribute.StringValue(audio.mediaType)
	})

	if audio.format != "" {
		attributes.setOutput("ai.response.audio.format", func() attribute.Value {
			return attribute.StringValue(audio.format)
		})
	}
}
