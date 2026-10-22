package ai

import (
	"context"
)

// EmbedTelemetry receives the lifecycle events of Embed and EmbedMany.
type EmbedTelemetry interface {
	// OnEmbedStart is called when Embed or EmbedMany begins.
	OnEmbedStart(ctx context.Context, event TelemetryEvent[EmbedStartEvent])

	// OnEmbedEnd is called when Embed or EmbedMany completes.
	OnEmbedEnd(ctx context.Context, event TelemetryEvent[EmbedEndEvent])

	// OnEmbeddingModelCallStart is called before each embedding model call
	// attempt. EmbedMany makes one call per batch of values, concurrently
	// when the model supports parallel calls.
	OnEmbeddingModelCallStart(ctx context.Context, event TelemetryEvent[EmbeddingModelCallStartEvent])

	// OnEmbeddingModelCallEnd is called after each embedding model call
	// succeeded.
	OnEmbeddingModelCallEnd(ctx context.Context, event TelemetryEvent[EmbeddingModelCallEndEvent])
}

// RerankTelemetry receives the lifecycle events of Rerank.
type RerankTelemetry interface {
	// OnRerankStart is called when Rerank begins.
	OnRerankStart(ctx context.Context, event TelemetryEvent[RerankStartEvent[any]])

	// OnRerankEnd is called when Rerank completes.
	OnRerankEnd(ctx context.Context, event TelemetryEvent[RerankEndEvent[any]])

	// OnRerankingModelCallStart is called before each reranking model call
	// attempt.
	OnRerankingModelCallStart(ctx context.Context, event TelemetryEvent[RerankingModelCallStartEvent])

	// OnRerankingModelCallEnd is called after each reranking model call
	// succeeded.
	OnRerankingModelCallEnd(ctx context.Context, event TelemetryEvent[RerankingModelCallEndEvent])
}

// SpeechTelemetry receives the lifecycle events of GenerateSpeech.
type SpeechTelemetry interface {
	// OnGenerateSpeechStart is called when GenerateSpeech begins.
	OnGenerateSpeechStart(ctx context.Context, event TelemetryEvent[GenerateSpeechStartEvent])

	// OnGenerateSpeechEnd is called when GenerateSpeech completes.
	OnGenerateSpeechEnd(ctx context.Context, event TelemetryEvent[GenerateSpeechEndEvent])
}

// TranscriptionTelemetry receives the lifecycle events of Transcribe and
// StreamTranscribe.
type TranscriptionTelemetry interface {
	// OnTranscriptionStart is called when Transcribe begins.
	OnTranscriptionStart(ctx context.Context, event TelemetryEvent[TranscriptionStartEvent])

	// OnTranscriptionEnd is called when Transcribe completes.
	OnTranscriptionEnd(ctx context.Context, event TelemetryEvent[TranscriptionEndEvent])

	// OnStreamTranscriptionStart is called when a StreamTranscribe stream
	// starts. The streaming transcription API is experimental.
	OnStreamTranscriptionStart(ctx context.Context, event TelemetryEvent[TranscriptionStartEvent])

	// OnStreamTranscriptionEnd is called when a StreamTranscribe stream
	// finishes with a transcript.
	OnStreamTranscriptionEnd(ctx context.Context, event TelemetryEvent[TranscriptionEndEvent])
}

// OnEmbedStart does nothing.
func (NopTelemetry) OnEmbedStart(context.Context, TelemetryEvent[EmbedStartEvent]) {}

// OnEmbedEnd does nothing.
func (NopTelemetry) OnEmbedEnd(context.Context, TelemetryEvent[EmbedEndEvent]) {}

// OnEmbeddingModelCallStart does nothing.
func (NopTelemetry) OnEmbeddingModelCallStart(context.Context, TelemetryEvent[EmbeddingModelCallStartEvent]) {
}

// OnEmbeddingModelCallEnd does nothing.
func (NopTelemetry) OnEmbeddingModelCallEnd(context.Context, TelemetryEvent[EmbeddingModelCallEndEvent]) {
}

// OnRerankStart does nothing.
func (NopTelemetry) OnRerankStart(context.Context, TelemetryEvent[RerankStartEvent[any]]) {}

// OnRerankEnd does nothing.
func (NopTelemetry) OnRerankEnd(context.Context, TelemetryEvent[RerankEndEvent[any]]) {}

// OnRerankingModelCallStart does nothing.
func (NopTelemetry) OnRerankingModelCallStart(context.Context, TelemetryEvent[RerankingModelCallStartEvent]) {
}

// OnRerankingModelCallEnd does nothing.
func (NopTelemetry) OnRerankingModelCallEnd(context.Context, TelemetryEvent[RerankingModelCallEndEvent]) {
}

// OnGenerateSpeechStart does nothing.
func (NopTelemetry) OnGenerateSpeechStart(context.Context, TelemetryEvent[GenerateSpeechStartEvent]) {
}

// OnGenerateSpeechEnd does nothing.
func (NopTelemetry) OnGenerateSpeechEnd(context.Context, TelemetryEvent[GenerateSpeechEndEvent]) {}

// OnTranscriptionStart does nothing.
func (NopTelemetry) OnTranscriptionStart(context.Context, TelemetryEvent[TranscriptionStartEvent]) {}

// OnTranscriptionEnd does nothing.
func (NopTelemetry) OnTranscriptionEnd(context.Context, TelemetryEvent[TranscriptionEndEvent]) {}

// OnStreamTranscriptionStart does nothing.
func (NopTelemetry) OnStreamTranscriptionStart(context.Context, TelemetryEvent[TranscriptionStartEvent]) {
}

// OnStreamTranscriptionEnd does nothing.
func (NopTelemetry) OnStreamTranscriptionEnd(context.Context, TelemetryEvent[TranscriptionEndEvent]) {
}

func (d telemetryDispatcher) onEmbedStart(ctx context.Context, event EmbedStartEvent) {
	notifyTelemetry(ctx, d, event, EmbedTelemetry.OnEmbedStart)
}

func (d telemetryDispatcher) onEmbedEnd(ctx context.Context, event EmbedEndEvent) {
	notifyTelemetry(ctx, d, event, EmbedTelemetry.OnEmbedEnd)
}

func (d telemetryDispatcher) onEmbeddingModelCallStart(ctx context.Context, event EmbeddingModelCallStartEvent) {
	notifyTelemetry(ctx, d, event, EmbedTelemetry.OnEmbeddingModelCallStart)
}

func (d telemetryDispatcher) onEmbeddingModelCallEnd(ctx context.Context, event EmbeddingModelCallEndEvent) {
	notifyTelemetry(ctx, d, event, EmbedTelemetry.OnEmbeddingModelCallEnd)
}

func (d telemetryDispatcher) onRerankStart(ctx context.Context, event RerankStartEvent[any]) {
	notifyTelemetry(ctx, d, event, RerankTelemetry.OnRerankStart)
}

func (d telemetryDispatcher) onRerankEnd(ctx context.Context, event RerankEndEvent[any]) {
	notifyTelemetry(ctx, d, event, RerankTelemetry.OnRerankEnd)
}

func (d telemetryDispatcher) onRerankingModelCallStart(ctx context.Context, event RerankingModelCallStartEvent) {
	notifyTelemetry(ctx, d, event, RerankTelemetry.OnRerankingModelCallStart)
}

func (d telemetryDispatcher) onRerankingModelCallEnd(ctx context.Context, event RerankingModelCallEndEvent) {
	notifyTelemetry(ctx, d, event, RerankTelemetry.OnRerankingModelCallEnd)
}

func (d telemetryDispatcher) onGenerateSpeechStart(ctx context.Context, event GenerateSpeechStartEvent) {
	notifyTelemetry(ctx, d, event, SpeechTelemetry.OnGenerateSpeechStart)
}

func (d telemetryDispatcher) onGenerateSpeechEnd(ctx context.Context, event GenerateSpeechEndEvent) {
	notifyTelemetry(ctx, d, event, SpeechTelemetry.OnGenerateSpeechEnd)
}

func (d telemetryDispatcher) onTranscriptionStart(ctx context.Context, event TranscriptionStartEvent) {
	notifyTelemetry(ctx, d, event, TranscriptionTelemetry.OnTranscriptionStart)
}

func (d telemetryDispatcher) onTranscriptionEnd(ctx context.Context, event TranscriptionEndEvent) {
	notifyTelemetry(ctx, d, event, TranscriptionTelemetry.OnTranscriptionEnd)
}

func (d telemetryDispatcher) onStreamTranscriptionStart(ctx context.Context, event TranscriptionStartEvent) {
	notifyTelemetry(ctx, d, event, TranscriptionTelemetry.OnStreamTranscriptionStart)
}

func (d telemetryDispatcher) onStreamTranscriptionEnd(ctx context.Context, event TranscriptionEndEvent) {
	notifyTelemetry(ctx, d, event, TranscriptionTelemetry.OnStreamTranscriptionEnd)
}
