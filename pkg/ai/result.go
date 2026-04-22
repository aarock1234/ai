package ai

import (
	"context"
	"iter"
	"strings"
	"sync"

	"github.com/aarock1234/ai/pkg/model"
)

// File is a file emitted by the model (image, audio, document). Either
// Data or URL is populated, mirroring what the provider returned.
type File struct {
	Data     []byte
	URL      string
	MIMEType string
	Filename string
	Metadata model.ProviderMetadata
}

// Source is a reference used to generate the response (web search
// citation, document excerpt).
type Source struct {
	SourceType model.SourceType
	ID         string
	URL        string
	Title      string
	MIMEType   string
	Filename   string
	StartIndex int
	EndIndex   int
	Metadata   model.ProviderMetadata
}

// Reasoning is a reasoning block emitted by the model. Providers that
// support multi-block reasoning (Anthropic extended thinking,
// OpenRouter reasoning_details) may produce more than one per step.
type Reasoning struct {
	Text      string
	Signature string
	ID        string
	Format    string
	Redacted  bool
	Metadata  model.ProviderMetadata
}

// GenerateResult is the result of a GenerateText call.
type GenerateResult struct {
	// Text is the generated text content, concatenated from all text
	// content segments.
	Text string

	// Reasoning contains reasoning blocks emitted by the model for the
	// final step, in arrival order.
	Reasoning []Reasoning

	// Files contains file outputs emitted by the model for the final
	// step (generated images, audio, etc.).
	Files []File

	// Sources contains source citations emitted by the model for the
	// final step (e.g. web search citations).
	Sources []Source

	// ToolCalls contains any tool invocations requested by the model.
	ToolCalls []ToolCall

	// ToolResults contains the results of executed tools (populated
	// when using the automatic tool loop).
	ToolResults []ToolResult

	// FinishReason indicates why the model stopped generating.
	FinishReason model.FinishReason

	// RawFinishReason is the provider-native finish reason string.
	RawFinishReason string

	// Usage tracks total token consumption across all steps.
	Usage Usage

	// Metadata carries provider/model response metadata for the final
	// generation step.
	Metadata model.ProviderMetadata

	// Warnings carries provider warnings when available.
	Warnings []string

	// Steps contains the individual generation steps when using the
	// tool loop (WithMaxSteps > 1).
	Steps []Step
}

// ReasoningText returns the concatenated reasoning text across all
// blocks in the final step. Redacted blocks contribute nothing.
func (r *GenerateResult) ReasoningText() string {
	if r == nil {
		return ""
	}

	return concatReasoning(r.Reasoning)
}

// Step records a single generation step in a multi-step tool loop.
type Step struct {
	Text            string
	Reasoning       []Reasoning
	Files           []File
	Sources         []Source
	ToolCalls       []ToolCall
	ToolResults     []ToolResult
	FinishReason    model.FinishReason
	RawFinishReason string
	Usage           Usage
	Metadata        model.ProviderMetadata
	Warnings        []string
}

// ReasoningText returns the concatenated reasoning text across all
// blocks in this step.
func (s Step) ReasoningText() string {
	return concatReasoning(s.Reasoning)
}

// toolLoopState carries multi-step accumulation data shared between
// the tool-loop stream iterator and the StreamResult that wraps it.
// The stream function populates steps as they complete; the
// StreamResult uses this data to build a multi-step GenerateResult.
type toolLoopState struct {
	steps []Step
	usage Usage
	err   error
}

// recordStep appends a completed step and aggregates its usage.
func (s *toolLoopState) recordStep(step Step) {
	s.steps = append(s.steps, step)
	addUsage(&s.usage, step.Usage)
}

// StreamResult wraps a streaming response and provides iterators for
// consuming the stream. It accumulates chunks internally so that
// Result() can return the final GenerateResult after the stream is
// fully consumed.
type StreamResult struct {
	stream   iter.Seq[model.Chunk]
	cancel   context.CancelFunc
	metadata model.ProviderMetadata
	onFinish []finishHook
	toolLoop *toolLoopState

	mu       sync.Mutex
	consumed bool
	final    *GenerateResult
	err      error
}

// newStreamResult creates a StreamResult from a model chunk iterator.
// If cancel is non-nil it is called when the stream is fully consumed.
// When toolLoop is non-nil the result is built from the multi-step
// state instead of the single-stream accumulators.
func newStreamResult(stream iter.Seq[model.Chunk], metadata model.ProviderMetadata, cancel context.CancelFunc, onFinish []finishHook, toolLoop *toolLoopState) *StreamResult {
	return &StreamResult{
		stream:   stream,
		cancel:   cancel,
		metadata: metadata.Clone(),
		onFinish: append([]finishHook(nil), onFinish...),
		toolLoop: toolLoop,
	}
}

// done is called when the stream is fully consumed or the caller
// breaks out of the iteration. It fires the context cancel function
// (if set). CancelFunc is idempotent, so multiple calls are safe.
func (s *StreamResult) done() {
	if s.cancel != nil {
		s.cancel()
	}
}

// Text returns an iterator that yields text deltas as they arrive
// from the model. This is the primary way to consume a streaming
// response.
func (s *StreamResult) Text() iter.Seq[string] {
	return func(yield func(string) bool) {
		s.consume(func(chunk model.Chunk) bool {
			text, ok := chunk.(model.TextDelta)
			if !ok {
				return true
			}

			return yield(text.Text)
		})
	}
}

// Reasoning returns an iterator that yields reasoning text fragments
// as they arrive from the model. Use Chunks() if you need the full
// reasoning block metadata.
func (s *StreamResult) Reasoning() iter.Seq[string] {
	return func(yield func(string) bool) {
		s.consume(func(chunk model.Chunk) bool {
			delta, ok := chunk.(model.ReasoningDelta)
			if !ok || delta.Text == "" {
				return true
			}

			return yield(delta.Text)
		})
	}
}

// Files returns an iterator that yields complete files as the model
// emits them (generated images, audio).
func (s *StreamResult) Files() iter.Seq[File] {
	return func(yield func(File) bool) {
		s.consume(func(chunk model.Chunk) bool {
			fc, ok := chunk.(model.FileChunk)
			if !ok {
				return true
			}

			return yield(fileFromChunk(fc))
		})
	}
}

// Sources returns an iterator that yields source citations as the
// model emits them.
func (s *StreamResult) Sources() iter.Seq[Source] {
	return func(yield func(Source) bool) {
		s.consume(func(chunk model.Chunk) bool {
			sc, ok := chunk.(model.SourceChunk)
			if !ok {
				return true
			}

			return yield(sourceFromChunk(sc))
		})
	}
}

// Chunks returns an iterator that yields raw model chunks. Use this
// when you need access to all chunk types (text, tool calls, finish,
// usage) rather than just text deltas. The context cancel function
// (if set via WithTimeout) is called when the iterator finishes.
func (s *StreamResult) Chunks() iter.Seq[model.Chunk] {
	return func(yield func(model.Chunk) bool) {
		s.consume(yield)
	}
}

// Metadata returns the best-known provider/model metadata for the stream.
// This includes metadata available at stream creation time and any later
// metadata chunks observed while consuming the stream.
func (s *StreamResult) Metadata() model.ProviderMetadata {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.metadata.Clone()
}

// Result blocks until the stream is fully consumed and returns the
// accumulated GenerateResult. If Text() or Chunks() has not been fully
// iterated, Result() consumes the remainder.
func (s *StreamResult) Result() (*GenerateResult, error) {
	s.mu.Lock()
	if s.consumed {
		s.mu.Unlock()

		return s.final, s.err
	}
	s.mu.Unlock()

	// Consume remaining stream.
	s.consume(func(model.Chunk) bool {
		return true
	})

	s.mu.Lock()
	defer s.mu.Unlock()

	return s.final, s.err
}

// streamAccumulator builds up a Step from a single provider stream.
// Kept as a struct so the single-step and tool-loop consumers share
// the same accumulation logic.
type streamAccumulator struct {
	text            strings.Builder
	reasoning       []Reasoning
	files           []File
	sources         []Source
	toolCalls       []ToolCall
	toolCallArgs    map[string]*strings.Builder
	toolCallNames   map[string]string
	finishReason    model.FinishReason
	rawFinishReason string
	usage           Usage
	warnings        []string

	// activeReasoningIdx is the index in reasoning whose text is still
	// growing. -1 means no block is active. Tracking by index instead
	// of pointer avoids dangling references when the slice is
	// reallocated by append.
	activeReasoningIdx int
}

func newStreamAccumulator() *streamAccumulator {
	return &streamAccumulator{
		toolCallArgs:       make(map[string]*strings.Builder),
		toolCallNames:      make(map[string]string),
		activeReasoningIdx: -1,
	}
}

func (a *streamAccumulator) observe(chunk model.Chunk) {
	switch c := chunk.(type) {
	case model.StreamStartChunk:
		a.warnings = append(a.warnings, c.Warnings...)
	case model.TextDelta:
		a.text.WriteString(c.Text)
	case model.ReasoningDelta:
		a.appendReasoning(c)
	case model.ToolCallDelta:
		if _, ok := a.toolCallArgs[c.ToolCallID]; !ok {
			a.toolCallArgs[c.ToolCallID] = &strings.Builder{}
		}
		a.toolCallArgs[c.ToolCallID].WriteString(c.ArgsJSON)
		if c.ToolName != "" {
			a.toolCallNames[c.ToolCallID] = c.ToolName
		}
	case model.FileChunk:
		a.files = append(a.files, fileFromChunk(c))
	case model.SourceChunk:
		a.sources = append(a.sources, sourceFromChunk(c))
	case model.FinishChunk:
		a.finishReason = c.FinishReason
		a.rawFinishReason = c.RawFinishReason
		if a.rawFinishReason == "" {
			a.rawFinishReason = string(c.FinishReason)
		}
	case model.UsageChunk:
		a.usage = c.Usage
	case model.ErrorChunk:
		if c.Err != nil {
			a.warnings = append(a.warnings, c.Err.Error())
		}
	}
}

// appendReasoning merges consecutive reasoning deltas that belong to
// the same block, while still allowing multiple distinct blocks per
// step. The provider emits a new block when ID changes (or when a
// redacted block appears).
func (a *streamAccumulator) appendReasoning(delta model.ReasoningDelta) {
	if a.activeReasoningIdx >= 0 && !delta.Redacted {
		active := &a.reasoning[a.activeReasoningIdx]
		if delta.ID == active.ID && delta.Format == active.Format {
			active.Text += delta.Text
			if delta.Signature != "" {
				active.Signature = delta.Signature
			}

			return
		}
	}

	a.reasoning = append(a.reasoning, Reasoning{
		Text:      delta.Text,
		Signature: delta.Signature,
		ID:        delta.ID,
		Format:    delta.Format,
		Redacted:  delta.Redacted,
	})
	a.activeReasoningIdx = len(a.reasoning) - 1
}

func (a *streamAccumulator) buildStep(metadata model.ProviderMetadata) Step {
	for id, args := range a.toolCallArgs {
		a.toolCalls = append(a.toolCalls, ToolCall{
			ToolCallID: id,
			ToolName:   a.toolCallNames[id],
			Args:       []byte(args.String()),
		})
	}

	return Step{
		Text:            a.text.String(),
		Reasoning:       a.reasoning,
		Files:           a.files,
		Sources:         a.sources,
		ToolCalls:       a.toolCalls,
		FinishReason:    a.finishReason,
		RawFinishReason: a.rawFinishReason,
		Usage:           a.usage,
		Metadata:        metadata,
		Warnings:        a.warnings,
	}
}

func (s *StreamResult) consume(yield func(model.Chunk) bool) {
	acc := newStreamAccumulator()

	defer func() {
		var result *GenerateResult

		if s.toolLoop != nil {
			result = buildResult(s.toolLoop.steps, s.toolLoop.usage)
			result.Metadata = s.Metadata()
		} else {
			// Single-step path leaves Steps empty to preserve the
			// historical shape where Steps is only populated for the
			// tool loop.
			result = resultFromStep(acc.buildStep(s.Metadata()))
		}

		s.mu.Lock()
		s.consumed = true
		s.final = result
		if s.toolLoop != nil && s.toolLoop.err != nil {
			s.err = s.toolLoop.err
		}
		s.mu.Unlock()

		if err := runFinishHooks(s.onFinish, result); err != nil {
			s.mu.Lock()
			s.err = err
			s.mu.Unlock()
		}

		s.done()
	}()

	for chunk := range s.stream {
		acc.observe(chunk)
		if meta, ok := chunk.(model.ProviderMetadataChunk); ok {
			s.mu.Lock()
			s.metadata = meta.Metadata.Clone()
			s.mu.Unlock()
		}

		if !yield(chunk) {
			return
		}
	}
}

// resultFromStep builds a single-step GenerateResult from the completed
// step. It leaves Steps empty and uses the step's own usage as the
// total; the tool-loop path calls buildResult instead.
func resultFromStep(step Step) *GenerateResult {
	return &GenerateResult{
		Text:            step.Text,
		Reasoning:       step.Reasoning,
		Files:           step.Files,
		Sources:         step.Sources,
		ToolCalls:       step.ToolCalls,
		FinishReason:    step.FinishReason,
		RawFinishReason: step.RawFinishReason,
		Usage:           step.Usage,
		Metadata:        step.Metadata,
		Warnings:        step.Warnings,
	}
}

// fileFromChunk maps a streamed FileChunk to the consumer File type.
// The types share a field layout, so the struct conversion is sound;
// metadata is cloned to avoid aliasing into the chunk.
func fileFromChunk(c model.FileChunk) File {
	file := File(c)
	file.Metadata = c.Metadata.Clone()

	return file
}

// fileFromContent maps a non-streaming FileContent to the consumer File.
func fileFromContent(c model.FileContent) File {
	file := File(c)
	file.Metadata = c.Metadata.Clone()

	return file
}

// sourceFromChunk maps a streamed SourceChunk to the consumer Source.
func sourceFromChunk(c model.SourceChunk) Source {
	source := Source(c)
	source.Metadata = c.Metadata.Clone()

	return source
}

// sourceFromContent maps a non-streaming SourceContent to the consumer Source.
func sourceFromContent(c model.SourceContent) Source {
	source := Source(c)
	source.Metadata = c.Metadata.Clone()

	return source
}

// reasoningFromContent maps a non-streaming ReasoningContent to the
// consumer Reasoning.
func reasoningFromContent(c model.ReasoningContent) Reasoning {
	reasoning := Reasoning(c)
	reasoning.Metadata = c.Metadata.Clone()

	return reasoning
}

func concatReasoning(blocks []Reasoning) string {
	var b strings.Builder
	for i, block := range blocks {
		if block.Redacted || block.Text == "" {
			continue
		}
		if i > 0 && b.Len() > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(block.Text)
	}

	return b.String()
}

// addUsage accumulates step usage into a running total.
func addUsage(total *Usage, step Usage) {
	total.InputTokens += step.InputTokens
	total.OutputTokens += step.OutputTokens
	total.CachedInputTokens += step.CachedInputTokens
	total.ReasoningTokens += step.ReasoningTokens
	total.ImageInputTokens += step.ImageInputTokens
	total.ImageOutputTokens += step.ImageOutputTokens
	total.AudioInputTokens += step.AudioInputTokens
	total.AudioOutputTokens += step.AudioOutputTokens
}

// ObjectResult is the result of a GenerateObject call.
type ObjectResult[T any] struct {
	// Object is the generated structured output, unmarshaled into T.
	Object T

	// RawJSON is the raw JSON payload used to unmarshal Object.
	RawJSON string

	// Usage tracks token consumption for the request.
	Usage Usage

	// Metadata carries provider/model response metadata for the object
	// generation request.
	Metadata model.ProviderMetadata
}
