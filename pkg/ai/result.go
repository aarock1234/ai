package ai

import (
	"context"
	"iter"
	"strings"
	"sync"

	"github.com/aarock1234/ai/pkg/model"
)

// GenerateResult is the result of a GenerateText call.
type GenerateResult struct {
	// Text is the generated text content, concatenated from all text
	// content segments.
	Text string

	// Reasoning is provider or middleware-extracted reasoning text.
	Reasoning string

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

// Step records a single generation step in a multi-step tool loop.
type Step struct {
	Text            string
	Reasoning       string
	ToolCalls       []ToolCall
	ToolResults     []ToolResult
	FinishReason    model.FinishReason
	RawFinishReason string
	Usage           Usage
	Metadata        model.ProviderMetadata
	Warnings        []string
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

func (s *StreamResult) consume(yield func(model.Chunk) bool) {
	var (
		textBuilder      strings.Builder
		reasoningBuilder strings.Builder
		toolCalls        []ToolCall
		finishReason     model.FinishReason
		rawFinishReason  string
		usage            Usage
		warnings         []string
	)

	toolCallArgs := make(map[string]*strings.Builder)
	toolCallNames := make(map[string]string)

	defer func() {
		var result *GenerateResult

		if s.toolLoop != nil {
			// Multi-step tool loop: the stream function recorded all
			// steps (including the final one) into the shared state.
			// Build the result from that data.
			result = buildResult(s.toolLoop.steps, s.toolLoop.usage)
			result.Metadata = s.Metadata()
		} else {
			// Single-step: assemble tool calls from accumulated deltas
			// and build the result from stream accumulators.
			for id, args := range toolCallArgs {
				toolCalls = append(toolCalls, ToolCall{
					ToolCallID: id,
					ToolName:   toolCallNames[id],
					Args:       []byte(args.String()),
				})
			}

			result = &GenerateResult{
				Text:            textBuilder.String(),
				Reasoning:       reasoningBuilder.String(),
				ToolCalls:       toolCalls,
				FinishReason:    finishReason,
				RawFinishReason: rawFinishReason,
				Usage:           usage,
				Metadata:        s.Metadata(),
				Warnings:        warnings,
			}
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
		switch c := chunk.(type) {
		case model.TextDelta:
			textBuilder.WriteString(c.Text)
		case model.ReasoningDelta:
			reasoningBuilder.WriteString(c.Text)
		case model.ToolCallDelta:
			if _, ok := toolCallArgs[c.ToolCallID]; !ok {
				toolCallArgs[c.ToolCallID] = &strings.Builder{}
			}
			toolCallArgs[c.ToolCallID].WriteString(c.ArgsJSON)
			if c.ToolName != "" {
				toolCallNames[c.ToolCallID] = c.ToolName
			}
		case model.FinishChunk:
			finishReason = c.FinishReason
			rawFinishReason = c.RawFinishReason
			if rawFinishReason == "" {
				rawFinishReason = string(c.FinishReason)
			}
		case model.UsageChunk:
			usage = c.Usage
		case model.ProviderMetadataChunk:
			s.mu.Lock()
			s.metadata = c.Metadata.Clone()
			s.mu.Unlock()
		}

		if !yield(chunk) {
			return
		}
	}
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
