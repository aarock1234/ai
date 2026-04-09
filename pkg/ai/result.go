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

	// ToolCalls contains any tool invocations requested by the model.
	ToolCalls []ToolCall

	// ToolResults contains the results of executed tools (populated
	// when using the automatic tool loop).
	ToolResults []ToolResult

	// FinishReason indicates why the model stopped generating.
	FinishReason model.FinishReason

	// Usage tracks total token consumption across all steps.
	Usage Usage

	// Metadata carries provider/model response metadata for the final
	// generation step.
	Metadata model.ProviderMetadata

	// Steps contains the individual generation steps when using the
	// tool loop (WithMaxSteps > 1).
	Steps []Step
}

// Step records a single generation step in a multi-step tool loop.
type Step struct {
	Text         string
	ToolCalls    []ToolCall
	ToolResults  []ToolResult
	FinishReason model.FinishReason
	Usage        Usage
	Metadata     model.ProviderMetadata
}

// StreamResult wraps a streaming response and provides iterators for
// consuming the stream. It accumulates chunks internally so that
// Result() can return the final GenerateResult after the stream is
// fully consumed.
type StreamResult struct {
	stream   iter.Seq[model.Chunk]
	cancel   context.CancelFunc
	metadata model.ProviderMetadata

	mu       sync.Mutex
	consumed bool
	final    *GenerateResult
	err      error
}

// newStreamResult creates a StreamResult from a model chunk iterator.
// If cancel is non-nil it is called when the stream is fully consumed.
func newStreamResult(stream iter.Seq[model.Chunk], metadata model.ProviderMetadata, cancel context.CancelFunc) *StreamResult {
	return &StreamResult{
		stream:   stream,
		cancel:   cancel,
		metadata: metadata.Clone(),
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
		var textBuilder strings.Builder
		var toolCalls []ToolCall
		var finishReason model.FinishReason
		var usage Usage

		// Accumulate tool call deltas by ID.
		toolCallArgs := make(map[string]*strings.Builder)
		toolCallNames := make(map[string]string)

		for chunk := range s.stream {
			switch c := chunk.(type) {
			case model.TextDelta:
				textBuilder.WriteString(c.Text)
				if !yield(c.Text) {
					return
				}
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
			case model.UsageChunk:
				usage = c.Usage
			case model.ProviderMetadataChunk:
				s.mu.Lock()
				s.metadata = c.Metadata.Clone()
				s.mu.Unlock()
			}
		}

		// Build tool calls from accumulated deltas.
		for id, args := range toolCallArgs {
			toolCalls = append(toolCalls, ToolCall{
				ToolCallID: id,
				ToolName:   toolCallNames[id],
				Args:       []byte(args.String()),
			})
		}

		s.mu.Lock()
		s.consumed = true
		s.final = &GenerateResult{
			Text:         textBuilder.String(),
			ToolCalls:    toolCalls,
			FinishReason: finishReason,
			Usage:        usage,
			Metadata:     s.metadata.Clone(),
		}
		s.mu.Unlock()

		s.done()
	}
}

// Chunks returns an iterator that yields raw model chunks. Use this
// when you need access to all chunk types (text, tool calls, finish,
// usage) rather than just text deltas. The context cancel function
// (if set via WithTimeout) is called when the iterator finishes.
func (s *StreamResult) Chunks() iter.Seq[model.Chunk] {
	return func(yield func(model.Chunk) bool) {
		defer s.done()

		for chunk := range s.stream {
			if !yield(chunk) {
				return
			}
		}
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
	s.Text()(func(string) bool {
		return true
	})

	s.mu.Lock()
	defer s.mu.Unlock()

	return s.final, s.err
}

// ObjectResult is the result of a GenerateObject call.
type ObjectResult[T any] struct {
	// Object is the generated structured output, unmarshaled into T.
	Object T

	// Usage tracks token consumption for the request.
	Usage Usage

	// Metadata carries provider/model response metadata for the object
	// generation request.
	Metadata model.ProviderMetadata
}
