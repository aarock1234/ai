package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"
	"sync"

	"github.com/aarock1234/ai/internal/schema"
	"github.com/aarock1234/ai/pkg/model"
)

// ObjectStreamResult streams JSON deltas for a structured generation request.
type ObjectStreamResult[T any] struct {
	stream       iter.Seq[model.Chunk]
	cancel       context.CancelFunc
	metadata     model.ProviderMetadata
	useToolCalls bool
	onStepFinish []stepHook
	onFinish     []finishHook

	mu       sync.Mutex
	consumed bool
	final    *ObjectResult[T]
	err      error
}

// StreamObject sends a streaming structured generation request.
func StreamObject[T any](ctx context.Context, m model.Model, opts ...Option) (*ObjectStreamResult[T], error) {
	cfg, err := resolveConfig(opts)
	if err != nil {
		return nil, err
	}

	var cancel context.CancelFunc
	if cfg.timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, cfg.timeout)
	}

	jsonSchema := schema.Generate[T]()
	req := buildRequest(cfg)
	useToolCalls := !supportsResponseFormat(m, jsonSchema)
	if useToolCalls {
		req.Tools = append(req.Tools, model.ToolDefinition{
			Name:        "structured_output",
			Description: "Return the final response as structured JSON that matches the schema exactly.",
			InputSchema: jsonSchema,
			Strict:      true,
		})
		req.ToolChoice = &model.ToolChoice{
			Type:     "tool",
			ToolName: "structured_output",
		}
	} else {
		req.ResponseFormat = &model.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &model.JSONSchema{
				Name:   "response",
				Schema: jsonSchema,
				Strict: true,
			},
		}
	}

	for _, hook := range cfg.prepareStep {
		if err := hook(ctx, 0, req); err != nil {
			if cancel != nil {
				cancel()
			}

			return nil, fmt.Errorf("prepare step 0: %w", err)
		}
	}

	streamResp, err := doStream(ctx, m, req, cfg.maxRetries)
	if err != nil {
		if cancel != nil {
			cancel()
		}

		return nil, err
	}

	return &ObjectStreamResult[T]{
		stream:       streamResp.Stream,
		cancel:       cancel,
		metadata:     streamResp.Metadata.Clone(),
		useToolCalls: useToolCalls,
		onStepFinish: bindStepHooks(ctx, cfg.onStepFinish),
		onFinish:     bindFinishHooks(ctx, cfg.onFinish),
	}, nil
}

// JSON returns an iterator of raw JSON deltas.
func (s *ObjectStreamResult[T]) JSON() iter.Seq[string] {
	return func(yield func(string) bool) {
		s.consume(yield)
	}
}

// Result consumes the full stream and returns the final typed object.
func (s *ObjectStreamResult[T]) Result() (*ObjectResult[T], error) {
	s.mu.Lock()
	if s.consumed {
		s.mu.Unlock()

		return s.final, s.err
	}
	s.mu.Unlock()

	s.consume(func(string) bool {
		return true
	})

	s.mu.Lock()
	defer s.mu.Unlock()

	return s.final, s.err
}

func (s *ObjectStreamResult[T]) consume(yield func(string) bool) {
	var (
		jsonBuilder     strings.Builder
		usage           Usage
		finishReason    model.FinishReason
		rawFinishReason string
		toolCallID      string
		toolCallName    string
	)

	defer func() {
		s.done()

		raw := jsonBuilder.String()
		step := Step{
			FinishReason:    finishReason,
			RawFinishReason: rawFinishReason,
			Usage:           usage,
			Metadata:        s.metadata.Clone(),
		}
		if s.useToolCalls {
			if raw != "" || toolCallID != "" || toolCallName != "" {
				name := toolCallName
				if name == "" {
					name = "structured_output"
				}
				step.ToolCalls = []ToolCall{
					{
						ToolCallID: toolCallID,
						ToolName:   name,
						Args:       json.RawMessage(raw),
					},
				}
			}
		} else {
			step.Text = raw
		}

		if err := runStepHooks(s.onStepFinish, step); err != nil {
			s.mu.Lock()
			s.consumed = true
			s.err = err
			s.mu.Unlock()
			return
		}

		if err := runFinishHooks(s.onFinish, buildResult([]Step{step}, usage)); err != nil {
			s.mu.Lock()
			s.consumed = true
			s.err = err
			s.mu.Unlock()
			return
		}

		if raw == "" {
			s.mu.Lock()
			s.consumed = true
			s.err = fmt.Errorf("stream object: %w", ErrNoContent)
			s.mu.Unlock()
			return
		}

		var obj T
		if err := json.Unmarshal([]byte(raw), &obj); err != nil {
			s.mu.Lock()
			s.consumed = true
			s.err = fmt.Errorf("stream object: unmarshalling response: %w", err)
			s.mu.Unlock()
			return
		}

		s.mu.Lock()
		s.consumed = true
		s.final = &ObjectResult[T]{
			Object:   obj,
			RawJSON:  raw,
			Usage:    usage,
			Metadata: s.metadata.Clone(),
		}
		s.mu.Unlock()
	}()

	for chunk := range s.stream {
		switch c := chunk.(type) {
		case model.TextDelta:
			if !s.useToolCalls {
				jsonBuilder.WriteString(c.Text)
				if !yield(c.Text) {
					return
				}
			}
		case model.ToolCallDelta:
			if s.useToolCalls {
				if c.ToolCallID != "" {
					toolCallID = c.ToolCallID
				}
				if c.ToolName != "" {
					toolCallName = c.ToolName
				}
				jsonBuilder.WriteString(c.ArgsJSON)
				if !yield(c.ArgsJSON) {
					return
				}
			}
		case model.ProviderMetadataChunk:
			s.mu.Lock()
			s.metadata = c.Metadata.Clone()
			s.mu.Unlock()
		case model.UsageChunk:
			usage = c.Usage
		case model.FinishChunk:
			finishReason = c.FinishReason
			rawFinishReason = c.RawFinishReason
			if rawFinishReason == "" {
				rawFinishReason = string(c.FinishReason)
			}
		}
	}
}

func (s *ObjectStreamResult[T]) done() {
	if s.cancel != nil {
		s.cancel()
	}
}
