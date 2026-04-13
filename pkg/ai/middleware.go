package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/aarock1234/ai/pkg/model"
)

// LanguageModelMiddleware wraps a model.Model with additional behavior.
type LanguageModelMiddleware interface {
	Wrap(model.Model) model.Model
}

// WrapLanguageModel applies middleware around a language model.
func WrapLanguageModel(m model.Model, middleware ...LanguageModelMiddleware) model.Model {
	wrapped := m
	for i := len(middleware) - 1; i >= 0; i-- {
		wrapped = middleware[i].Wrap(wrapped)
	}

	return wrapped
}

type middlewareModel struct {
	model.Model
	generateFn func(ctx context.Context, req *model.Request) (*model.Response, error)
	streamFn   func(ctx context.Context, req *model.Request) (*model.StreamResponse, error)
}

func (m *middlewareModel) DoGenerate(ctx context.Context, req *model.Request) (*model.Response, error) {
	if m.generateFn != nil {
		return m.generateFn(ctx, req)
	}

	return m.Model.DoGenerate(ctx, req)
}

func (m *middlewareModel) DoStream(ctx context.Context, req *model.Request) (*model.StreamResponse, error) {
	if m.streamFn != nil {
		return m.streamFn(ctx, req)
	}

	return m.Model.DoStream(ctx, req)
}

func (m *middlewareModel) SupportsResponseFormat(rf *model.ResponseFormat) bool {
	support, ok := m.Model.(model.ResponseFormatSupport)
	if !ok {
		return false
	}

	return support.SupportsResponseFormat(rf)
}

type middlewareFunc func(model.Model) model.Model

func (fn middlewareFunc) Wrap(m model.Model) model.Model {
	return fn(m)
}

// DefaultSettingsMiddleware applies default request settings before dispatch.
func DefaultSettingsMiddleware(defaults model.Request) LanguageModelMiddleware {
	return middlewareFunc(func(next model.Model) model.Model {
		return &middlewareModel{
			Model: next,
			generateFn: func(ctx context.Context, req *model.Request) (*model.Response, error) {
				return next.DoGenerate(ctx, mergeRequestDefaults(req, &defaults))
			},
			streamFn: func(ctx context.Context, req *model.Request) (*model.StreamResponse, error) {
				return next.DoStream(ctx, mergeRequestDefaults(req, &defaults))
			},
		}
	})
}

// ExtractJSONMiddleware strips Markdown JSON fences from model text output.
func ExtractJSONMiddleware() LanguageModelMiddleware {
	return middlewareFunc(func(next model.Model) model.Model {
		return &middlewareModel{
			Model: next,
			generateFn: func(ctx context.Context, req *model.Request) (*model.Response, error) {
				resp, err := next.DoGenerate(ctx, req)
				if err != nil {
					return nil, err
				}

				stripResponseJSON(resp)

				return resp, nil
			},
			streamFn: func(ctx context.Context, req *model.Request) (*model.StreamResponse, error) {
				resp, err := next.DoStream(ctx, req)
				if err != nil {
					return nil, err
				}

				return transformTextStream(resp, &jsonFenceTransformer{}), nil
			},
		}
	})
}

// ExtractReasoningMiddleware extracts <reasoning>...</reasoning> blocks.
func ExtractReasoningMiddleware() LanguageModelMiddleware {
	return middlewareFunc(func(next model.Model) model.Model {
		return &middlewareModel{
			Model: next,
			generateFn: func(ctx context.Context, req *model.Request) (*model.Response, error) {
				resp, err := next.DoGenerate(ctx, req)
				if err != nil {
					return nil, err
				}

				extractResponseReasoning(resp)

				return resp, nil
			},
			streamFn: func(ctx context.Context, req *model.Request) (*model.StreamResponse, error) {
				resp, err := next.DoStream(ctx, req)
				if err != nil {
					return nil, err
				}

				return transformTextStream(resp, &reasoningTransformer{}), nil
			},
		}
	})
}

// SimulateStreamingMiddleware turns DoGenerate output into a synthetic stream.
func SimulateStreamingMiddleware(delay time.Duration) LanguageModelMiddleware {
	return middlewareFunc(func(next model.Model) model.Model {
		return &middlewareModel{
			Model: next,
			streamFn: func(ctx context.Context, req *model.Request) (*model.StreamResponse, error) {
				resp, err := next.DoGenerate(ctx, req)
				if err != nil {
					return nil, err
				}

				return &model.StreamResponse{
					Metadata: resp.Metadata.Clone(),
					Stream: func(yield func(model.Chunk) bool) {
						if resp.Reasoning != "" {
							if !yield(model.ReasoningDelta{Text: resp.Reasoning}) {
								return
							}
						}

						for _, token := range splitStreamText(resp) {
							if delay > 0 {
								timer := time.NewTimer(delay)
								select {
								case <-ctx.Done():
									timer.Stop()
									return
								case <-timer.C:
								}
							}

							if !yield(model.TextDelta{Text: token}) {
								return
							}
						}

						for _, content := range resp.Content {
							toolCall, ok := content.(model.ToolCallContent)
							if !ok {
								continue
							}

							if !yield(model.ToolCallDelta{
								ToolCallID: toolCall.ToolCallID,
								ToolName:   toolCall.ToolName,
								ArgsJSON:   toolCall.Args,
							}) {
								return
							}
						}

						if !yield(model.FinishChunk{
							FinishReason:    resp.FinishReason,
							RawFinishReason: resp.RawFinishReason,
						}) {
							return
						}

						yield(model.UsageChunk{Usage: resp.Usage})
					},
				}, nil
			},
		}
	})
}

func mergeRequestDefaults(req *model.Request, defaults *model.Request) *model.Request {
	if defaults == nil {
		return req.Clone()
	}

	merged := defaults.Clone()
	if merged == nil {
		merged = &model.Request{}
	}
	if req == nil {
		return merged
	}

	if len(req.Messages) > 0 {
		merged.Messages = req.Clone().Messages
	}
	if req.Temperature != nil {
		merged.Temperature = req.Temperature
	}
	if req.MaxTokens != nil {
		merged.MaxTokens = req.MaxTokens
	}
	if req.TopP != nil {
		merged.TopP = req.TopP
	}
	if req.TopK != nil {
		merged.TopK = req.TopK
	}
	if req.PresencePenalty != nil {
		merged.PresencePenalty = req.PresencePenalty
	}
	if req.FrequencyPenalty != nil {
		merged.FrequencyPenalty = req.FrequencyPenalty
	}
	if req.Seed != nil {
		merged.Seed = req.Seed
	}
	if len(req.StopSequences) > 0 {
		merged.StopSequences = append([]string(nil), req.StopSequences...)
	}
	if len(req.Tools) > 0 {
		merged.Tools = req.Clone().Tools
	}
	if len(req.ActiveTools) > 0 {
		merged.ActiveTools = append([]string(nil), req.ActiveTools...)
	}
	if req.ToolChoice != nil {
		merged.ToolChoice = &model.ToolChoice{
			Type:     req.ToolChoice.Type,
			ToolName: req.ToolChoice.ToolName,
		}
	}
	if req.ResponseFormat != nil {
		merged.ResponseFormat = req.Clone().ResponseFormat
	}
	if len(req.Headers) > 0 {
		if merged.Headers == nil {
			merged.Headers = make(http.Header)
		}
		for key, values := range req.Headers {
			for _, value := range values {
				merged.Headers.Add(key, value)
			}
		}
	}
	if req.ProviderOptions != nil {
		merged.ProviderOptions = mergeProviderOptions(merged.ProviderOptions, req.ProviderOptions)
	}

	return merged
}

func stripResponseJSON(resp *model.Response) {
	for i, content := range resp.Content {
		text, ok := content.(model.TextContent)
		if !ok {
			continue
		}

		resp.Content[i] = model.TextContent{
			Text: stripJSONFence(text.Text),
		}
	}
}

func stripJSONFence(text string) string {
	text = strings.TrimSpace(text)
	text = strings.TrimPrefix(text, "```json")
	text = strings.TrimPrefix(text, "```JSON")
	text = strings.TrimPrefix(text, "```")
	text = strings.TrimSuffix(text, "```")

	return strings.TrimSpace(text)
}

func extractResponseReasoning(resp *model.Response) {
	for i, content := range resp.Content {
		text, ok := content.(model.TextContent)
		if !ok {
			continue
		}

		reasoning, cleaned := splitReasoning(text.Text)
		resp.Reasoning += reasoning
		resp.Content[i] = model.TextContent{Text: cleaned}
	}
}

func splitReasoning(text string) (string, string) {
	const (
		openTag  = "<reasoning>"
		closeTag = "</reasoning>"
	)

	start := strings.Index(text, openTag)
	end := strings.Index(text, closeTag)
	if start == -1 || end == -1 || end < start {
		return "", text
	}

	reasoning := strings.TrimSpace(text[start+len(openTag) : end])
	cleaned := strings.TrimSpace(text[:start] + text[end+len(closeTag):])

	return reasoning, cleaned
}

func splitStreamText(resp *model.Response) []string {
	var parts []string
	for _, content := range resp.Content {
		text, ok := content.(model.TextContent)
		if !ok || text.Text == "" {
			continue
		}

		for _, token := range strings.Fields(text.Text) {
			parts = append(parts, token+" ")
		}
		if len(parts) > 0 {
			parts[len(parts)-1] = strings.TrimSuffix(parts[len(parts)-1], " ")
		}
	}

	return parts
}

type streamTextTransformer interface {
	writeText(text string, yield func(model.Chunk) bool) bool
	finish(yield func(model.Chunk) bool) bool
	hasPending() bool
}

func transformTextStream(resp *model.StreamResponse, transformer streamTextTransformer) *model.StreamResponse {
	return &model.StreamResponse{
		Metadata: resp.Metadata.Clone(),
		Stream: func(yield func(model.Chunk) bool) {
			var buffered []model.Chunk
			buffering := false

			for chunk := range resp.Stream {
				if buffering {
					buffered = append(buffered, chunk)
					continue
				}

				text, ok := chunk.(model.TextDelta)
				if ok {
					if !transformer.writeText(text.Text, yield) {
						return
					}

					continue
				}

				if transformer.hasPending() {
					buffering = true
					buffered = append(buffered, chunk)
					continue
				}

				if !yield(chunk) {
					return
				}
			}

			if buffering {
				if !replayTransformedChunks(buffered, transformer, yield) {
					return
				}

				return
			}

			if !transformer.finish(yield) {
				return
			}
		},
	}
}

func replayTransformedChunks(chunks []model.Chunk, transformer streamTextTransformer, yield func(model.Chunk) bool) bool {
	var delayed []model.Chunk
	flushDelayed := func() bool {
		for _, chunk := range delayed {
			if !yield(chunk) {
				return false
			}
		}

		delayed = nil

		return true
	}

	for _, chunk := range chunks {
		text, ok := chunk.(model.TextDelta)
		if ok {
			if !transformer.writeText(text.Text, yield) {
				return false
			}

			if !transformer.hasPending() && len(delayed) > 0 {
				if !flushDelayed() {
					return false
				}
			}

			continue
		}

		if transformer.hasPending() {
			delayed = append(delayed, chunk)
			continue
		}

		if !yield(chunk) {
			return false
		}
	}

	if !transformer.finish(yield) {
		return false
	}

	if len(delayed) > 0 {
		return flushDelayed()
	}

	return true
}

type jsonFenceMode uint8

const (
	jsonFenceModeUnknown jsonFenceMode = iota
	jsonFenceModePlain
	jsonFenceModeFenced
)

type jsonFenceTransformer struct {
	mode        jsonFenceMode
	leading     string
	trailing    string
	trimLeading bool
}

func (t *jsonFenceTransformer) writeText(text string, yield func(model.Chunk) bool) bool {
	switch t.mode {
	case jsonFenceModeUnknown:
		t.leading += text
		return t.flushUnknown(yield)
	case jsonFenceModePlain:
		return yield(model.TextDelta{Text: text})
	case jsonFenceModeFenced:
		return t.flushFenced(text, yield)
	default:
		return true
	}
}

func (t *jsonFenceTransformer) finish(yield func(model.Chunk) bool) bool {
	switch t.mode {
	case jsonFenceModeUnknown:
		cleaned := stripJSONFence(t.leading)
		t.leading = ""
		if cleaned == "" {
			return true
		}

		return yield(model.TextDelta{Text: cleaned})
	case jsonFenceModeFenced:
		cleaned := strings.TrimSpace(t.trailing)
		t.trailing = ""
		if cleaned == "" || cleaned == "```" {
			return true
		}

		return yield(model.TextDelta{Text: cleaned})
	default:
		return true
	}
}

func (t *jsonFenceTransformer) hasPending() bool {
	switch t.mode {
	case jsonFenceModeUnknown:
		return t.leading != ""
	case jsonFenceModeFenced:
		return t.trailing != ""
	default:
		return false
	}
}

func (t *jsonFenceTransformer) flushUnknown(yield func(model.Chunk) bool) bool {
	trimmed := strings.TrimLeftFunc(t.leading, unicode.IsSpace)
	if isPrefixFragment(trimmed, "```") {
		return true
	}

	if !strings.HasPrefix(trimmed, "```") {
		t.mode = jsonFenceModePlain
		text := t.leading
		t.leading = ""

		return yield(model.TextDelta{Text: text})
	}

	markerLen, ok := jsonFenceMarkerLength(trimmed)
	if !ok {
		return true
	}

	t.mode = jsonFenceModeFenced
	t.leading = ""
	t.trimLeading = true

	return t.flushFenced(trimmed[markerLen:], yield)
}

func (t *jsonFenceTransformer) flushFenced(text string, yield func(model.Chunk) bool) bool {
	t.trailing += text

	safe, trailing := splitJSONFenceTrailing(t.trailing)
	t.trailing = trailing
	if t.trimLeading {
		safe = strings.TrimLeftFunc(safe, unicode.IsSpace)
		if safe != "" {
			t.trimLeading = false
		}
	}

	if safe == "" {
		return true
	}

	return yield(model.TextDelta{Text: safe})
}

func jsonFenceMarkerLength(text string) (int, bool) {
	if !strings.HasPrefix(text, "```") {
		return 0, false
	}

	rest := text[len("```"):]
	switch {
	case isPrefixFragment(rest, "json"):
		return 0, false
	case isPrefixFragment(rest, "JSON"):
		return 0, false
	case strings.HasPrefix(rest, "json"), strings.HasPrefix(rest, "JSON"):
		return len("```json"), true
	default:
		return len("```"), true
	}
}

func splitJSONFenceTrailing(text string) (string, string) {
	start := len(text)
	for start > 0 {
		r, size := utf8.DecodeLastRuneInString(text[:start])
		if r != '`' && !unicode.IsSpace(r) {
			break
		}

		start -= size
	}

	suffix := text[start:]
	switch strings.TrimSpace(suffix) {
	case "", "`", "``", "```":
		return text[:start], suffix
	default:
		return text, ""
	}
}

type reasoningTransformer struct {
	outsidePending     string
	reasoningLiteral   string
	reasoningPending   string
	inReasoning        bool
	trimOutsideLeading bool
}

func (t *reasoningTransformer) writeText(text string, yield func(model.Chunk) bool) bool {
	if t.inReasoning {
		t.reasoningPending += text

		return t.flushReasoning(yield)
	}

	t.outsidePending += text

	return t.flushOutside(yield)
}

func (t *reasoningTransformer) finish(yield func(model.Chunk) bool) bool {
	if t.inReasoning {
		literal := t.reasoningLiteral + t.reasoningPending
		t.reasoningLiteral = ""
		t.reasoningPending = ""
		t.inReasoning = false
		if !t.emitOutsideText(literal, yield) {
			return false
		}
	}

	if t.outsidePending == "" {
		return true
	}

	text := t.outsidePending
	t.outsidePending = ""

	return t.emitOutsideText(text, yield)
}

func (t *reasoningTransformer) hasPending() bool {
	return t.outsidePending != "" || t.inReasoning
}

func (t *reasoningTransformer) flushOutside(yield func(model.Chunk) bool) bool {
	const openTag = "<reasoning>"

	idx := strings.Index(t.outsidePending, openTag)
	if idx == -1 {
		hold := longestSuffixPrefix(t.outsidePending, openTag)
		safe := t.outsidePending[:len(t.outsidePending)-hold]
		t.outsidePending = t.outsidePending[len(t.outsidePending)-hold:]
		if safe == "" {
			return true
		}

		return t.emitOutsideText(safe, yield)
	}

	prefix := t.outsidePending[:idx]
	trimmedPrefix := strings.TrimRightFunc(prefix, unicode.IsSpace)
	t.reasoningLiteral = prefix[len(trimmedPrefix):] + openTag
	t.outsidePending = t.outsidePending[idx+len(openTag):]

	if trimmedPrefix != "" {
		if !t.emitOutsideText(trimmedPrefix, yield) {
			return false
		}
	}

	t.inReasoning = true
	t.reasoningPending = t.outsidePending
	t.outsidePending = ""

	return t.flushReasoning(yield)
}

func (t *reasoningTransformer) flushReasoning(yield func(model.Chunk) bool) bool {
	const closeTag = "</reasoning>"

	idx := strings.Index(t.reasoningPending, closeTag)
	if idx == -1 {
		return true
	}

	reasoning := strings.TrimSpace(t.reasoningPending[:idx])
	rest := t.reasoningPending[idx+len(closeTag):]

	t.reasoningLiteral = ""
	t.reasoningPending = ""
	t.inReasoning = false
	t.trimOutsideLeading = true

	if reasoning != "" {
		if !yield(model.ReasoningDelta{Text: reasoning}) {
			return false
		}
	}

	if rest == "" {
		return true
	}

	t.outsidePending = rest

	return t.flushOutside(yield)
}

func (t *reasoningTransformer) emitOutsideText(text string, yield func(model.Chunk) bool) bool {
	if t.trimOutsideLeading {
		text = strings.TrimLeftFunc(text, unicode.IsSpace)
		if text != "" {
			t.trimOutsideLeading = false
		}
	}

	if text == "" {
		return true
	}

	return yield(model.TextDelta{Text: text})
}

func longestSuffixPrefix(text, prefix string) int {
	limit := len(prefix) - 1
	if len(text) < limit {
		limit = len(text)
	}

	for i := limit; i > 0; i-- {
		if strings.HasSuffix(text, prefix[:i]) {
			return i
		}
	}

	return 0
}

func isPrefixFragment(fragment, full string) bool {
	return len(fragment) < len(full) && full[:len(fragment)] == fragment
}

func mergeProviderOptions(defaults, overrides model.ProviderOptions) model.ProviderOptions {
	if len(defaults) == 0 {
		return overrides.Clone()
	}

	merged := defaults.Clone()
	if len(overrides) == 0 {
		return merged
	}

	for provider, optionSet := range overrides {
		if merged[provider] == nil {
			merged[provider] = make(model.ProviderOptionSet)
		}

		for key, value := range optionSet {
			merged[provider][key] = append(json.RawMessage(nil), value...)
		}
	}

	return merged
}
