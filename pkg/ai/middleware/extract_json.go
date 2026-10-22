package middleware

import (
	"context"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/aarock1234/ai/pkg/model"
)

var (
	codeFencePrefix         = regexp.MustCompile("^```(?:json)?\\s*\\n?")
	completeCodeFencePrefix = regexp.MustCompile("^```(?:json)?\\s*\\n")
	codeFenceSuffix         = regexp.MustCompile("\\n?```\\s*$")
)

// ExtractJSONOptions configures ExtractJSON.
type ExtractJSONOptions struct {
	// Transform rewrites the complete text of each text block. The default
	// strips a surrounding Markdown code fence and whitespace. A custom
	// transform makes streaming calls buffer each block until it ends.
	Transform func(text string) string
}

// ExtractJSON returns middleware that extracts JSON from text output by
// removing Markdown code fences, for models that wrap structured output in
// them.
func ExtractJSON(options ExtractJSONOptions) Language {
	transform := options.Transform
	if transform == nil {
		transform = stripCodeFence
	}

	return Language{
		WrapGenerate: func(ctx context.Context, call LanguageCall) (*model.GenerateResult, error) {
			result, err := call.DoGenerate(ctx)
			if err != nil {
				return nil, err
			}

			transformed := *result
			transformed.Content = make([]model.Content, len(result.Content))

			for index, content := range result.Content {
				if text, ok := content.(model.Text); ok {
					text.Text = transform(text.Text)
					content = text
				}

				transformed.Content[index] = content
			}

			return &transformed, nil
		},

		WrapStream: func(ctx context.Context, call LanguageCall) (*model.StreamResult, error) {
			stream, err := call.DoStream(ctx)
			if err != nil {
				return nil, err
			}

			return transformStream(stream, &jsonExtraction{
				rewrite:            transform,
				hasCustomTransform: options.Transform != nil,
				blocks:             make(map[string]*jsonTextBlock),
			}), nil
		},
	}
}

func stripCodeFence(text string) string {
	text = codeFencePrefix.ReplaceAllString(text, "")
	text = codeFenceSuffix.ReplaceAllString(text, "")

	return strings.TrimSpace(text)
}

func stripCodeFenceSuffix(text string) string {
	text = codeFenceSuffix.ReplaceAllString(text, "")

	return strings.TrimRightFunc(text, unicode.IsSpace)
}

// jsonBlockPhase tracks how much of a text block has been decided.
type jsonBlockPhase int

const (
	// jsonPhasePrefix waits to learn whether the block opens a code fence.
	jsonPhasePrefix jsonBlockPhase = iota

	// jsonPhaseStreaming forwards text, holding back a possible fence
	// suffix.
	jsonPhaseStreaming

	// jsonPhaseBuffering holds the whole block for a custom transform.
	jsonPhaseBuffering
)

type jsonTextBlock struct {
	start          model.TextStart
	phase          jsonBlockPhase
	buffer         string
	prefixStripped bool
}

// jsonExtraction strips code fences from streamed text blocks.
type jsonExtraction struct {
	rewrite            func(text string) string
	hasCustomTransform bool
	blocks             map[string]*jsonTextBlock
}

func (e *jsonExtraction) transform(part model.StreamPart, out *partEmitter) {
	switch typed := part.(type) {
	case model.TextStart:
		phase := jsonPhasePrefix
		if e.hasCustomTransform {
			phase = jsonPhaseBuffering
		}

		e.blocks[typed.ID] = &jsonTextBlock{
			start: typed,
			phase: phase,
		}

	case model.TextDelta:
		block, ok := e.blocks[typed.ID]
		if !ok {
			out.emit(typed)

			return
		}

		e.appendDelta(block, typed.ID, typed.Delta, out)

	case model.TextEnd:
		block, ok := e.blocks[typed.ID]
		if !ok {
			out.emit(typed)

			return
		}

		e.finishBlock(block, typed, out)
		delete(e.blocks, typed.ID)

	default:
		out.emit(part)
	}
}

func (e *jsonExtraction) appendDelta(block *jsonTextBlock, id, delta string, out *partEmitter) {
	block.buffer += delta

	if block.phase == jsonPhaseBuffering {
		return
	}

	if block.phase == jsonPhasePrefix {
		e.decidePrefix(block, out)
	}

	if block.phase != jsonPhaseStreaming {
		return
	}

	suffixStart := potentialSuffixStart(block.buffer)
	ready := block.buffer[:suffixStart]
	block.buffer = block.buffer[suffixStart:]

	if ready == "" {
		return
	}

	out.emit(model.TextDelta{
		ID:    id,
		Delta: ready,
	})
}

// decidePrefix starts streaming once the buffer shows whether the block
// opens with a code fence, stripping a complete fence line.
func (e *jsonExtraction) decidePrefix(block *jsonTextBlock, out *partEmitter) {
	buffer := block.buffer

	switch {
	case buffer != "" && !strings.HasPrefix(buffer, "`"):
		e.startStreaming(block, out)

	case strings.HasPrefix(buffer, "```"):
		if !strings.Contains(buffer, "\n") {
			return
		}

		if fence := completeCodeFencePrefix.FindString(buffer); fence != "" {
			block.buffer = buffer[len(fence):]
			block.prefixStripped = true
		}

		e.startStreaming(block, out)

	case len(buffer) >= 3:
		e.startStreaming(block, out)
	}
}

func (e *jsonExtraction) startStreaming(block *jsonTextBlock, out *partEmitter) {
	block.phase = jsonPhaseStreaming
	out.emit(block.start)
}

func (e *jsonExtraction) finishBlock(block *jsonTextBlock, end model.TextEnd, out *partEmitter) {
	if block.phase != jsonPhaseStreaming {
		out.emit(block.start)
	}

	remaining := block.buffer

	switch block.phase {
	case jsonPhaseBuffering:
		remaining = e.rewrite(remaining)

	case jsonPhasePrefix:
		// No text has been streamed, so the full transform is safe.
		remaining = e.rewrite(remaining)

	default:
		// Earlier text has streamed, so trimming the start of the remainder
		// could remove meaningful whitespace.
		remaining = stripCodeFenceSuffix(remaining)
	}

	if remaining != "" {
		out.emit(model.TextDelta{
			ID:    end.ID,
			Delta: remaining,
		})
	}

	out.emit(end)
}

// potentialSuffixStart returns the index where trailing whitespace or a
// partial closing code fence begins, which must be held back until the block
// ends.
func potentialSuffixStart(text string) int {
	index := trimTrailingSpaceIndex(text, len(text))

	backticks := 0
	for index > 0 && backticks < 3 && text[index-1] == '`' {
		index--
		backticks++
	}

	if backticks > 0 {
		index = trimTrailingSpaceIndex(text, index)
	}

	return index
}

func trimTrailingSpaceIndex(text string, end int) int {
	for end > 0 {
		character, size := utf8.DecodeLastRuneInString(text[:end])
		if !unicode.IsSpace(character) {
			break
		}

		end -= size
	}

	return end
}
