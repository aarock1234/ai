package middleware

import (
	"context"
	"errors"
	"regexp"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// ExtractReasoningOptions configures ExtractReasoning.
type ExtractReasoningOptions struct {
	// TagName is an XML tag name without angle brackets, such as "think".
	// It is ignored when OpeningTag and ClosingTag are set.
	TagName string

	// OpeningTag and ClosingTag are literal delimiters, for formats that do
	// not use XML tags.
	OpeningTag string
	ClosingTag string

	// Separator joins reasoning sections and the text around a removed
	// section. Nil selects a newline.
	Separator *string

	// StartWithReasoning treats the text as starting inside a reasoning
	// section, for models that omit the opening tag.
	StartWithReasoning bool
}

// reasoningDelimiters are the resolved options of ExtractReasoning.
type reasoningDelimiters struct {
	opening            string
	closing            string
	separator          string
	startWithReasoning bool
}

// ExtractReasoning returns middleware that moves delimited reasoning
// sections out of the generated text into reasoning content. It fails when
// the delimiters are empty.
func ExtractReasoning(options ExtractReasoningOptions) (Language, error) {
	delimiters := reasoningDelimiters{
		opening:            options.OpeningTag,
		closing:            options.ClosingTag,
		separator:          "\n",
		startWithReasoning: options.StartWithReasoning,
	}

	if delimiters.opening == "" && delimiters.closing == "" && options.TagName != "" {
		delimiters.opening = "<" + options.TagName + ">"
		delimiters.closing = "</" + options.TagName + ">"
	}

	if delimiters.opening == "" || delimiters.closing == "" {
		return Language{}, errors.New("extract reasoning: reasoning delimiters must not be empty")
	}

	if options.Separator != nil {
		delimiters.separator = *options.Separator
	}

	sectionPattern := regexp.MustCompile(
		"(?s)" + regexp.QuoteMeta(delimiters.opening) + "(.*?)" + regexp.QuoteMeta(delimiters.closing),
	)

	return Language{
		WrapGenerate: func(ctx context.Context, call LanguageCall) (*model.GenerateResult, error) {
			result, err := call.DoGenerate(ctx)
			if err != nil {
				return nil, err
			}

			transformed := *result
			transformed.Content = extractReasoningContent(result.Content, delimiters, sectionPattern)

			return &transformed, nil
		},

		WrapStream: func(ctx context.Context, call LanguageCall) (*model.StreamResult, error) {
			stream, err := call.DoStream(ctx)
			if err != nil {
				return nil, err
			}

			return transformStream(stream, &reasoningExtraction{
				delimiters:        delimiters,
				blocks:            make(map[string]*reasoningBlock),
				delayedTextStarts: make(map[string]model.TextStart),
			}), nil
		},
	}, nil
}

// extractReasoningContent splits each text item that contains reasoning
// sections into a reasoning item followed by the remaining text.
func extractReasoningContent(
	content []model.Content,
	delimiters reasoningDelimiters,
	sectionPattern *regexp.Regexp,
) []model.Content {
	transformed := make([]model.Content, 0, len(content))

	for _, item := range content {
		text, ok := item.(model.Text)
		if !ok {
			transformed = append(transformed, item)

			continue
		}

		source := text.Text
		if delimiters.startWithReasoning {
			source = delimiters.opening + source
		}

		matches := sectionPattern.FindAllStringSubmatchIndex(source, -1)
		if len(matches) == 0 {
			transformed = append(transformed, item)

			continue
		}

		sections := make([]string, len(matches))
		for index, match := range matches {
			sections[index] = source[match[2]:match[3]]
		}

		text.Text = removeSections(source, matches, delimiters.separator)

		reasoning := model.Reasoning{
			Text: strings.Join(sections, delimiters.separator),
		}

		transformed = append(transformed, reasoning, text)
	}

	return transformed
}

// removeSections removes the matched sections from the last to the first,
// joining the surrounding text with the separator when both sides are
// non-empty.
func removeSections(text string, matches [][]int, separator string) string {
	for index := len(matches) - 1; index >= 0; index-- {
		before := text[:matches[index][0]]
		after := text[matches[index][1]:]

		if before != "" && after != "" {
			text = before + separator + after
		} else {
			text = before + after
		}
	}

	return text
}

// reasoningBlock is the extraction state of one streamed text block.
type reasoningBlock struct {
	textID      string
	reasoningID string
	buffer      string

	isReasoning      bool
	isFirstReasoning bool
	isFirstText      bool
	afterSwitch      bool
}

// reasoningExtraction moves delimited reasoning out of streamed text blocks.
// Text starts are delayed until the block's first text so that a block
// opening with reasoning emits its reasoning start first.
type reasoningExtraction struct {
	delimiters        reasoningDelimiters
	blocks            map[string]*reasoningBlock
	delayedTextStarts map[string]model.TextStart
	nextReasoningID   int
}

func (e *reasoningExtraction) transform(part model.StreamPart, out *partEmitter) {
	switch typed := part.(type) {
	case model.TextStart:
		e.delayedTextStarts[typed.ID] = typed

	case model.TextDelta:
		block, ok := e.blocks[typed.ID]
		if !ok {
			block = &reasoningBlock{
				textID:           typed.ID,
				isReasoning:      e.delimiters.startWithReasoning,
				isFirstReasoning: true,
				isFirstText:      true,
			}
			e.blocks[typed.ID] = block
		}

		block.buffer += typed.Delta
		e.drainBuffer(block, out)

	case model.TextEnd:
		if block, ok := e.blocks[typed.ID]; ok {
			e.publish(block, block.buffer, out)
			block.buffer = ""
		}

		e.emitDelayedTextStart(typed.ID, out)
		out.emit(typed)

	default:
		out.emit(part)
	}
}

// drainBuffer publishes the buffered text up to the next delimiter, switching
// between text and reasoning at every complete delimiter, and keeps a partial
// delimiter buffered.
func (e *reasoningExtraction) drainBuffer(block *reasoningBlock, out *partEmitter) {
	for {
		delimiter := e.delimiters.opening
		if block.isReasoning {
			delimiter = e.delimiters.closing
		}

		start, found := potentialStartIndex(block.buffer, delimiter)
		if !found {
			e.publish(block, block.buffer, out)
			block.buffer = ""

			return
		}

		e.publish(block, block.buffer[:start], out)

		if start+len(delimiter) > len(block.buffer) {
			block.buffer = block.buffer[start:]

			return
		}

		block.buffer = block.buffer[start+len(delimiter):]

		if block.isReasoning {
			e.endReasoning(block, out)
		}

		block.isReasoning = !block.isReasoning
		block.afterSwitch = true
	}
}

// endReasoning closes the current reasoning section, opening it first when
// it was empty.
func (e *reasoningExtraction) endReasoning(block *reasoningBlock, out *partEmitter) {
	if block.isFirstReasoning {
		out.emit(model.ReasoningStart{
			ID: e.reasoningID(block),
		})
	}

	out.emit(model.ReasoningEnd{
		ID: e.reasoningID(block),
	})

	block.reasoningID = ""
}

func (e *reasoningExtraction) publish(block *reasoningBlock, text string, out *partEmitter) {
	if text == "" {
		return
	}

	prefix := ""
	isFirstOfKind := block.isFirstText
	if block.isReasoning {
		isFirstOfKind = block.isFirstReasoning
	}

	if block.afterSwitch && !isFirstOfKind {
		prefix = e.delimiters.separator
	}

	if block.isReasoning {
		if block.afterSwitch || block.isFirstReasoning {
			out.emit(model.ReasoningStart{
				ID: e.reasoningID(block),
			})
		}

		out.emit(model.ReasoningDelta{
			ID:    e.reasoningID(block),
			Delta: prefix + text,
		})

		block.isFirstReasoning = false
	} else {
		e.emitDelayedTextStart(block.textID, out)

		out.emit(model.TextDelta{
			ID:    block.textID,
			Delta: prefix + text,
		})

		block.isFirstText = false
	}

	block.afterSwitch = false
}

func (e *reasoningExtraction) emitDelayedTextStart(id string, out *partEmitter) {
	start, ok := e.delayedTextStarts[id]
	if !ok {
		return
	}

	out.emit(start)
	delete(e.delayedTextStarts, id)
}

// reasoningID returns the ID of the block's current reasoning section,
// assigning a new one when the section starts.
func (e *reasoningExtraction) reasoningID(block *reasoningBlock) string {
	if block.reasoningID == "" {
		block.reasoningID = "reasoning-" + strconv.Itoa(e.nextReasoningID)
		e.nextReasoningID++
	}

	return block.reasoningID
}

// potentialStartIndex returns the index of the first complete occurrence of
// searched in text or, failing that, the start of the shortest suffix of text
// that begins searched, as upstream's getPotentialStartIndex does.
func potentialStartIndex(text, searched string) (int, bool) {
	if searched == "" {
		return 0, false
	}

	if index := strings.Index(text, searched); index >= 0 {
		return index, true
	}

	for index := len(text) - 1; index >= 0; index-- {
		if strings.HasPrefix(searched, text[index:]) {
			return index, true
		}
	}

	return 0, false
}
