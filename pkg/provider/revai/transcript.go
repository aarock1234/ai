package revai

import (
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// textElement is the element type that carries timestamps.
const textElement = "text"

type transcriptResponse struct {
	Monologues []monologue `json:"monologues"`
}

type monologue struct {
	Elements []element `json:"elements"`
}

// element is a word or punctuation of a monologue, with times in seconds.
type element struct {
	Type  *string  `json:"type"`
	Value *string  `json:"value"`
	TS    *float64 `json:"ts"`
	EndTS *float64 `json:"end_ts"`
}

func (e element) value() string {
	if e.Value == nil {
		return ""
	}

	return *e.Value
}

func (e element) isText() bool {
	return e.Type != nil && *e.Type == textElement
}

// transcriptText joins the elements of each monologue, and the monologues
// with spaces.
func transcriptText(monologues []monologue) string {
	texts := make([]string, 0, len(monologues))
	for _, monologue := range monologues {
		var text strings.Builder
		for _, element := range monologue.Elements {
			text.WriteString(element.value())
		}

		texts = append(texts, text.String())
	}

	return strings.Join(texts, " ")
}

// segmentBuilder accumulates the elements of one segment.
type segmentBuilder struct {
	text    strings.Builder
	start   float64
	started bool
}

func (b *segmentBuilder) reset() {
	b.text.Reset()
	b.started = false
}

// transcriptSegments splits the monologues into segments that end at each
// timed text element, and returns the latest end time in seconds.
func transcriptSegments(monologues []monologue) ([]model.TranscriptionSegment, float64) {
	segments := make([]model.TranscriptionSegment, 0)
	duration := 0.0

	for _, monologue := range monologues {
		var builder segmentBuilder

		for _, element := range monologue.Elements {
			builder.text.WriteString(element.value())

			if !element.isText() {
				continue
			}

			if element.EndTS != nil && *element.EndTS > duration {
				duration = *element.EndTS
			}

			if !builder.started && element.TS != nil {
				builder.start = *element.TS
				builder.started = true
			}

			if element.EndTS == nil || !builder.started {
				continue
			}

			if text := strings.TrimSpace(builder.text.String()); text != "" {
				segments = append(segments, segment(text, builder.start, *element.EndTS))
			}

			builder.reset()
		}

		text := strings.TrimSpace(builder.text.String())
		if !builder.started || text == "" {
			continue
		}

		end := builder.start + 1
		if duration > builder.start {
			end = duration
		}

		segments = append(segments, segment(text, builder.start, end))
	}

	return segments, duration
}

func segment(text string, start float64, end float64) model.TranscriptionSegment {
	return model.TranscriptionSegment{
		Text:  text,
		Start: seconds(start),
		End:   seconds(end),
	}
}
