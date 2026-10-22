package harness

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// partialOutputTracker projects the text of the first text block onto the
// partial output of a configured ai.Output, as upstream's output transform
// does: text deltas are held back and republished in chunks whenever the
// parsed partial output changes, so each published delta carries the
// partial value it produced.
type partialOutputTracker struct {
	firstTextID string
	started     bool

	text          string
	pendingText   string
	textMetadata  model.ProviderMetadata
	lastPublished string
}

// project returns the parts to log for part.
func (t *partialOutputTracker) project(output ai.Output, part ai.StreamPart) []loggedPart {
	var projected []loggedPart

	if _, isStepEnd := part.(ai.FinishStepPart); isStepEnd && t.pendingText != "" {
		projected = append(projected, t.publish(nil, false))
	}

	textID, isText := textPartID(part)
	if !isText {
		return append(projected, passThrough(part))
	}

	if !t.started {
		t.started = true
		t.firstTextID = textID
	} else if textID != t.firstTextID {
		return append(projected, passThrough(part))
	}

	switch typed := part.(type) {
	case ai.TextStartPart:
		return append(projected, passThrough(part))
	case ai.TextEndPart:
		if t.pendingText != "" {
			projected = append(projected, t.publish(nil, false))
		}

		return append(projected, passThrough(part))
	case ai.TextDeltaPart:
		return append(projected, t.projectDelta(output, typed)...)
	default:
		return append(projected, passThrough(part))
	}
}

// projectDelta accumulates a delta and publishes the pending text when the
// partial output changed.
func (t *partialOutputTracker) projectDelta(output ai.Output, part ai.TextDeltaPart) []loggedPart {
	t.text += part.Text
	t.pendingText += part.Text

	if part.ProviderMetadata != nil {
		t.textMetadata = part.ProviderMetadata
	}

	if part.Text == "" && part.ProviderMetadata != nil {
		return []loggedPart{
			passThrough(part),
		}
	}

	partial, ok := output.ParsePartialOutput(t.text)
	if !ok {
		return nil
	}

	current := partialOutputKey(partial)
	if current == t.lastPublished {
		return nil
	}

	t.lastPublished = current

	return []loggedPart{
		t.publish(partial, true),
	}
}

// publish emits the pending text as one delta.
func (t *partialOutputTracker) publish(partial any, hasPartial bool) loggedPart {
	published := loggedPart{
		part: ai.TextDeltaPart{
			ID:               t.firstTextID,
			Text:             t.pendingText,
			ProviderMetadata: t.textMetadata,
		},
		partialOutput:    partial,
		hasPartialOutput: hasPartial,
	}

	t.pendingText = ""

	return published
}

// textPartID returns the block ID of a text part.
func textPartID(part ai.StreamPart) (string, bool) {
	switch typed := part.(type) {
	case ai.TextStartPart:
		return typed.ID, true
	case ai.TextDeltaPart:
		return typed.ID, true
	case ai.TextEndPart:
		return typed.ID, true
	default:
		return "", false
	}
}

// partialOutputKey identifies a partial output value for change detection.
func partialOutputKey(partial any) string {
	if text, ok := partial.(string); ok {
		return text
	}

	encoded, err := json.Marshal(partial)
	if err != nil {
		return ""
	}

	return string(encoded)
}

// passThrough logs a part without partial output.
func passThrough(part ai.StreamPart) loggedPart {
	return loggedPart{
		part: part,
	}
}
