package ai

import "github.com/aarock1234/ai/pkg/model"

// modelContentRecorder rebuilds the provider content of a streamed call,
// for errors that report it.
type modelContentRecorder struct {
	content          []model.Content
	textIndexes      map[string]int
	reasoningIndexes map[string]int
}

func newModelContentRecorder() modelContentRecorder {
	return modelContentRecorder{
		textIndexes:      make(map[string]int),
		reasoningIndexes: make(map[string]int),
	}
}

func (r *modelContentRecorder) record(part model.StreamPart) {
	switch typed := part.(type) {
	case model.TextStart:
		r.textIndexes[typed.ID] = len(r.content)
		r.content = append(r.content, model.Text{
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.TextDelta:
		index, ok := r.textIndexes[typed.ID]
		if !ok {
			return
		}

		if text, isText := r.content[index].(model.Text); isText {
			text.Text += typed.Delta
			r.content[index] = text
		}

	case model.ReasoningStart:
		r.reasoningIndexes[typed.ID] = len(r.content)
		r.content = append(r.content, model.Reasoning{
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.ReasoningDelta:
		index, ok := r.reasoningIndexes[typed.ID]
		if !ok {
			return
		}

		if reasoning, isReasoning := r.content[index].(model.Reasoning); isReasoning {
			reasoning.Text += typed.Delta
			r.content[index] = reasoning
		}

	case model.ToolCall:
		r.content = append(r.content, typed)

	case model.ToolResult:
		r.content = append(r.content, typed)

	case model.ToolApprovalRequest:
		r.content = append(r.content, typed)

	case model.Source:
		r.content = append(r.content, typed)

	case model.Custom:
		r.content = append(r.content, typed)

	case model.File:
		r.content = append(r.content, typed)

	case model.ReasoningFile:
		r.content = append(r.content, typed)
	}
}

// callContentRecorder rebuilds the content of a streamed model call for its
// end event. A disabled recorder records nothing.
type callContentRecorder struct {
	enabled bool
	parts   []ContentPart

	textIndexes      map[string]int
	reasoningIndexes map[string]int
}

func newCallContentRecorder(enabled bool) callContentRecorder {
	return callContentRecorder{
		enabled:          enabled,
		textIndexes:      make(map[string]int),
		reasoningIndexes: make(map[string]int),
	}
}

func (r *callContentRecorder) add(part ContentPart) {
	if r.enabled {
		r.parts = append(r.parts, part)
	}
}

func (r *callContentRecorder) startText(partID string, metadata model.ProviderMetadata) {
	r.updateText(partID, func(text *TextContent) {
		text.ProviderMetadata = metadataOrDefault(metadata, text.ProviderMetadata)
	})
}

func (r *callContentRecorder) appendText(partID, delta string, metadata model.ProviderMetadata) {
	r.updateText(partID, func(text *TextContent) {
		text.Text += delta
		text.ProviderMetadata = metadataOrDefault(metadata, text.ProviderMetadata)
	})
}

func (r *callContentRecorder) endText(partID string, citations []model.Citation, metadata model.ProviderMetadata) {
	r.updateText(partID, func(text *TextContent) {
		text.ProviderMetadata = metadataOrDefault(metadata, text.ProviderMetadata)

		if citations != nil {
			text.Citations = citations
		}
	})

	delete(r.textIndexes, partID)
}

func (r *callContentRecorder) startReasoning(partID string, metadata model.ProviderMetadata) {
	r.updateReasoning(partID, func(reasoning *ReasoningContent) {
		reasoning.ProviderMetadata = metadataOrDefault(metadata, reasoning.ProviderMetadata)
	})
}

func (r *callContentRecorder) appendReasoning(partID, delta string, metadata model.ProviderMetadata) {
	r.updateReasoning(partID, func(reasoning *ReasoningContent) {
		reasoning.Text += delta
		reasoning.ProviderMetadata = metadataOrDefault(metadata, reasoning.ProviderMetadata)
	})
}

func (r *callContentRecorder) endReasoning(partID string, metadata model.ProviderMetadata) {
	r.updateReasoning(partID, func(reasoning *ReasoningContent) {
		reasoning.ProviderMetadata = metadataOrDefault(metadata, reasoning.ProviderMetadata)
	})

	delete(r.reasoningIndexes, partID)
}

// updateText updates the text block with the given ID, adding it when it
// is not open.
func (r *callContentRecorder) updateText(partID string, update func(text *TextContent)) {
	if !r.enabled {
		return
	}

	index, open := r.textIndexes[partID]
	if !open {
		index = len(r.parts)
		r.textIndexes[partID] = index
		r.parts = append(r.parts, TextContent{})
	}

	text, _ := contentAt[TextContent](r.parts, index)
	update(&text)
	r.parts[index] = text
}

// updateReasoning updates the reasoning block with the given ID, adding it
// when it is not open.
func (r *callContentRecorder) updateReasoning(partID string, update func(reasoning *ReasoningContent)) {
	if !r.enabled {
		return
	}

	index, open := r.reasoningIndexes[partID]
	if !open {
		index = len(r.parts)
		r.reasoningIndexes[partID] = index
		r.parts = append(r.parts, ReasoningContent{})
	}

	reasoning, _ := contentAt[ReasoningContent](r.parts, index)
	update(&reasoning)
	r.parts[index] = reasoning
}
