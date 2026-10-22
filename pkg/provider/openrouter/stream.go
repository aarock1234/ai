package openrouter

import (
	"encoding/json"
	"iter"

	"github.com/aarock1234/ai/pkg/model"
)

// detailsReasoningID is the ID of the reasoning block emitted for reasoning
// details that arrived without reasoning text.
const detailsReasoningID = "reasoning-details"

// extensionStream adds OpenRouter's extensions to an OpenAI-compatible
// stream. It reads each raw chunk before the parts converted from it.
type extensionStream struct {
	includeRawChunks bool
	generateID       func() string

	details         reasoningAccumulator
	reasoningActive bool
}

func (s *extensionStream) parts(inner iter.Seq[model.StreamPart]) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		for part := range inner {
			for _, converted := range s.convert(part) {
				if !yield(converted) {
					return
				}
			}
		}
	}
}

// convert returns the parts to emit for one inner part.
func (s *extensionStream) convert(part model.StreamPart) []model.StreamPart {
	switch part := part.(type) {
	case model.Raw:
		return s.rawChunk(part)

	case model.ReasoningStart:
		s.reasoningActive = true

		return []model.StreamPart{
			part,
		}

	case model.ReasoningEnd:
		s.reasoningActive = false
		part.ProviderMetadata = reasoningMetadata(s.details.take())

		return []model.StreamPart{
			part,
		}

	case model.TextStart, model.ToolInputStart, model.Finish:
		return append(s.detailsBlock(), part)

	default:
		return []model.StreamPart{
			part,
		}
	}
}

// rawChunk records the chunk's reasoning details and returns its images and
// citations, preceded by the raw chunk when the caller asked for it.
func (s *extensionStream) rawChunk(raw model.Raw) []model.StreamPart {
	var parts []model.StreamPart
	if s.includeRawChunks {
		parts = append(parts, raw)
	}

	// Chunks that do not have the expected shape carry no extensions; the
	// chat model reports them as stream errors.
	var extensions responseExtensions
	if err := json.Unmarshal(raw.Value, &extensions); err != nil {
		return parts
	}

	delta := extensions.first()
	s.details.add(delta.ReasoningDetails)

	files, err := imageFiles(delta.Images)
	if err != nil {
		return append(parts, model.StreamError{
			Err: err,
		})
	}

	for _, file := range files {
		parts = append(parts, file)
	}

	for _, source := range citationSources(delta.Annotations, s.generateID) {
		parts = append(parts, source)
	}

	return parts
}

// detailsBlock emits pending reasoning details as an empty reasoning block
// when they arrived outside a reasoning block, so that they can be replayed.
func (s *extensionStream) detailsBlock() []model.StreamPart {
	if s.reasoningActive || s.details.empty() {
		return nil
	}

	return []model.StreamPart{
		model.ReasoningStart{
			ID: detailsReasoningID,
		},
		model.ReasoningEnd{
			ID:               detailsReasoningID,
			ProviderMetadata: reasoningMetadata(s.details.take()),
		},
	}
}
