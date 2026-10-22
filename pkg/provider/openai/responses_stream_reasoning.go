package openai

import (
	"fmt"
	"maps"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// responsesSummaryStatus tracks one reasoning summary part. When responses
// are not stored, finished parts stay open until the reasoning item ends,
// so that the last part can carry the encrypted content.
type responsesSummaryStatus int

const (
	responsesSummaryActive responsesSummaryStatus = iota
	responsesSummaryCanConclude
	responsesSummaryConcluded
)

// responsesActiveReasoning is a reasoning item being streamed.
type responsesActiveReasoning struct {
	encryptedContent *string
	summaryParts     map[int]responsesSummaryStatus
}

// reasoningDone ends every open summary part of a reasoning item with its
// encrypted content.
func (s *responsesStreamState) reasoningDone(index int, item responsesReasoningOutput) {
	itemID := s.outputItemID(item.ID, &index)

	if active := s.activeReasoning[itemID]; active != nil {
		for _, summaryIndex := range slices.Sorted(maps.Keys(active.summaryParts)) {
			if active.summaryParts[summaryIndex] == responsesSummaryConcluded {
				continue
			}

			s.emit(model.ReasoningEnd{
				ID:               responsesSummaryID(itemID, summaryIndex),
				ProviderMetadata: s.context.reasoningMetadata(itemID, item.EncryptedContent),
			})
		}

		delete(s.activeReasoning, itemID)
	}

	delete(s.activeOutputItemIDs, index)
}

// summary handles reasoning summary part and text events.
func (s *responsesStreamState) summary(event responsesSummaryEvent) {
	itemID := s.outputItemID(event.ItemID, event.OutputIndex)
	partID := responsesSummaryID(itemID, event.SummaryIndex)
	active := s.activeReasoning[itemID]

	switch event.Type {
	case responsesEventSummaryPartAdded:
		// The first part starts with the reasoning item.
		if event.SummaryIndex > 0 && active != nil {
			s.startSummaryPart(itemID, event.SummaryIndex, active)
		}

	case responsesEventSummaryTextDelta:
		s.emit(model.ReasoningDelta{
			ID:               partID,
			Delta:            event.Delta,
			ProviderMetadata: s.context.reasoningMetadata(itemID, nil),
		})

	case responsesEventSummaryPartDone:
		if active == nil {
			return
		}

		// Stored reasoning needs no encrypted content, so the part can end
		// now. Otherwise the last part must carry it when the item ends.
		if !s.context.store {
			active.summaryParts[event.SummaryIndex] = responsesSummaryCanConclude

			return
		}

		s.emit(model.ReasoningEnd{
			ID:               partID,
			ProviderMetadata: s.context.reasoningMetadata(itemID, nil),
		})

		active.summaryParts[event.SummaryIndex] = responsesSummaryConcluded
	}
}

// startSummaryPart starts a new summary part, which concludes the parts
// that were waiting for a successor.
func (s *responsesStreamState) startSummaryPart(itemID string, summaryIndex int, active *responsesActiveReasoning) {
	active.summaryParts[summaryIndex] = responsesSummaryActive

	for _, index := range slices.Sorted(maps.Keys(active.summaryParts)) {
		if active.summaryParts[index] != responsesSummaryCanConclude {
			continue
		}

		s.emit(model.ReasoningEnd{
			ID:               responsesSummaryID(itemID, index),
			ProviderMetadata: s.context.reasoningMetadata(itemID, nil),
		})

		active.summaryParts[index] = responsesSummaryConcluded
	}

	s.emit(model.ReasoningStart{
		ID:               responsesSummaryID(itemID, summaryIndex),
		ProviderMetadata: s.context.reasoningMetadata(itemID, active.encryptedContent),
	})
}

// responsesSummaryID identifies one summary part of a reasoning item.
func responsesSummaryID(itemID string, summaryIndex int) string {
	return fmt.Sprintf("%s:%d", itemID, summaryIndex)
}
