package openrouter

import (
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// ReasoningDetailType identifies a reasoning detail block.
type ReasoningDetailType string

// Reasoning detail types.
const (
	ReasoningDetailText      ReasoningDetailType = "reasoning.text"
	ReasoningDetailSummary   ReasoningDetailType = "reasoning.summary"
	ReasoningDetailEncrypted ReasoningDetailType = "reasoning.encrypted"
)

// ReasoningDetail is one structured reasoning block. Upstream providers need
// the blocks of earlier turns, including signatures and encrypted reasoning,
// to continue reasoning across tool calls. The JSON names follow OpenRouter's
// format because the blocks are sent back unchanged.
type ReasoningDetail struct {
	Type ReasoningDetailType `json:"type"`

	// Text is the reasoning of a text block.
	Text string `json:"text,omitempty"`

	// Summary is the reasoning summary of a summary block.
	Summary string `json:"summary,omitempty"`

	// Data is the opaque reasoning of an encrypted block.
	Data string `json:"data,omitempty"`

	Signature string `json:"signature,omitempty"`
	ID        string `json:"id,omitempty"`
	Format    string `json:"format,omitempty"`

	// Index orders the blocks. Streamed fragments with the same index belong
	// to one block.
	Index *int `json:"index,omitempty"`
}

// ReasoningMetadata is the provider metadata of reasoning content. When the
// reasoning is replayed in a later call, its details are sent back on the
// assistant message.
type ReasoningMetadata struct {
	ReasoningDetails []ReasoningDetail `json:"reasoningDetails,omitempty"`
}

// reasoningMetadata returns provider metadata holding details, or nil.
func reasoningMetadata(details []ReasoningDetail) model.ProviderMetadata {
	if len(details) == 0 {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: ReasoningMetadata{
			ReasoningDetails: details,
		},
	}
}

// reasoningAccumulator merges streamed reasoning detail fragments into whole
// blocks.
type reasoningAccumulator struct {
	details []ReasoningDetail
}

// add merges fragments into the block with the same type and index, and
// appends fragments without an index.
func (a *reasoningAccumulator) add(fragments []ReasoningDetail) {
	for _, fragment := range fragments {
		position := a.position(fragment)
		if position < 0 {
			a.details = append(a.details, fragment)

			continue
		}

		a.details[position] = mergeFragment(a.details[position], fragment)
	}
}

func (a *reasoningAccumulator) position(fragment ReasoningDetail) int {
	if fragment.Index == nil {
		return -1
	}

	return slices.IndexFunc(a.details, func(detail ReasoningDetail) bool {
		return detail.Index != nil && *detail.Index == *fragment.Index && detail.Type == fragment.Type
	})
}

// take returns the accumulated blocks and starts over.
func (a *reasoningAccumulator) take() []ReasoningDetail {
	details := a.details
	a.details = nil

	return details
}

func (a *reasoningAccumulator) empty() bool {
	return len(a.details) == 0
}

// mergeFragment appends the fragment's content and keeps its latest
// identifying fields.
func mergeFragment(detail ReasoningDetail, fragment ReasoningDetail) ReasoningDetail {
	detail.Text += fragment.Text
	detail.Summary += fragment.Summary
	detail.Data += fragment.Data

	if fragment.Signature != "" {
		detail.Signature = fragment.Signature
	}

	if fragment.ID != "" {
		detail.ID = fragment.ID
	}

	if fragment.Format != "" {
		detail.Format = fragment.Format
	}

	return detail
}
