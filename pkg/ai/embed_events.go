package ai

import (
	"github.com/aarock1234/ai/pkg/model"
)

// EmbeddingModelCallStartEvent describes one embedding model call attempt
// before it is sent. Embed makes one call; EmbedMany makes one per batch.
type EmbeddingModelCallStartEvent struct {
	// CallID identifies the Embed or EmbedMany call.
	CallID string

	// EmbedCallID identifies this model call, to correlate its start and end
	// among parallel batches.
	EmbedCallID string

	// OperationID is "ai.embed.doEmbed" or "ai.embedMany.doEmbed".
	OperationID string

	Provider string
	ModelID  string

	// Values are the values embedded by this model call.
	Values     []string
	Dimensions *int
}

// EmbeddingModelCallEndEvent describes one successful embedding model call.
type EmbeddingModelCallEndEvent struct {
	CallID      string
	EmbedCallID string
	OperationID string
	Provider    string
	ModelID     string
	Values      []string
	Embeddings  [][]float64

	// Usage is nil when the provider did not report it.
	Usage *model.EmbeddingUsage
}
