package azure

import (
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

var _ model.EmbeddingModel = (*EmbeddingModel)(nil)

// EmbeddingModel is an Azure OpenAI embedding deployment. It uses the OpenAI
// embeddings API and reads openai.EmbeddingOptions from the "openai"
// provider options namespace.
type EmbeddingModel struct {
	*openai.EmbeddingModel

	provider string
}

// Provider returns "azure.embeddings".
func (m *EmbeddingModel) Provider() string {
	return m.provider
}
