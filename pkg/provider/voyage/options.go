package voyage

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// InputType tells Voyage how the embedded text will be used.
type InputType string

// Input types.
const (
	InputQuery    InputType = "query"
	InputDocument InputType = "document"
)

// EmbeddingDataType is the data type of returned embeddings.
type EmbeddingDataType string

// Embedding data types.
const (
	EmbeddingFloat   EmbeddingDataType = "float"
	EmbeddingInt8    EmbeddingDataType = "int8"
	EmbeddingUint8   EmbeddingDataType = "uint8"
	EmbeddingBinary  EmbeddingDataType = "binary"
	EmbeddingUbinary EmbeddingDataType = "ubinary"
)

// EmbeddingOptions are the provider options of embedding models.
type EmbeddingOptions struct {
	// InputType prepends a retrieval prompt for queries or documents.
	InputType InputType `json:"inputType,omitempty"`

	// Truncation truncates values that exceed the context length.
	Truncation *bool `json:"truncation,omitempty"`

	// OutputDimension is the vector size. It takes precedence over
	// model.EmbedOptions.Dimensions.
	OutputDimension *int `json:"outputDimension,omitempty"`

	// OutputDtype is the data type of the returned vectors.
	OutputDtype EmbeddingDataType `json:"outputDtype,omitempty"`
}

func readEmbeddingOptions(options model.ProviderOptions) (EmbeddingOptions, error) {
	embeddingOptions, _, err := model.ProviderValue[EmbeddingOptions](options, Namespace)
	if err != nil {
		return EmbeddingOptions{}, err
	}

	switch embeddingOptions.InputType {
	case "", InputQuery, InputDocument:
	default:
		return EmbeddingOptions{}, fmt.Errorf("%w: voyage inputType %q", model.ErrInvalidArgument, embeddingOptions.InputType)
	}

	switch embeddingOptions.OutputDtype {
	case "", EmbeddingFloat, EmbeddingInt8, EmbeddingUint8, EmbeddingBinary, EmbeddingUbinary:
		return embeddingOptions, nil
	default:
		return EmbeddingOptions{}, fmt.Errorf("%w: voyage outputDtype %q", model.ErrInvalidArgument, embeddingOptions.OutputDtype)
	}
}

// RerankingOptions are the provider options of reranking models.
type RerankingOptions struct {
	// ReturnDocuments includes the documents in the response.
	ReturnDocuments *bool `json:"returnDocuments,omitempty"`

	// Truncation truncates the query and documents to fit the context.
	Truncation *bool `json:"truncation,omitempty"`
}
