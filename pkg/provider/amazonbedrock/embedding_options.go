package amazonbedrock

import (
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// EmbeddingPurpose is the purpose of a Nova embedding.
type EmbeddingPurpose string

// Nova embedding purposes.
const (
	EmbeddingGenericIndex      EmbeddingPurpose = "GENERIC_INDEX"
	EmbeddingTextRetrieval     EmbeddingPurpose = "TEXT_RETRIEVAL"
	EmbeddingImageRetrieval    EmbeddingPurpose = "IMAGE_RETRIEVAL"
	EmbeddingVideoRetrieval    EmbeddingPurpose = "VIDEO_RETRIEVAL"
	EmbeddingDocumentRetrieval EmbeddingPurpose = "DOCUMENT_RETRIEVAL"
	EmbeddingAudioRetrieval    EmbeddingPurpose = "AUDIO_RETRIEVAL"
	EmbeddingGenericRetrieval  EmbeddingPurpose = "GENERIC_RETRIEVAL"
	EmbeddingClassification    EmbeddingPurpose = "CLASSIFICATION"
	EmbeddingClustering        EmbeddingPurpose = "CLUSTERING"
)

// EmbeddingInputType is the input type of a Cohere embedding.
type EmbeddingInputType string

// Cohere input types.
const (
	EmbeddingSearchDocument EmbeddingInputType = "search_document"
	EmbeddingSearchQuery    EmbeddingInputType = "search_query"
	EmbeddingInputClassify  EmbeddingInputType = "classification"
	EmbeddingInputCluster   EmbeddingInputType = "clustering"
)

// Truncation selects how input longer than the context is truncated.
type Truncation string

// Truncation modes.
const (
	TruncateNone  Truncation = "NONE"
	TruncateStart Truncation = "START"
	TruncateEnd   Truncation = "END"
)

// EmbeddingOptions are the provider options of embedding models, read from
// the "amazonBedrock" namespace, or from "bedrock" when the call has no
// "amazonBedrock" options.
type EmbeddingOptions struct {
	// Dimensions is the Titan Text Embeddings V2 size: 1024 (the default),
	// 512 or 256.
	Dimensions *int `json:"dimensions,omitempty"`

	// Normalize normalizes Titan Text Embeddings V2 vectors. It defaults
	// to true.
	Normalize *bool `json:"normalize,omitempty"`

	// EmbeddingDimension is the Nova embedding size: 256, 384, 1024 (the
	// default) or 3072.
	EmbeddingDimension *int `json:"embeddingDimension,omitempty"`

	// EmbeddingPurpose is the Nova embedding purpose. The default is
	// EmbeddingGenericIndex.
	EmbeddingPurpose *EmbeddingPurpose `json:"embeddingPurpose,omitempty"`

	// InputType is the Cohere input type. The default is
	// EmbeddingSearchQuery.
	InputType *EmbeddingInputType `json:"inputType,omitempty"`

	// Truncate is the Cohere and Nova truncation. Nova defaults to
	// TruncateEnd.
	Truncate *Truncation `json:"truncate,omitempty"`

	// OutputDimension is the Cohere Embed v4 size: 256, 512, 1024 or 1536
	// (the default).
	OutputDimension *int `json:"outputDimension,omitempty"`
}

func readEmbeddingOptions(options model.ProviderOptions) (EmbeddingOptions, error) {
	value, _, err := namespacedValue[EmbeddingOptions](options)
	if err != nil {
		return EmbeddingOptions{}, err
	}

	if err := value.validate(); err != nil {
		return EmbeddingOptions{}, err
	}

	return value, nil
}

func (o EmbeddingOptions) validate() error {
	sizes := []struct {
		name    string
		value   *int
		allowed []int
	}{
		{
			name:  "dimensions",
			value: o.Dimensions,
			allowed: []int{
				1024,
				512,
				256,
			},
		},
		{
			name:  "embeddingDimension",
			value: o.EmbeddingDimension,
			allowed: []int{
				256,
				384,
				1024,
				3072,
			},
		},
		{
			name:  "outputDimension",
			value: o.OutputDimension,
			allowed: []int{
				256,
				512,
				1024,
				1536,
			},
		},
	}

	for _, size := range sizes {
		if size.value != nil && !slices.Contains(size.allowed, *size.value) {
			return fmt.Errorf("%w: amazon bedrock %s %d", model.ErrInvalidArgument, size.name, *size.value)
		}
	}

	if o.EmbeddingPurpose != nil {
		purposes := []EmbeddingPurpose{
			EmbeddingGenericIndex,
			EmbeddingTextRetrieval,
			EmbeddingImageRetrieval,
			EmbeddingVideoRetrieval,
			EmbeddingDocumentRetrieval,
			EmbeddingAudioRetrieval,
			EmbeddingGenericRetrieval,
			EmbeddingClassification,
			EmbeddingClustering,
		}

		if err := checkEnum("embeddingPurpose", *o.EmbeddingPurpose, purposes); err != nil {
			return err
		}
	}

	if o.InputType != nil {
		inputTypes := []EmbeddingInputType{
			EmbeddingSearchDocument,
			EmbeddingSearchQuery,
			EmbeddingInputClassify,
			EmbeddingInputCluster,
		}

		if err := checkEnum("inputType", *o.InputType, inputTypes); err != nil {
			return err
		}
	}

	if o.Truncate != nil {
		truncations := []Truncation{
			TruncateNone,
			TruncateStart,
			TruncateEnd,
		}

		return checkEnum("truncate", *o.Truncate, truncations)
	}

	return nil
}
