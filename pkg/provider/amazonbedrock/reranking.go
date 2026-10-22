package amazonbedrock

import (
	"context"
	"encoding/json"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
)

var _ model.RerankingModel = (*RerankingModel)(nil)

// RerankingOptions are the provider options of reranking models, read from
// the "amazonBedrock" namespace, or from "bedrock" when the call has no
// "amazonBedrock" options.
type RerankingOptions struct {
	// NextToken requests the next batch of results of an earlier call
	// whose results did not fit in one response.
	NextToken *string `json:"nextToken,omitempty"`

	// AdditionalModelRequestFields are passed to the reranking model.
	AdditionalModelRequestFields jsonobject.Object `json:"additionalModelRequestFields,omitempty"`
}

// RerankingModel is an Amazon Bedrock reranking model, such as
// amazon.rerank-v1:0 or cohere.rerank-v3-5:0.
type RerankingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "amazon-bedrock" by default.
func (m *RerankingModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model ID.
func (m *RerankingModel) ModelID() string {
	return m.modelID
}

type rerankRequest struct {
	NextToken              *string                `json:"nextToken,omitempty"`
	Queries                []rerankQuery          `json:"queries"`
	RerankingConfiguration rerankingConfiguration `json:"rerankingConfiguration"`
	Sources                []rerankSource         `json:"sources"`
}

type rerankQuery struct {
	TextQuery rerankText `json:"textQuery"`
	Type      string     `json:"type"`
}

type rerankText struct {
	Text string `json:"text"`
}

type rerankingConfiguration struct {
	BedrockRerankingConfiguration bedrockRerankingConfiguration `json:"bedrockRerankingConfiguration"`
	Type                          string                        `json:"type"`
}

type bedrockRerankingConfiguration struct {
	ModelConfiguration rerankModelConfiguration `json:"modelConfiguration"`
	NumberOfResults    *int                     `json:"numberOfResults,omitempty"`
}

type rerankModelConfiguration struct {
	ModelARN                     string            `json:"modelArn"`
	AdditionalModelRequestFields jsonobject.Object `json:"additionalModelRequestFields,omitempty"`
}

type rerankSource struct {
	Type                 string               `json:"type"`
	InlineDocumentSource inlineDocumentSource `json:"inlineDocumentSource"`
}

type inlineDocumentSource struct {
	Type         string          `json:"type"`
	TextDocument *rerankText     `json:"textDocument,omitempty"`
	JSONDocument json.RawMessage `json:"jsonDocument,omitempty"`
}

type rerankResponse struct {
	Results []rerankResult `json:"results"`
}

type rerankResult struct {
	Index          *int     `json:"index"`
	RelevanceScore *float64 `json:"relevanceScore"`
}

// Rerank ranks the documents by relevance to the query with the Bedrock
// Agent Runtime Rerank API.
func (m *RerankingModel) Rerank(ctx context.Context, options model.RerankOptions) (*model.RerankResult, error) {
	rerankOptions, _, err := namespacedValue[RerankingOptions](options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	region, err := m.provider.config.LoadRegion()
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(ctx, bedrock.AgentRuntime, "/rerank", options.Headers)
	if err != nil {
		return nil, err
	}

	body := rerankRequest{
		NextToken: rerankOptions.NextToken,
		Queries: []rerankQuery{
			{
				TextQuery: rerankText{
					Text: options.Query,
				},
				Type: "TEXT",
			},
		},
		RerankingConfiguration: rerankingConfiguration{
			BedrockRerankingConfiguration: bedrockRerankingConfiguration{
				ModelConfiguration: rerankModelConfiguration{
					ModelARN:                     "arn:aws:bedrock:" + region + "::foundation-model/" + m.modelID,
					AdditionalModelRequestFields: rerankOptions.AdditionalModelRequestFields,
				},
				NumberOfResults: options.TopN,
			},
			Type: "BEDROCK_RERANKING_MODEL",
		},
		Sources: rerankSources(options),
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded rerankResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	ranking, err := decoded.ranking()
	if err != nil {
		return nil, err
	}

	return &model.RerankResult{
		Ranking: ranking,
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// rerankSources returns the documents as inline text or JSON sources.
func rerankSources(options model.RerankOptions) []rerankSource {
	if options.ObjectDocuments != nil {
		sources := make([]rerankSource, 0, len(options.ObjectDocuments))
		for _, document := range options.ObjectDocuments {
			sources = append(sources, rerankSource{
				Type: "INLINE",
				InlineDocumentSource: inlineDocumentSource{
					Type:         "JSON",
					JSONDocument: document,
				},
			})
		}

		return sources
	}

	sources := make([]rerankSource, 0, len(options.TextDocuments))
	for _, document := range options.TextDocuments {
		sources = append(sources, rerankSource{
			Type: "INLINE",
			InlineDocumentSource: inlineDocumentSource{
				Type: "TEXT",
				TextDocument: &rerankText{
					Text: document,
				},
			},
		})
	}

	return sources
}

func (r rerankResponse) ranking() ([]model.RankedDocument, error) {
	if r.Results == nil {
		return nil, errMissingField("results")
	}

	ranking := make([]model.RankedDocument, 0, len(r.Results))
	for _, result := range r.Results {
		if result.Index == nil || result.RelevanceScore == nil {
			return nil, errMissingField("result index or relevance score")
		}

		ranking = append(ranking, model.RankedDocument{
			Index:          *result.Index,
			RelevanceScore: *result.RelevanceScore,
		})
	}

	return ranking, nil
}
