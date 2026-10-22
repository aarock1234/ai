package perplexity

import (
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// AgentMetadata is the result provider metadata of Agent API models, stored
// under "perplexity". Null members were not reported.
type AgentMetadata struct {
	Usage UsageMetadata `json:"usage"`

	// Images is always null in the Agent API; it keeps the shape of the
	// former chat completions metadata.
	Images []string `json:"images"`

	Cost      *CostMetadata               `json:"cost"`
	ToolCalls map[string]ToolCallsDetails `json:"toolCalls"`
}

// UsageMetadata reports search activity.
type UsageMetadata struct {
	// CitationTokens is always null in the Agent API.
	CitationTokens *int `json:"citationTokens"`

	// NumSearchQueries sums the invocations of search tools.
	NumSearchQueries *float64 `json:"numSearchQueries"`
}

// CostMetadata is the cost of a response.
type CostMetadata struct {
	InputTokensCost  *float64 `json:"inputTokensCost"`
	OutputTokensCost *float64 `json:"outputTokensCost"`

	// RequestCost is always null in the Agent API.
	RequestCost *float64 `json:"requestCost"`

	TotalCost         *float64 `json:"totalCost"`
	Currency          *string  `json:"currency"`
	CacheCreationCost *float64 `json:"cacheCreationCost"`
	CacheReadCost     *float64 `json:"cacheReadCost"`
	ToolCallsCost     *float64 `json:"toolCallsCost"`
}

// ToolCallsDetails counts the invocations of one native tool.
type ToolCallsDetails struct {
	Invocation *float64 `json:"invocation"`
}

// SearchSourceMetadata is the provider metadata of a search result source.
type SearchSourceMetadata struct {
	ResultID    *int    `json:"resultId"`
	Snippet     *string `json:"snippet"`
	Date        *string `json:"date"`
	LastUpdated *string `json:"lastUpdated"`
	Source      *string `json:"source"`
}

// FetchedSourceMetadata is the provider metadata of a fetched URL source.
type FetchedSourceMetadata struct {
	Snippet *string `json:"snippet"`
}

// ToolCallMetadata is the provider metadata of a function call. Replaying
// the thought signature lets the model verify its reasoning.
type ToolCallMetadata struct {
	ItemID           *string `json:"itemId"`
	ThoughtSignature string  `json:"thoughtSignature,omitempty"`
}

// EmbeddingMetadata is the result provider metadata of embedding models,
// stored under "perplexity" when the cost is reported.
type EmbeddingMetadata struct {
	Cost EmbeddingCost `json:"cost"`
}

// EmbeddingCost is the cost of an embedding call.
type EmbeddingCost struct {
	InputCost *float64 `json:"inputCost"`
	TotalCost *float64 `json:"totalCost"`
	Currency  *string  `json:"currency"`
}

// agentMetadata describes the usage, cost and native tool calls.
func agentMetadata(usage *agentUsage) model.ProviderMetadata {
	metadata := AgentMetadata{}

	if usage != nil {
		metadata.Cost = costMetadata(usage.Cost)

		if usage.ToolCallsDetails != nil {
			metadata.Usage.NumSearchQueries = new(searchQueries(usage.ToolCallsDetails))
			metadata.ToolCalls = make(map[string]ToolCallsDetails, len(usage.ToolCallsDetails))

			for name, details := range usage.ToolCallsDetails {
				metadata.ToolCalls[name] = ToolCallsDetails(details)
			}
		}
	}

	return model.ProviderMetadata{
		Namespace: metadata,
	}
}

// searchQueries sums the invocations of tools whose name contains
// "search".
func searchQueries(details map[string]toolDetails) float64 {
	total := 0.0
	for name, detail := range details {
		if strings.Contains(name, "search") {
			total += optional.Or(detail.Invocation, 0)
		}
	}

	return total
}

func costMetadata(cost *agentCost) *CostMetadata {
	if cost == nil {
		return nil
	}

	return &CostMetadata{
		InputTokensCost:   cost.InputCost,
		OutputTokensCost:  cost.OutputCost,
		TotalCost:         cost.TotalCost,
		Currency:          cost.Currency,
		CacheCreationCost: cost.CacheCreationCost,
		CacheReadCost:     cost.CacheReadCost,
		ToolCallsCost:     cost.ToolCallsCost,
	}
}

// convertUsage converts Agent API usage. Missing usage reports no counts.
func convertUsage(usage *agentUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	inputTokens := optional.Or(usage.InputTokens, 0)
	outputTokens := optional.Or(usage.OutputTokens, 0)

	cacheReadTokens := 0
	cacheWriteTokens := 0
	if details := usage.InputTokensDetails; details != nil {
		switch {
		case details.CacheReadInputTokens != nil:
			cacheReadTokens = *details.CacheReadInputTokens
		case details.CachedTokens != nil:
			cacheReadTokens = *details.CachedTokens
		}

		cacheWriteTokens = optional.Or(details.CacheCreationInputTokens, 0)
	}

	reasoningTokens := 0
	if usage.OutputTokensDetails != nil {
		reasoningTokens = optional.Or(usage.OutputTokensDetails.ReasoningTokens, 0)
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:      new(inputTokens),
			NoCache:    new(max(0, inputTokens-cacheReadTokens-cacheWriteTokens)),
			CacheRead:  new(cacheReadTokens),
			CacheWrite: new(cacheWriteTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(outputTokens),
			Text:      new(max(0, outputTokens-reasoningTokens)),
			Reasoning: new(reasoningTokens),
		},
		Raw: usage.raw,
	}
}

// finishReason maps the response status and incomplete reason. The raw
// reason is the incomplete reason, or else the status.
func finishReason(response *agentResponse, hasFunctionCall bool) model.FinishReason {
	incompleteReason := ""
	if response.IncompleteDetails != nil {
		incompleteReason = response.IncompleteDetails.Reason
	}

	raw := incompleteReason
	if response.IncompleteDetails == nil {
		raw = response.Status
	}

	return model.FinishReason{
		Unified: unifiedFinishReason(response.Status, incompleteReason, hasFunctionCall),
		Raw:     raw,
	}
}

func unifiedFinishReason(status string, incompleteReason string, hasFunctionCall bool) model.UnifiedFinishReason {
	switch incompleteReason {
	case "max_output_tokens":
		return model.FinishLength
	case "content_filter":
		return model.FinishContentFilter
	}

	switch {
	case status == "completed" && !hasFunctionCall:
		return model.FinishStop
	case status == "requires_action":
		return model.FinishToolCalls
	case status == "failed":
		return model.FinishError
	case hasFunctionCall:
		return model.FinishToolCalls
	default:
		return model.FinishOther
	}
}

// responseMetadata converts the response identification fields.
func responseMetadata(response *agentResponse) model.ResponseMetadata {
	metadata := model.ResponseMetadata{
		ID:      response.ID,
		ModelID: response.Model,
	}

	if response.CreatedAt != nil {
		metadata.Timestamp = time.Unix(int64(*response.CreatedAt), 0)
	}

	return metadata
}
