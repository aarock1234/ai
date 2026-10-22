package anthropic

import (
	"encoding/json"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// MessagesMetadata is the provider metadata of a finished call, stored under
// the "anthropic" key of GenerateResult.ProviderMetadata and of the stream's
// Finish part.
type MessagesMetadata struct {
	// Usage is Anthropic's native usage object.
	Usage json.RawMessage `json:"usage"`

	// StopSequence is the stop sequence that ended generation, if any.
	StopSequence *string `json:"stopSequence"`

	// StopDetails explains a refusal when the API provides details. Branch on
	// the content-filter finish reason instead: a refusal may arrive without
	// details.
	StopDetails *StopDetails `json:"stopDetails,omitempty"`

	// InputTransformations lists input blocks that Anthropic transformed
	// before inference, such as dropped mismatched thinking blocks.
	InputTransformations []InputTransformation `json:"inputTransformations,omitempty"`

	// SafeguardResults are the verdicts of the requested safeguards, in the
	// API's snake_case shape.
	SafeguardResults []SafeguardResult `json:"safeguardResults,omitempty"`

	// Iterations breaks usage down by sampling iteration when compaction,
	// the advisor or a server-side fallback ran. Nil otherwise.
	Iterations []UsageIteration `json:"iterations"`

	// Container describes the code execution container, if one was used.
	Container *Container `json:"container"`

	// ContextManagement lists the context edits that were applied.
	ContextManagement *AppliedContextManagement `json:"contextManagement"`
}

// StopDetails explains why a request stopped.
type StopDetails struct {
	// Type is "refusal" for classifier blocks and model refusals.
	Type string `json:"type"`

	// Category is the classifier category, such as "cyber".
	Category string `json:"category,omitempty"`

	// Explanation is a human-readable reason.
	Explanation string `json:"explanation,omitempty"`

	// RecommendedModel is a model to retry directly when a requested
	// fallback could not be attempted.
	RecommendedModel string `json:"recommendedModel,omitempty"`
}

// InputTransformation describes an input block that Anthropic transformed.
type InputTransformation struct {
	Type   string `json:"type"`
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

// SafeguardResult is the verdict of one safeguard.
type SafeguardResult struct {
	Type   string          `json:"type"`
	Status SafeguardStatus `json:"status"`
}

// SafeguardStatus reports whether a safeguard ran. For dangerous_tool_use,
// ToolUses maps tool call IDs to their verdicts.
type SafeguardStatus struct {
	Type     string                      `json:"type"`
	ToolUses map[string]SafeguardVerdict `json:"tool_uses,omitempty"`
}

// SafeguardVerdict is the verdict for one tool call, such as an evaluated
// call with a "flagged" outcome.
type SafeguardVerdict struct {
	Type        string  `json:"type"`
	Outcome     *string `json:"outcome,omitempty"`
	Explanation *string `json:"explanation,omitempty"`
}

// UsageIterationType is the kind of sampling iteration.
type UsageIterationType string

// Usage iteration types. Advisor iterations bill at the advisor model's
// rates and are not included in the call's usage totals.
const (
	IterationCompaction      UsageIterationType = "compaction"
	IterationMessage         UsageIterationType = "message"
	IterationAdvisorMessage  UsageIterationType = "advisor_message"
	IterationFallbackMessage UsageIterationType = "fallback_message"
)

// UsageIteration is the usage of one sampling iteration.
type UsageIteration struct {
	Type UsageIterationType `json:"type"`

	// Model is set for fallback and advisor iterations.
	Model string `json:"model,omitempty"`

	InputTokens              int `json:"inputTokens"`
	OutputTokens             int `json:"outputTokens"`
	CacheCreationInputTokens int `json:"cacheCreationInputTokens,omitempty"`
	CacheReadInputTokens     int `json:"cacheReadInputTokens,omitempty"`
}

// Container describes a code execution container.
type Container struct {
	// ExpiresAt is an RFC 3339 timestamp.
	ExpiresAt string `json:"expiresAt"`

	ID string `json:"id"`

	// Skills are the skills loaded in the container, or nil.
	Skills []ContainerSkill `json:"skills"`
}

// ContainerSkill is a skill loaded in a container.
type ContainerSkill struct {
	Type    SkillType `json:"type"`
	SkillID string    `json:"skillId"`
	Version string    `json:"version"`
}

// AppliedContextManagement lists the context edits applied to a request.
type AppliedContextManagement struct {
	AppliedEdits []AppliedContextEdit `json:"appliedEdits"`
}

// AppliedContextEdit is one applied context edit. The counts apply to the
// strategies named in their comments.
type AppliedContextEdit struct {
	Type ContextEditType `json:"type"`

	// ClearedToolUses applies to ContextEditClearToolUses.
	ClearedToolUses *int `json:"clearedToolUses,omitempty"`

	// ClearedThinkingTurns applies to ContextEditClearThinking.
	ClearedThinkingTurns *int `json:"clearedThinkingTurns,omitempty"`

	// ClearedInputTokens applies to ContextEditClearToolUses and
	// ContextEditClearThinking.
	ClearedInputTokens *int `json:"clearedInputTokens,omitempty"`
}

// MessageStartMetadata is the metadata of the "anthropic.message_start"
// custom stream part.
type MessageStartMetadata struct {
	ID    *string         `json:"id"`
	Model *string         `json:"model"`
	Usage json.RawMessage `json:"usage"`
}

// ReasoningMetadata is the metadata of reasoning content. Replay it as the
// reasoning part's provider options to send the thinking block back.
type ReasoningMetadata struct {
	// Signature verifies a thinking block.
	Signature string `json:"signature,omitempty"`

	// RedactedData is the encrypted content of a redacted thinking block.
	RedactedData string `json:"redactedData,omitempty"`
}

// TextMetadata is the metadata of text content: its citations, or a
// compaction summary when Type is "compaction".
type TextMetadata struct {
	Type      string     `json:"type,omitempty"`
	Signature string     `json:"signature,omitempty"`
	Citations []Citation `json:"citations,omitempty"`
}

// ToolCallMetadata is the metadata of tool calls and tool results. MCP tool
// calls have Type "mcp-tool-use" and a ServerName; toolset member calls have
// a ToolsetName; programmatic calls from code execution have a Caller.
type ToolCallMetadata struct {
	Type        string      `json:"type,omitempty"`
	ServerName  string      `json:"serverName,omitempty"`
	ToolsetName string      `json:"toolsetName,omitempty"`
	Caller      *ToolCaller `json:"caller,omitempty"`
}

// ToolCaller identifies what triggered a tool call. ToolID is the code
// execution tool call ID for programmatic calls.
type ToolCaller struct {
	Type   CallerType `json:"type"`
	ToolID string     `json:"toolId,omitempty"`
}

// FallbackMetadata is the metadata of the "anthropic.fallback" custom
// content, which records a server-side fallback from one model to another.
type FallbackMetadata struct {
	Type string   `json:"type"`
	From ModelRef `json:"from"`
	To   ModelRef `json:"to"`
}

// ModelRef names a model.
type ModelRef struct {
	Model string `json:"model"`
}

// ContainerUploadMetadata is the metadata of the
// "anthropic.container_upload" custom content.
type ContainerUploadMetadata struct {
	FileID string `json:"fileId"`
}

// WebSearchSourceMetadata is the metadata of sources from web search
// results.
type WebSearchSourceMetadata struct {
	PageAge *string `json:"pageAge"`
}

// CitationSourceMetadata is the metadata of sources created from citations.
// The location fields depend on the citation type.
type CitationSourceMetadata struct {
	CitedText       string `json:"citedText"`
	EncryptedIndex  string `json:"encryptedIndex,omitempty"`
	StartPageNumber *int   `json:"startPageNumber,omitempty"`
	EndPageNumber   *int   `json:"endPageNumber,omitempty"`
	StartCharIndex  *int   `json:"startCharIndex,omitempty"`
	EndCharIndex    *int   `json:"endCharIndex,omitempty"`
}

// CitationType is the kind of citation location.
type CitationType string

// Citation types.
const (
	CitationWebSearchResultLocation CitationType = "web_search_result_location"
	CitationPageLocation            CitationType = "page_location"
	CitationCharLocation            CitationType = "char_location"
	CitationContentBlockLocation    CitationType = "content_block_location"
	CitationSearchResultLocation    CitationType = "search_result_location"
)

// Citation is a citation as returned by the API, with its snake_case keys.
// The location fields depend on Type.
type Citation struct {
	Type      CitationType `json:"type"`
	CitedText string       `json:"cited_text"`

	// URL, Title and EncryptedIndex describe web search result locations;
	// Title also describes search result locations.
	URL            string  `json:"url,omitempty"`
	Title          *string `json:"title,omitempty"`
	EncryptedIndex string  `json:"encrypted_index,omitempty"`

	// DocumentIndex, DocumentTitle and FileID describe document locations.
	DocumentIndex *int    `json:"document_index,omitempty"`
	DocumentTitle *string `json:"document_title,omitempty"`
	FileID        *string `json:"file_id,omitempty"`

	StartPageNumber *int `json:"start_page_number,omitempty"`
	EndPageNumber   *int `json:"end_page_number,omitempty"`
	StartCharIndex  *int `json:"start_char_index,omitempty"`
	EndCharIndex    *int `json:"end_char_index,omitempty"`
	StartBlockIndex *int `json:"start_block_index,omitempty"`
	EndBlockIndex   *int `json:"end_block_index,omitempty"`

	// SearchResultIndex and Source describe search result locations.
	SearchResultIndex *int   `json:"search_result_index,omitempty"`
	Source            string `json:"source,omitempty"`
}

// ForwardContainerID returns provider options that reuse the code execution
// container of the most recent step that used one. stepMetadata holds the
// provider metadata of earlier steps, oldest first. It returns nil when no
// step used a container.
func ForwardContainerID(stepMetadata []model.ProviderMetadata) (model.ProviderOptions, error) {
	for _, providerMetadata := range slices.Backward(stepMetadata) {
		metadata, found, err := model.ProviderValue[MessagesMetadata](providerMetadata, Namespace)
		if err != nil {
			return nil, err
		}

		if !found || metadata.Container == nil || metadata.Container.ID == "" {
			continue
		}

		return model.ProviderOptions{
			Namespace: MessagesOptions{
				Container: &ContainerOptions{
					ID: metadata.Container.ID,
				},
			},
		}, nil
	}

	return nil, nil
}
