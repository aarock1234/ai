package perplexity

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// inputItem is a messageInput, functionCallInput or functionOutputInput.
type inputItem interface {
	isInputItem()
}

// messageInput content is a string, or parts when it contains images.
type messageInput struct {
	Type    string       `json:"type"`
	Role    string       `json:"role"`
	Content inputContent `json:"content"`
}

type inputContent struct {
	text  string
	parts []inputPart
}

// MarshalJSON encodes the parts, or the text when there are no parts.
func (c inputContent) MarshalJSON() ([]byte, error) {
	if c.parts != nil {
		return json.Marshal(c.parts)
	}

	return json.Marshal(c.text)
}

// inputPart is one part of a user message. Text parts set Text and image
// parts set ImageURL.
type inputPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type functionCallInput struct {
	Type             string `json:"type"`
	CallID           string `json:"call_id"`
	Name             string `json:"name"`
	Arguments        string `json:"arguments"`
	ThoughtSignature string `json:"thought_signature,omitempty"`
}

type functionOutputInput struct {
	Type             string `json:"type"`
	CallID           string `json:"call_id"`
	Name             string `json:"name,omitempty"`
	Output           string `json:"output"`
	ThoughtSignature string `json:"thought_signature,omitempty"`
}

func (messageInput) isInputItem()        {}
func (functionCallInput) isInputItem()   {}
func (functionOutputInput) isInputItem() {}

type functionTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict,omitempty"`
}

type responseFormat struct {
	Type       string     `json:"type"`
	JSONSchema jsonSchema `json:"json_schema"`
}

type jsonSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema"`
	Strict      bool            `json:"strict"`
}

type agentResponse struct {
	ID                string             `json:"id"`
	CreatedAt         *float64           `json:"created_at"`
	Model             string             `json:"model"`
	Output            []outputItem       `json:"output"`
	Status            string             `json:"status"`
	IncompleteDetails *incompleteDetails `json:"incomplete_details"`
	Error             *responseError     `json:"error"`
	Usage             *agentUsage        `json:"usage"`
}

type incompleteDetails struct {
	Reason string `json:"reason"`
}

// responseError is the error of a failed response or stream.
type responseError struct {
	Code    *string `json:"code"`
	Message string  `json:"message"`
	Type    *string `json:"type"`
}

// Error returns the provider's message.
func (e *responseError) Error() string {
	return e.Message
}

// Output item types handled by the model. Other native tool traces remain
// available in raw responses and raw chunks.
const (
	itemMessage         = "message"
	itemSearchResults   = "search_results"
	itemFetchURLResults = "fetch_url_results"
	itemFunctionCall    = "function_call"
)

// outputItem is one response output item. Only handled item types decode
// their fields, because other traces use different shapes.
type outputItem struct {
	Type string

	Message      *messageItem
	SearchResult *searchResultsItem
	FetchResults *fetchResultsItem
	FunctionCall *functionCallItem
}

type itemHeader struct {
	Type string `json:"type"`
}

// UnmarshalJSON decodes the item variant named by its type.
func (i *outputItem) UnmarshalJSON(data []byte) error {
	var header itemHeader
	if err := json.Unmarshal(data, &header); err != nil {
		return err
	}

	*i = outputItem{
		Type: header.Type,
	}

	switch header.Type {
	case itemMessage:
		return decodeItem(data, &i.Message)

	case itemSearchResults:
		return decodeItem(data, &i.SearchResult)

	case itemFetchURLResults:
		return decodeItem(data, &i.FetchResults)

	case itemFunctionCall:
		return decodeItem(data, &i.FunctionCall)

	default:
		return nil
	}
}

func decodeItem[T any](data []byte, target **T) error {
	var item T
	if err := json.Unmarshal(data, &item); err != nil {
		return fmt.Errorf("decode output item: %w", err)
	}

	*target = &item

	return nil
}

type messageItem struct {
	ID      *string       `json:"id"`
	Content []contentPart `json:"content"`
}

type contentPart struct {
	Type        string       `json:"type"`
	Text        *string      `json:"text"`
	Annotations []annotation `json:"annotations"`
}

type annotation struct {
	Type  *string `json:"type"`
	URL   *string `json:"url"`
	Title *string `json:"title"`
}

type searchResultsItem struct {
	Results []searchResult `json:"results"`
}

// searchResult is a web search result.
type searchResult struct {
	ID          *int    `json:"id"`
	Title       string  `json:"title"`
	URL         string  `json:"url"`
	Snippet     *string `json:"snippet"`
	Date        *string `json:"date"`
	LastUpdated *string `json:"last_updated"`
	Source      *string `json:"source"`
}

type fetchResultsItem struct {
	Contents []fetchedContent `json:"contents"`
}

// fetchedContent is a fetched URL.
type fetchedContent struct {
	Title   string  `json:"title"`
	URL     string  `json:"url"`
	Snippet *string `json:"snippet"`
}

type functionCallItem struct {
	ID               *string `json:"id"`
	CallID           *string `json:"call_id"`
	Name             *string `json:"name"`
	Arguments        *string `json:"arguments"`
	ThoughtSignature *string `json:"thought_signature"`
}

// complete reports whether the call has an ID, a name and arguments.
func (c *functionCallItem) complete() bool {
	return c != nil && c.CallID != nil && c.Name != nil && c.Arguments != nil
}

// agentUsage is the token usage and cost of a response. The raw object is
// kept as the usage Raw value.
type agentUsage struct {
	InputTokens         *int                   `json:"input_tokens"`
	OutputTokens        *int                   `json:"output_tokens"`
	TotalTokens         *int                   `json:"total_tokens"`
	InputTokensDetails  *inputTokensDetails    `json:"input_tokens_details"`
	OutputTokensDetails *outputTokensDetails   `json:"output_tokens_details"`
	ToolCallsDetails    map[string]toolDetails `json:"tool_calls_details"`
	Cost                *agentCost             `json:"cost"`

	raw json.RawMessage
}

type inputTokensDetails struct {
	CachedTokens             *int `json:"cached_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
}

type outputTokensDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

type toolDetails struct {
	Invocation *float64 `json:"invocation"`
}

type agentCost struct {
	Currency          *string  `json:"currency"`
	InputCost         *float64 `json:"input_cost"`
	OutputCost        *float64 `json:"output_cost"`
	TotalCost         *float64 `json:"total_cost"`
	CacheCreationCost *float64 `json:"cache_creation_cost"`
	CacheReadCost     *float64 `json:"cache_read_cost"`
	ToolCallsCost     *float64 `json:"tool_calls_cost"`
}

// UnmarshalJSON decodes the usage and keeps the raw object.
func (u *agentUsage) UnmarshalJSON(data []byte) error {
	type plain agentUsage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = agentUsage(decoded)
	u.raw = bytes.Clone(data)

	return nil
}

// agentChunk is one streamed event. The fields of each event type share
// their JSON names.
type agentChunk struct {
	Type           string           `json:"type"`
	SequenceNumber *float64         `json:"sequence_number"`
	Response       *agentResponse   `json:"response"`
	Item           *outputItem      `json:"item"`
	OutputIndex    *int             `json:"output_index"`
	ItemID         *string          `json:"item_id"`
	ContentIndex   *int             `json:"content_index"`
	Delta          *string          `json:"delta"`
	Text           *string          `json:"text"`
	Thought        *string          `json:"thought"`
	Results        []searchResult   `json:"results"`
	Contents       []fetchedContent `json:"contents"`
	Error          *responseError   `json:"error"`
}
