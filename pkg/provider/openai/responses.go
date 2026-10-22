package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

// responsesURLPattern matches the URLs the Responses API downloads itself.
var responsesURLPattern = regexp.MustCompile(`^https?://.*$`)

// ResponsesModel is a language model served by the OpenAI Responses API.
type ResponsesModel struct {
	modelID string
	config  config
}

// newResponsesModel returns a Responses API model.
func newResponsesModel(modelID string, c config) *ResponsesModel {
	return &ResponsesModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as "openai.responses".
func (m *ResponsesModel) Provider() string {
	return m.config.modelProvider("responses")
}

// ModelID returns the model identifier.
func (m *ResponsesModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that the API downloads HTTP image and PDF URLs.
func (m *ResponsesModel) SupportedURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{
		"image/*": {
			responsesURLPattern,
		},
		"application/pdf": {
			responsesURLPattern,
		},
	}
}

// Generate performs one non-streaming Responses API call.
func (m *ResponsesModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	call, err := m.prepare(options)
	if err != nil {
		return nil, err
	}

	request, err := m.config.request(ctx, "/responses", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, call.body)
	if err != nil {
		return nil, err
	}

	var decoded responsesResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if err := responsesResponseFailure(decoded, request.URL, response); err != nil {
		return nil, err
	}

	mapper := newResponsesOutputMapper(call.context)
	if err := mapper.mapOutput(decoded.Output); err != nil {
		return nil, err
	}

	return &model.GenerateResult{
		Content: mapper.content,
		FinishReason: model.FinishReason{
			Unified: responsesFinishReason(decoded.incompleteReason(), mapper.hasFunctionCall),
			Raw:     decoded.incompleteReason(),
		},
		Usage: responsesConvertUsage(decoded.Usage),
		ProviderMetadata: model.ProviderMetadata{
			call.context.namespace: ResponsesMetadata{
				ResponseID:       decoded.ID,
				Logprobs:         mapper.logprobs,
				ServiceTier:      decoded.ServiceTier,
				ReasoningContext: decoded.reasoningContext(),
			},
		},
		Warnings: call.warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID:        decoded.ID,
				Timestamp: responsesTimestamp(decoded.CreatedAt),
				ModelID:   decoded.Model,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// Stream performs one streaming Responses API call. Errors that the API
// reports before any output are returned instead of a stream.
func (m *ResponsesModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	call, err := m.prepare(options)
	if err != nil {
		return nil, err
	}

	request, err := m.config.request(ctx, "/responses", options.Headers)
	if err != nil {
		return nil, err
	}

	body := call.body
	body.Stream = true

	stream, err := apicall.PostJSONStream(ctx, request, body)
	if err != nil {
		return nil, err
	}

	events := newResponsesEventReader(stream.Body)

	failure := responsesStreamFailure{
		url:     request.URL,
		headers: stream.Headers,
	}
	if err := events.checkEarlyError(failure); err != nil {
		return nil, errors.Join(err, events.close())
	}

	state := newResponsesStreamState(call, options, failure)

	result := model.NewStreamResult(state.parts(events), events.close)
	result.Request = model.RequestInfo{
		Body: stream.RequestBody,
	}
	result.Response = model.StreamResponseInfo{
		Headers: stream.Headers,
	}

	return result, nil
}

// responsesCall is a prepared request with the context needed to read its
// response.
type responsesCall struct {
	body     responsesRequest
	warnings []model.Warning
	context  responsesCallContext
}

// responsesCallContext is the request information that response mapping
// needs.
type responsesCallContext struct {
	// generateID names sources and tool calls that have no wire ID.
	generateID func() string

	// namespace keys provider metadata, "openai" or "azure".
	namespace string

	toolNames toolnames.Mapping

	// webSearchToolName is the application name of the web search tool, or
	// "web_search" when the call has none.
	webSearchToolName string

	// isShellProviderExecuted reports that shell commands run in an OpenAI
	// container.
	isShellProviderExecuted bool

	// store reports that store was explicitly enabled.
	store bool

	// collectLogprobs reports that logprobs were requested.
	collectLogprobs bool

	// functionToolNames holds the declared function tools.
	functionToolNames map[string]bool

	// approvalToolCallIDs maps MCP approval request IDs from the prompt to
	// the tool call IDs created for them.
	approvalToolCallIDs map[string]string
}

// toolName returns the application name of a provider tool.
func (c responsesCallContext) toolName(providerName string) string {
	return c.toolNames.ToApplication(providerName)
}

// itemMetadata returns provider metadata that identifies an output item.
func (c responsesCallContext) itemMetadata(itemID string) model.ProviderMetadata {
	return model.ProviderMetadata{
		c.namespace: ResponsesItemMetadata{
			ItemID: itemID,
		},
	}
}

// metadata wraps value in the provider namespace.
func (c responsesCallContext) metadata(value any) model.ProviderMetadata {
	return model.ProviderMetadata{
		c.namespace: value,
	}
}

// responsesResponseFailure converts an error body or a missing output into
// an API call error.
func responsesResponseFailure(response responsesResponse, url string, raw *apicall.Response) error {
	if response.Error != nil {
		return &model.APICallError{
			Message:         response.Error.Message,
			URL:             url,
			StatusCode:      http.StatusBadRequest,
			ResponseHeaders: raw.Headers,
			ResponseBody:    string(raw.Body),
		}
	}

	if response.Output != nil {
		return nil
	}

	message := "Responses API returned no output"
	if reason := response.incompleteReason(); reason != "" {
		message = fmt.Sprintf("%s (%s)", message, reason)
	}

	return &model.APICallError{
		Message:         message,
		URL:             url,
		StatusCode:      http.StatusInternalServerError,
		ResponseHeaders: raw.Headers,
		ResponseBody:    string(raw.Body),
	}
}

func (r responsesResponse) incompleteReason() string {
	if r.IncompleteDetails == nil {
		return ""
	}

	return r.IncompleteDetails.Reason
}

func (r responsesResponse) reasoningContext() string {
	if r.Reasoning == nil {
		return ""
	}

	return r.Reasoning.Context
}

// responsesTimestamp converts Unix seconds into a time. Nil gives the zero
// time.
func responsesTimestamp(seconds *float64) time.Time {
	if seconds == nil {
		return time.Time{}
	}

	return time.Unix(0, int64(*seconds*float64(time.Second)))
}

// responsesFinishReason maps the incomplete reason of a response. Client
// tool calls turn a normal stop into a tool call finish.
func responsesFinishReason(incompleteReason string, hasFunctionCall bool) model.UnifiedFinishReason {
	switch incompleteReason {
	case "":
		if hasFunctionCall {
			return model.FinishToolCalls
		}

		return model.FinishStop
	case "max_output_tokens":
		return model.FinishLength
	case "content_filter":
		return model.FinishContentFilter
	default:
		if hasFunctionCall {
			return model.FinishToolCalls
		}

		return model.FinishOther
	}
}

// responsesConvertUsage converts API usage. Nil usage gives unknown counts.
func responsesConvertUsage(usage *responsesUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	cachedTokens := 0
	var cacheWriteTokens *int

	if details := usage.InputTokensDetails; details != nil {
		if details.CachedTokens != nil {
			cachedTokens = *details.CachedTokens
		}

		cacheWriteTokens = details.CacheWriteTokens
	}

	reasoningTokens := 0
	if details := usage.OutputTokensDetails; details != nil && details.ReasoningTokens != nil {
		reasoningTokens = *details.ReasoningTokens
	}

	noCacheTokens := usage.InputTokens - cachedTokens
	if cacheWriteTokens != nil {
		noCacheTokens -= *cacheWriteTokens
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:      new(usage.InputTokens),
			NoCache:    new(noCacheTokens),
			CacheRead:  new(cachedTokens),
			CacheWrite: cacheWriteTokens,
		},
		OutputTokens: model.OutputTokens{
			Total:     new(usage.OutputTokens),
			Text:      new(usage.OutputTokens - reasoningTokens),
			Reasoning: new(reasoningTokens),
		},
		Raw: usage.Raw,
	}
}

// responsesEncodeJSON encodes a value whose type cannot fail to encode, such
// as a struct of strings, numbers and decoded raw JSON.
func responsesEncodeJSON[T any](value T) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("openai: encode %T: %v", value, err))
	}

	return encoded
}

// responsesEscapeJSONString returns text escaped for embedding inside a JSON
// string literal, without the surrounding quotes.
func responsesEscapeJSONString(text string) string {
	var buffer bytes.Buffer

	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)

	if err := encoder.Encode(text); err != nil {
		panic(fmt.Sprintf("openai: encode string: %v", err))
	}

	encoded := strings.TrimSuffix(buffer.String(), "\n")

	return encoded[1 : len(encoded)-1]
}
