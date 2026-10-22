package xai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"iter"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// batchResultsPageSize is the number of results fetched per page.
const batchResultsPageSize = 1000

// BatchResults streams the results of a finished batch, fetching one page
// of results at a time.
func (b *Batch) BatchResults(ctx context.Context, options model.BatchOperationOptions) (*model.BatchResultStream, error) {
	batch, err := b.retrieveBatch(ctx, options)
	if err != nil {
		return nil, err
	}

	status, err := batch.status()
	if err != nil {
		return nil, err
	}

	if status.State == model.BatchPending {
		return nil, fmt.Errorf("%w: xAI batch %q is not complete", model.ErrInvalidArgument, options.BatchID)
	}

	return model.NewBatchResultStream(b.resultItems(ctx, options), nil), nil
}

// batchResultsPage is one page of batch results.
type batchResultsPage struct {
	Results         []batchResultEntry `json:"results"`
	PaginationToken *string            `json:"pagination_token"`
}

// batchResultEntry is the result of one request.
type batchResultEntry struct {
	BatchRequestID string            `json:"batch_request_id"`
	BatchResult    *batchResultValue `json:"batch_result"`
	ErrorMessage   *string           `json:"error_message"`
}

type batchResultValue struct {
	Response *batchResultResponse `json:"response"`
	Error    *batchResultError    `json:"error"`
}

// batchResultResponse holds the response of a text or an image request.
type batchResultResponse struct {
	ChatGetCompletion json.RawMessage `json:"chat_get_completion"`
	ImageGeneration   json.RawMessage `json:"image_generation"`
}

// batchResultError is a request error. Its code is a string or a number,
// and zero means no error.
type batchResultError struct {
	Code    json.RawMessage `json:"code"`
	Message *string         `json:"message"`
}

// resultItems yields the item result of every request, page by page.
func (b *Batch) resultItems(ctx context.Context, options model.BatchOperationOptions) iter.Seq2[model.BatchItemResult, error] {
	return func(yield func(model.BatchItemResult, error) bool) {
		paginationToken := ""
		seenTokens := map[string]bool{}

		for {
			page, err := b.resultsPage(ctx, options, paginationToken)
			if err != nil {
				yield(nil, err)

				return
			}

			for _, entry := range page.Results {
				item, err := b.convertResult(ctx, entry)
				if !yield(item, err) || err != nil {
					return
				}
			}

			nextToken, hasNextPage := nextPaginationToken(page, seenTokens)
			if !hasNextPage {
				return
			}

			seenTokens[nextToken] = true
			paginationToken = nextToken
		}
	}
}

// nextPaginationToken returns the token of the page after page. A missing,
// empty or already requested token ends the pagination, so a server that
// repeats a token cannot loop forever.
func nextPaginationToken(page *batchResultsPage, seenTokens map[string]bool) (string, bool) {
	if page.PaginationToken == nil {
		return "", false
	}

	token := *page.PaginationToken

	if token == "" || seenTokens[token] {
		return "", false
	}

	return token, true
}

func (b *Batch) resultsPage(ctx context.Context, options model.BatchOperationOptions, paginationToken string) (*batchResultsPage, error) {
	query := url.Values{}
	query.Set("limit", strconv.Itoa(batchResultsPageSize))

	if paginationToken != "" {
		query.Set("pagination_token", paginationToken)
	}

	path := "/batches/" + url.PathEscape(options.BatchID) + "/results?" + query.Encode()

	request, err := b.provider.apiRequest(path, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, err
	}

	var page batchResultsPage
	if err := apicall.DecodeJSON(response.Body, &page); err != nil {
		return nil, err
	}

	for _, entry := range page.Results {
		if entry.BatchRequestID == "" {
			return nil, fmt.Errorf("%w: batch result has no batch_request_id", model.ErrInvalidResponse)
		}
	}

	return &page, nil
}

// convertResult converts one result into a text or an image item result.
// Only downloading images can fail the stream.
func (b *Batch) convertResult(ctx context.Context, entry batchResultEntry) (model.BatchItemResult, error) {
	if batchErr, failed := entry.failure(); failed {
		status := model.BatchItemFailed
		if entry.isCancelled() {
			status = model.BatchItemCancelled
		}

		return model.TextBatchItemResult{
			ID:     entry.BatchRequestID,
			Status: status,
			Error:  batchErr,
		}, nil
	}

	var response batchResultResponse
	if entry.BatchResult != nil && entry.BatchResult.Response != nil {
		response = *entry.BatchResult.Response
	}

	if wirejson.HasValue(response.ChatGetCompletion) {
		return textBatchItem(entry.BatchRequestID, response.ChatGetCompletion), nil
	}

	if wirejson.HasValue(response.ImageGeneration) {
		return b.imageBatchItem(ctx, entry.BatchRequestID, response.ImageGeneration)
	}

	return invalidTextBatchItem(entry.BatchRequestID), nil
}

// failure returns the error of a failed request. A request failed when it
// has an error message, a nonzero error code, or an error message without a
// code.
func (e batchResultEntry) failure() (*model.BatchError, bool) {
	var resultError batchResultError
	if e.BatchResult != nil && e.BatchResult.Error != nil {
		resultError = *e.BatchResult.Error
	}

	code := batchErrorCode(resultError.Code)
	errorMessage := optional.Or(e.ErrorMessage, "")
	resultMessage := optional.Or(resultError.Message, "")

	hasCode := code != "" && code != "0"
	failed := errorMessage != "" || hasCode || (!wirejson.HasValue(resultError.Code) && resultMessage != "")

	if !failed {
		return nil, false
	}

	message := errorMessage
	if message == "" {
		message = resultMessage
	}

	if message == "" {
		message = "xAI batch request failed."
	}

	batchErr := &model.BatchError{
		Message: message,
	}

	if hasCode {
		batchErr.Code = code
	}

	return batchErr, true
}

// isCancelled reports whether the error code marks a cancelled request.
func (e batchResultEntry) isCancelled() bool {
	if e.BatchResult == nil || e.BatchResult.Error == nil {
		return false
	}

	switch strings.ToLower(batchErrorCode(e.BatchResult.Error.Code)) {
	case "1", "cancelled", "batch_cancelled":
		return true
	default:
		return false
	}
}

// batchErrorCode returns a string code, or a number's literal text, and an
// empty string when the code is absent or null.
func batchErrorCode(raw json.RawMessage) string {
	if !wirejson.HasValue(raw) {
		return ""
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}

	return string(bytes.TrimSpace(raw))
}

func invalidTextBatchItem(id string) model.BatchItemResult {
	return model.TextBatchItemResult{
		ID:     id,
		Status: model.BatchItemFailed,
		Error: &model.BatchError{
			Message: "xAI returned an invalid Responses batch result.",
			Code:    "invalid_response",
		},
	}
}

func invalidImageBatchItem(id string) model.BatchItemResult {
	return model.ImageBatchItemResult{
		ID:     id,
		Status: model.BatchItemFailed,
		Error: &model.BatchError{
			Message: "xAI returned an invalid image batch result.",
			Code:    "invalid_response",
		},
	}
}

// batchTextResponse is a text result. xAI returns text batch results in
// the Chat Completions shape even for Responses API requests.
type batchTextResponse struct {
	ID          *string           `json:"id"`
	Created     *int64            `json:"created"`
	Model       *string           `json:"model"`
	Choices     []batchTextChoice `json:"choices"`
	Usage       *batchTextUsage   `json:"usage"`
	Citations   []string          `json:"citations"`
	ServiceTier *string           `json:"service_tier"`
	Code        *string           `json:"code"`
	Error       *string           `json:"error"`
}

type batchTextChoice struct {
	Message      batchTextMessage `json:"message"`
	Index        int              `json:"index"`
	FinishReason *string          `json:"finish_reason"`
}

type batchTextMessage struct {
	Role             string              `json:"role"`
	Content          *string             `json:"content"`
	ReasoningContent *string             `json:"reasoning_content"`
	ToolCalls        []batchTextToolCall `json:"tool_calls"`
}

type batchTextToolCall struct {
	ID       string                `json:"id"`
	Type     string                `json:"type"`
	Function batchTextToolFunction `json:"function"`
}

type batchTextToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type batchTextUsage struct {
	PromptTokens            int                     `json:"prompt_tokens"`
	CompletionTokens        int                     `json:"completion_tokens"`
	TotalTokens             int                     `json:"total_tokens"`
	CostInUSDTicks          *float64                `json:"cost_in_usd_ticks"`
	PromptTokensDetails     *batchPromptDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *batchCompletionDetails `json:"completion_tokens_details"`
}

type batchPromptDetails struct {
	CachedTokens *int `json:"cached_tokens"`
}

type batchCompletionDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

// decodeBatchTextResponse decodes a text result, reporting false when it
// does not have the expected shape.
func decodeBatchTextResponse(raw json.RawMessage) (batchTextResponse, json.RawMessage, bool) {
	var response batchTextResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return batchTextResponse{}, nil, false
	}

	for _, choice := range response.Choices {
		if choice.Message.Role != "assistant" && choice.Message.Role != "tool" {
			return batchTextResponse{}, nil, false
		}

		for _, toolCall := range choice.Message.ToolCalls {
			if toolCall.ID == "" || toolCall.Type != "function" {
				return batchTextResponse{}, nil, false
			}
		}
	}

	var rawUsage struct {
		Usage json.RawMessage `json:"usage"`
	}

	if err := json.Unmarshal(raw, &rawUsage); err != nil {
		return batchTextResponse{}, nil, false
	}

	return response, rawUsage.Usage, true
}

// textBatchItem converts a text result.
func textBatchItem(id string, raw json.RawMessage) model.BatchItemResult {
	response, rawUsage, ok := decodeBatchTextResponse(raw)
	if !ok {
		return invalidTextBatchItem(id)
	}

	if response.Error != nil {
		return model.TextBatchItemResult{
			ID:     id,
			Status: model.BatchItemFailed,
			Error: &model.BatchError{
				Message: *response.Error,
				Code:    optional.Or(response.Code, ""),
			},
		}
	}

	if len(response.Choices) == 0 {
		return model.TextBatchItemResult{
			ID:     id,
			Status: model.BatchItemFailed,
			Error: &model.BatchError{
				Message: "xAI returned a batch response without any choices.",
				Code:    "invalid_response",
			},
		}
	}

	return model.TextBatchItemResult{
		ID:     id,
		Status: model.BatchItemSucceeded,
		Result: batchTextResult(response, rawUsage),
	}
}

// batchTextResult converts the choices of a text result. Tool choices carry
// the results of server-side tool calls, which are provider-executed.
func batchTextResult(response batchTextResponse, rawUsage json.RawMessage) *model.GenerateResult {
	providerExecuted := providerExecutedCallIDs(response.Choices)

	var content []model.Content
	var lastAssistant *batchTextChoice

	for index := range response.Choices {
		choice := &response.Choices[index]

		if choice.Message.Role == "tool" {
			content = append(content, toolChoiceResults(choice.Message)...)

			continue
		}

		lastAssistant = choice
		content = append(content, assistantChoiceContent(choice.Message, providerExecuted)...)
	}

	for _, citation := range response.Citations {
		content = append(content, model.Source{
			SourceType: model.SourceURL,
			ID:         citation,
			URL:        citation,
		})
	}

	finishReason := ""
	if lastAssistant != nil {
		finishReason = optional.Or(lastAssistant.FinishReason, "")
	}

	result := &model.GenerateResult{
		Content: content,
		FinishReason: model.FinishReason{
			Unified: chatFinishReason(finishReason),
			Raw:     finishReason,
		},
		Usage: batchTextUsageOf(response.Usage, rawUsage),
		Response: model.ResponseInfo{
			ResponseMetadata: batchTextResponseMetadata(response),
		},
	}

	metadata := BatchTextMetadata{
		ServiceTier: optional.Or(response.ServiceTier, ""),
	}

	if response.Usage != nil {
		metadata.CostInUSDTicks = response.Usage.CostInUSDTicks
	}

	if metadata.CostInUSDTicks != nil || response.ServiceTier != nil {
		result.ProviderMetadata = model.ProviderMetadata{
			Namespace: metadata,
		}
	}

	return result
}

func providerExecutedCallIDs(choices []batchTextChoice) map[string]bool {
	callIDs := make(map[string]bool)

	for _, choice := range choices {
		if choice.Message.Role != "tool" {
			continue
		}

		for _, toolCall := range choice.Message.ToolCalls {
			callIDs[toolCall.ID] = true
		}
	}

	return callIDs
}

// toolChoiceResults returns the dynamic results of the tool calls answered
// by a tool choice.
func toolChoiceResults(message batchTextMessage) []model.Content {
	if message.Content == nil {
		return nil
	}

	encoded, err := json.Marshal(*message.Content)
	if err != nil {
		return nil
	}

	content := make([]model.Content, 0, len(message.ToolCalls))
	for _, toolCall := range message.ToolCalls {
		content = append(content, model.ToolResult{
			ToolCallID: toolCall.ID,
			ToolName:   toolCall.Function.Name,
			Result:     encoded,
			Dynamic:    true,
		})
	}

	return content
}

func assistantChoiceContent(message batchTextMessage, providerExecuted map[string]bool) []model.Content {
	var content []model.Content

	if text := optional.Or(message.Content, ""); text != "" {
		content = append(content, model.Text{
			Text: text,
		})
	}

	if reasoningText := optional.Or(message.ReasoningContent, ""); reasoningText != "" {
		content = append(content, model.Reasoning{
			Text: reasoningText,
		})
	}

	for _, toolCall := range message.ToolCalls {
		executed := providerExecuted[toolCall.ID]

		content = append(content, model.ToolCall{
			ToolCallID:       toolCall.ID,
			ToolName:         toolCall.Function.Name,
			Input:            toolCall.Function.Arguments,
			ProviderExecuted: executed,
			Dynamic:          executed,
		})
	}

	return content
}

// chatFinishReason maps a Chat Completions finish reason.
func chatFinishReason(reason string) model.UnifiedFinishReason {
	switch reason {
	case "stop":
		return model.FinishStop
	case "length":
		return model.FinishLength
	case "tool_calls", "function_call":
		return model.FinishToolCalls
	case "content_filter":
		return model.FinishContentFilter
	default:
		return model.FinishOther
	}
}

// batchTextUsageOf converts the usage of a text result. Cached tokens are
// included in the prompt tokens unless they exceed them; reasoning tokens
// are reported in addition to the completion tokens.
func batchTextUsageOf(usage *batchTextUsage, raw json.RawMessage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	cacheReadTokens := 0
	if usage.PromptTokensDetails != nil {
		cacheReadTokens = optional.Or(usage.PromptTokensDetails.CachedTokens, 0)
	}

	reasoningTokens := 0
	if usage.CompletionTokensDetails != nil {
		reasoningTokens = optional.Or(usage.CompletionTokensDetails.ReasoningTokens, 0)
	}

	total := usage.PromptTokens
	noCache := usage.PromptTokens - cacheReadTokens
	if cacheReadTokens > usage.PromptTokens {
		total = usage.PromptTokens + cacheReadTokens
		noCache = usage.PromptTokens
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:     new(total),
			NoCache:   new(noCache),
			CacheRead: new(cacheReadTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(usage.CompletionTokens + reasoningTokens),
			Text:      new(usage.CompletionTokens),
			Reasoning: new(reasoningTokens),
		},
		Raw: raw,
	}
}

func batchTextResponseMetadata(response batchTextResponse) model.ResponseMetadata {
	metadata := model.ResponseMetadata{
		ID:      optional.Or(response.ID, ""),
		ModelID: optional.Or(response.Model, ""),
	}

	if response.Created != nil {
		metadata.Timestamp = time.Unix(*response.Created, 0)
	}

	return metadata
}

// imageBatchItem converts an image result. Moderated images fail the item;
// images delivered by URL are downloaded.
func (b *Batch) imageBatchItem(ctx context.Context, id string, raw json.RawMessage) (model.BatchItemResult, error) {
	var response imageResponse
	if err := json.Unmarshal(raw, &response); err != nil || response.Data == nil {
		return invalidImageBatchItem(id), nil
	}

	for _, item := range response.Data {
		if item.RespectModeration != nil && !*item.RespectModeration {
			return model.ImageBatchItemResult{
				ID:     id,
				Status: model.BatchItemFailed,
				Error: &model.BatchError{
					Message: "Image generation was blocked due to a content policy violation.",
				},
			}, nil
		}
	}

	images, err := b.batchImages(ctx, response.Data)
	if err != nil {
		return nil, err
	}

	return model.ImageBatchItemResult{
		ID:     id,
		Status: model.BatchItemSucceeded,
		Result: &model.ImageResult{
			Images:           images,
			ProviderMetadata: imageMetadata(response),
			Response: model.ResponseInfo{
				ResponseMetadata: model.ResponseMetadata{
					Timestamp: time.Now(),
				},
			},
		},
	}, nil
}

func (b *Batch) batchImages(ctx context.Context, data []imageData) ([][]byte, error) {
	images := make([][]byte, 0, len(data))

	for _, item := range data {
		image, err := b.batchImage(ctx, item)
		if err != nil {
			return nil, err
		}

		images = append(images, image)
	}

	return images, nil
}

func (b *Batch) batchImage(ctx context.Context, item imageData) ([]byte, error) {
	if item.B64JSON != nil {
		image, err := base64.StdEncoding.DecodeString(*item.B64JSON)
		if err != nil {
			return nil, fmt.Errorf("%w: decode image: %w", model.ErrInvalidResponse, err)
		}

		return image, nil
	}

	if item.URL == nil {
		return nil, fmt.Errorf("%w: xAI returned an image without data or a URL", model.ErrInvalidArgument)
	}

	downloaded, err := download.Fetch(ctx, download.Request{
		URL:           *item.URL,
		TrustedOrigin: b.provider.baseURL,
		Client:        b.provider.httpClient,
	})
	if err != nil {
		return nil, err
	}

	return downloaded.Data, nil
}
