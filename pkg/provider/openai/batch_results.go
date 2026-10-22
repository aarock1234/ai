package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/jsonlines"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/internal/urlpath"
	"github.com/aarock1234/ai/pkg/model"
)

// BatchResults streams the results of a finished batch from its output and
// error files. Both are downloaded lazily, one at a time.
func (b *Batch) BatchResults(ctx context.Context, options model.BatchOperationOptions) (*model.BatchResultStream, error) {
	batch, err := b.retrieveBatch(ctx, options)
	if err != nil {
		return nil, err
	}

	status := batch.status()
	if status.State == model.BatchPending {
		return nil, fmt.Errorf("%w: OpenAI batch %q is not complete", model.ErrInvalidArgument, options.BatchID)
	}

	var fileIDs []string
	for _, fileID := range []*string{batch.OutputFileID, batch.ErrorFileID} {
		if fileID != nil {
			fileIDs = append(fileIDs, *fileID)
		}
	}

	if status.State == model.BatchCompleted && len(fileIDs) == 0 {
		return nil, fmt.Errorf("%w: OpenAI batch %q completed without batch output", model.ErrInvalidResponse, options.BatchID)
	}

	reader := &batchResultReader{
		batch:   b,
		ctx:     ctx,
		options: options,
		fileIDs: fileIDs,
		lines:   jsonlines.NewReader(b.config.batchMaxLineBytes),
	}

	return model.NewBatchResultStream(reader.items(), reader.lines.Close), nil
}

// batchResultReader reads the JSON Lines result files of one batch. Its
// lines reader owns the body of the file being read.
type batchResultReader struct {
	batch   *Batch
	ctx     context.Context
	options model.BatchOperationOptions
	fileIDs []string
	lines   *jsonlines.Reader
}

// items yields the item result of every line of every file.
func (r *batchResultReader) items() iter.Seq2[model.BatchItemResult, error] {
	return func(yield func(model.BatchItemResult, error) bool) {
		for _, fileID := range r.fileIDs {
			if !r.readFile(fileID, yield) {
				return
			}
		}
	}
}

// readFile yields the results of one file and reports whether to continue
// with the next.
func (r *batchResultReader) readFile(fileID string, yield func(model.BatchItemResult, error) bool) bool {
	if err := r.open(fileID); err != nil {
		yield(nil, err)

		return false
	}

	for line, err := range r.lines.Lines() {
		if err != nil {
			yield(nil, err)

			return false
		}

		item, err := r.batch.convertResultLine(line)
		if !yield(item, err) || err != nil {
			return false
		}
	}

	return true
}

func (r *batchResultReader) open(fileID string) error {
	request, err := r.batch.config.request(r.ctx, "/files/"+urlpath.Escape(fileID)+"/content", r.options.Headers)
	if err != nil {
		return err
	}

	stream, err := apicall.GetStream(r.ctx, request)
	if err != nil {
		return err
	}

	return r.lines.Open(stream.Body)
}

// batchResultLine is one line of a result file.
type batchResultLine struct {
	CustomID string             `json:"custom_id"`
	Response *batchLineResponse `json:"response"`
	Error    *batchLineError    `json:"error"`
}

type batchLineResponse struct {
	StatusCode int             `json:"status_code"`
	RequestID  *string         `json:"request_id"`
	Body       json.RawMessage `json:"body"`
}

type batchLineError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// convertResultLine converts one line into a text item result. Lines that
// cannot be decoded end the stream; request failures become failed items.
func (b *Batch) convertResultLine(line []byte) (model.BatchItemResult, error) {
	var decoded batchResultLine
	if err := json.Unmarshal(line, &decoded); err != nil {
		return nil, fmt.Errorf("%w: decode batch result line: %w", model.ErrInvalidResponse, err)
	}

	if decoded.CustomID == "" {
		return nil, fmt.Errorf("%w: batch result line has no custom_id", model.ErrInvalidResponse)
	}

	if decoded.Error != nil {
		return lineErrorResult(decoded.CustomID, decoded.Error), nil
	}

	if decoded.Response == nil {
		return failedBatchItem(decoded.CustomID, &model.BatchError{
			Message: "OpenAI returned a batch result without a response or error.",
			Code:    "invalid_batch_result",
		}), nil
	}

	statusCode := decoded.Response.StatusCode
	if statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices {
		return failedBatchItem(decoded.CustomID, batchRequestError(decoded.Response.Body, statusCode)), nil
	}

	result, batchErr := b.convertResponsesResult(decoded.Response.Body)
	if batchErr != nil {
		return failedBatchItem(decoded.CustomID, batchErr), nil
	}

	return model.TextBatchItemResult{
		ID:     decoded.CustomID,
		Status: model.BatchItemSucceeded,
		Result: result,
	}, nil
}

// lineErrorResult maps a line error, recognizing cancelled and expired
// requests.
func lineErrorResult(customID string, lineError *batchLineError) model.BatchItemResult {
	batchErr := &model.BatchError{
		Message: lineError.Message,
		Code:    lineError.Code,
	}

	status := model.BatchItemFailed

	switch lineError.Code {
	case "batch_cancelled":
		status = model.BatchItemCancelled
	case "batch_expired":
		status = model.BatchItemExpired
	}

	return model.TextBatchItemResult{
		ID:     customID,
		Status: status,
		Error:  batchErr,
	}
}

func failedBatchItem(customID string, batchErr *model.BatchError) model.BatchItemResult {
	return model.TextBatchItemResult{
		ID:     customID,
		Status: model.BatchItemFailed,
		Error:  batchErr,
	}
}

// batchErrorBody is the error body of a failed batch request.
type batchErrorBody struct {
	Error *struct {
		Message string          `json:"message"`
		Type    *string         `json:"type"`
		Code    json.RawMessage `json:"code"`
	} `json:"error"`
}

// batchRequestError converts the body of a failed request.
func batchRequestError(body json.RawMessage, statusCode int) *model.BatchError {
	var decoded batchErrorBody
	if err := json.Unmarshal(body, &decoded); err != nil || decoded.Error == nil {
		return &model.BatchError{
			Message:    fmt.Sprintf("OpenAI batch request failed with status code %d.", statusCode),
			StatusCode: statusCode,
		}
	}

	return &model.BatchError{
		Message:    decoded.Error.Message,
		Type:       valueOr(decoded.Error.Type, ""),
		Code:       jsonScalarText(decoded.Error.Code),
		StatusCode: statusCode,
	}
}

// jsonScalarText returns a JSON string's value or another scalar's literal
// text, and an empty string for null.
func jsonScalarText(raw json.RawMessage) string {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}

	trimmed := string(bytes.TrimSpace(raw))
	if trimmed == "null" {
		return ""
	}

	return trimmed
}

// convertResponsesResult converts a successful Responses API body. Bodies
// with an error, without output or with output batches cannot convert
// become a batch error.
func (b *Batch) convertResponsesResult(body json.RawMessage) (*model.GenerateResult, *model.BatchError) {
	var response responsesResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, &model.BatchError{
			Message: "OpenAI returned an invalid Responses batch result.",
			Code:    "invalid_response",
		}
	}

	if response.Error != nil {
		return nil, &model.BatchError{
			Message: response.Error.Message,
			Type:    response.Error.Type,
			Code:    response.Error.Code,
		}
	}

	if response.Output == nil {
		message := "OpenAI Responses returned no output."
		if reason := response.incompleteReason(); reason != "" {
			message = fmt.Sprintf("OpenAI Responses returned no output (%s).", reason)
		}

		return nil, &model.BatchError{
			Message: message,
			Code:    "invalid_response",
		}
	}

	converter := newBatchOutputConverter(b.config.generateID)
	if batchErr := converter.convert(response.Output); batchErr != nil {
		return nil, batchErr
	}

	return &model.GenerateResult{
		Content: converter.content,
		FinishReason: model.FinishReason{
			Unified: responsesFinishReason(response.incompleteReason(), converter.hasFunctionCall),
			Raw:     response.incompleteReason(),
		},
		Usage: responsesConvertUsage(response.Usage),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: ResponsesMetadata{
				ResponseID:       response.ID,
				Logprobs:         converter.logprobs,
				ServiceTier:      response.ServiceTier,
				ReasoningContext: response.reasoningContext(),
			},
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID:        response.ID,
				Timestamp: responsesTimestamp(response.CreatedAt),
				ModelID:   response.Model,
			},
		},
	}, nil
}

// batchOutputConverter converts the output items batch results support.
// Hosted tool calls are dynamic, because the batch result does not carry
// the request's tool declarations.
type batchOutputConverter struct {
	callContext responsesCallContext

	content         []model.Content
	logprobs        [][]ResponsesLogprob
	hasFunctionCall bool
}

func newBatchOutputConverter(generateID func() string) *batchOutputConverter {
	return &batchOutputConverter{
		callContext: responsesCallContext{
			generateID:        generateID,
			namespace:         Namespace,
			toolNames:         toolnames.New(nil, responsesProviderToolNames),
			webSearchToolName: "web_search",
		},
	}
}

func (c *batchOutputConverter) convert(output responsesOutput) *model.BatchError {
	for _, item := range output {
		if batchErr := c.item(item); batchErr != nil {
			return batchErr
		}
	}

	return nil
}

func (c *batchOutputConverter) item(item responsesOutputItem) *model.BatchError {
	switch typed := item.(type) {
	case responsesReasoningOutput:
		c.reasoning(typed)

	case responsesMessageItem:
		c.message(typed)

	case responsesFunctionCallItem:
		c.hasFunctionCall = true
		c.content = append(c.content, c.callContext.functionCall(typed, nil))

	case responsesCustomToolCallItem:
		c.hasFunctionCall = true
		c.content = append(c.content, c.callContext.customToolCall(typed, nil))

	case responsesWebSearchCallItem:
		c.addHostedCall(c.callContext.webSearchCall(typed.ID), c.callContext.webSearchResult(typed))

	case responsesFileSearchCallItem:
		c.addHostedCall(c.callContext.fileSearchCall(typed.ID), c.callContext.fileSearchResult(typed))

	case responsesCodeInterpreterCallItem:
		c.addHostedCall(
			c.callContext.codeInterpreterCall(typed.ID, typed.Code, typed.ContainerID),
			c.callContext.codeInterpreterResult(typed),
		)

	default:
		return &model.BatchError{
			Message: fmt.Sprintf("OpenAI returned an unsupported %q output item in an AI SDK text batch.", item.responsesOutputType()),
			Code:    "unsupported_content",
		}
	}

	return nil
}

func (c *batchOutputConverter) reasoning(item responsesReasoningOutput) {
	summaries := item.Summary
	if len(summaries) == 0 {
		summaries = []responsesSummaryText{
			{
				Type: "summary_text",
			},
		}
	}

	for _, summary := range summaries {
		c.content = append(c.content, model.Reasoning{
			Text:             summary.Text,
			ProviderMetadata: c.callContext.reasoningMetadata(item.ID, item.EncryptedContent),
		})
	}
}

func (c *batchOutputConverter) message(item responsesMessageItem) {
	for _, part := range item.Content {
		c.content = append(c.content, model.Text{
			Text: part.Text,
		})

		if part.Logprobs != nil {
			c.logprobs = append(c.logprobs, part.Logprobs)
		}
	}
}

// addHostedCall adds a provider-executed call and its result as dynamic
// tool content.
func (c *batchOutputConverter) addHostedCall(call model.ToolCall, result model.ToolResult) {
	call.Dynamic = true
	result.Dynamic = true

	c.content = append(c.content, call, result)
}
