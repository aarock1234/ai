package google

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/jsonlines"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/internal/urlpath"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// BatchItemMetadata is the Google metadata of a batch request whose prompt
// Google blocked, stored under ProviderMetadata["google"].
type BatchItemMetadata struct {
	PromptFeedback BatchPromptFeedback `json:"promptFeedback"`
}

// BatchPromptFeedback reports why the prompt of a batch request was
// blocked.
type BatchPromptFeedback struct {
	BlockReason *string `json:"blockReason"`
}

// batchResultLine is one line of a responses file, or one inline response.
type batchResultLine struct {
	Key      *string         `json:"key"`
	Response json.RawMessage `json:"response"`
	Error    *rpcStatus      `json:"error"`
}

// batchResponsePreview is the part of a response that tells whether it has
// any candidate.
type batchResponsePreview struct {
	Candidates     []json.RawMessage `json:"candidates"`
	PromptFeedback *struct {
		BlockReason *string `json:"blockReason"`
	} `json:"promptFeedback"`
}

// BatchResults streams the results of a finished batch, from its inline
// responses or from its downloaded responses file.
func (b *Batch) BatchResults(ctx context.Context, options model.BatchOperationOptions) (*model.BatchResultStream, error) {
	operation, err := b.retrieveBatch(ctx, options)
	if err != nil {
		return nil, err
	}

	status := operation.status()
	if status.State == model.BatchPending {
		return nil, fmt.Errorf("%w: batchId: Google batch %q is not complete", model.ErrInvalidArgument, options.BatchID)
	}

	if inline := batchOutputField(operation, inlinedResponsesOf); inline != nil {
		return model.NewBatchResultStream(b.inlineItems(inline.InlinedResponses), nil), nil
	}

	responsesFile := batchOutputField(operation, responsesFileOf)
	if responsesFile == nil {
		if status.State == model.BatchCompleted {
			return nil, fmt.Errorf("%w: Google batch %q completed without batch output", model.ErrInvalidResponse, options.BatchID)
		}

		return model.NewBatchResultStream(noBatchItems, nil), nil
	}

	body, err := b.downloadResponses(ctx, *responsesFile, options)
	if err != nil {
		return nil, err
	}

	reader := &batchResultReader{
		batch: b,
		lines: jsonlines.NewReader(b.provider.batchMaxLineBytes),
	}

	if err := reader.lines.Open(body); err != nil {
		return nil, err
	}

	return model.NewBatchResultStream(reader.items(), reader.lines.Close), nil
}

// batchOutputField returns the output field that selectField finds in the
// operation metadata, or else in the operation response.
func batchOutputField[T any](operation *batchOperation, selectField func(*batchOutput) *T) *T {
	if operation.Metadata != nil && operation.Metadata.Output != nil {
		if value := selectField(operation.Metadata.Output); value != nil {
			return value
		}
	}

	if operation.Response != nil {
		return selectField(operation.Response)
	}

	return nil
}

func inlinedResponsesOf(found *batchOutput) *batchInlinedResponses {
	return found.InlinedResponses
}

func responsesFileOf(found *batchOutput) *string {
	return found.ResponsesFile
}

func noBatchItems(func(model.BatchItemResult, error) bool) {}

// inlineItems yields the result of every inline response.
func (b *Batch) inlineItems(responses []batchInlinedResponse) iter.Seq2[model.BatchItemResult, error] {
	return func(yield func(model.BatchItemResult, error) bool) {
		for _, response := range responses {
			item := b.convertResult(response.Metadata.Key, response.Response, response.Error)

			if !yield(item, nil) {
				return
			}
		}
	}
}

// downloadResponses opens the responses file, whose name is escaped per
// path segment.
func (b *Batch) downloadResponses(ctx context.Context, responsesFile string, options model.BatchOperationOptions) (io.ReadCloser, error) {
	downloadPath := "/download" + apiVersionSuffix + "/" + urlpath.EscapeEach(responsesFile) + ":download?alt=media"

	request, err := b.provider.originRequest(ctx, downloadPath, options.Headers)
	if err != nil {
		return nil, err
	}

	stream, err := apicall.GetStream(ctx, request)
	if err != nil {
		return nil, err
	}

	return stream.Body, nil
}

// batchResultReader reads the JSON Lines responses file of one batch. Its
// lines reader owns the body of the download.
type batchResultReader struct {
	batch *Batch
	lines *jsonlines.Reader
}

// items yields the result of every line.
func (r *batchResultReader) items() iter.Seq2[model.BatchItemResult, error] {
	return func(yield func(model.BatchItemResult, error) bool) {
		for line, err := range r.lines.Lines() {
			if err != nil {
				yield(nil, err)

				return
			}

			item, err := r.batch.convertResultLine(line)
			if !yield(item, err) || err != nil {
				return
			}
		}
	}
}

// convertResultLine decodes one line. Lines that cannot be decoded end the
// stream; request failures become failed items.
func (b *Batch) convertResultLine(line []byte) (model.BatchItemResult, error) {
	var decoded batchResultLine
	if err := json.Unmarshal(line, &decoded); err != nil {
		return nil, fmt.Errorf("%w: decode batch result line: %w", model.ErrInvalidResponse, err)
	}

	if decoded.Key == nil {
		return nil, fmt.Errorf("%w: batch result line has no key", model.ErrInvalidResponse)
	}

	return b.convertResult(*decoded.Key, decoded.Response, decoded.Error), nil
}

// convertResult converts the response or error of one request.
func (b *Batch) convertResult(key string, response json.RawMessage, rpcErr *rpcStatus) model.BatchItemResult {
	if rpcErr != nil {
		status := model.BatchItemFailed
		if (rpcErr.Status != nil && *rpcErr.Status == "CANCELLED") || rpcErr.Code == "1" {
			status = model.BatchItemCancelled
		}

		return model.TextBatchItemResult{
			ID:     key,
			Status: status,
			Error:  rpcErr.batchError("Google batch request failed."),
		}
	}

	if !wirejson.HasValue(response) {
		return failedTextItem(key, &model.BatchError{
			Message: "Google returned a batch result without a response or error.",
			Code:    "invalid_batch_result",
		})
	}

	if blocked, ok := blockedBatchItem(key, response); ok {
		return blocked
	}

	var decoded generateContentResponse
	if err := json.Unmarshal(response, &decoded); err != nil {
		return invalidBatchResponse(key)
	}

	result, err := b.generateResult(decoded)
	if err != nil {
		return invalidBatchResponse(key)
	}

	if image, ok := imageBatchItem(key, result); ok {
		return image
	}

	if contentType, unsupported := unsupportedBatchContent(result.Content); unsupported {
		return failedTextItem(key, &model.BatchError{
			Message: fmt.Sprintf(
				"Google returned a %q content block, but that content is not supported in AI SDK text batches.",
				contentType,
			),
			Code: "unsupported_content",
		})
	}

	return model.TextBatchItemResult{
		ID:     key,
		Status: model.BatchItemSucceeded,
		Result: result,
	}
}

// blockedBatchItem reports a response without candidates as failed,
// blocked when Google gave a block reason.
func blockedBatchItem(key string, response json.RawMessage) (model.BatchItemResult, bool) {
	var preview batchResponsePreview
	if err := json.Unmarshal(response, &preview); err != nil || len(preview.Candidates) > 0 {
		return nil, false
	}

	var blockReason *string
	if preview.PromptFeedback != nil {
		blockReason = preview.PromptFeedback.BlockReason
	}

	batchErr := &model.BatchError{
		Message: "Google returned a batch response without any candidates.",
		Code:    "invalid_response",
	}

	if blockReason != nil {
		batchErr = &model.BatchError{
			Message: fmt.Sprintf("Google blocked the batch request (%s).", *blockReason),
			Type:    *blockReason,
			Code:    "prompt_blocked",
		}
	}

	item := model.TextBatchItemResult{
		ID:     key,
		Status: model.BatchItemFailed,
		Error:  batchErr,
	}

	if preview.PromptFeedback != nil {
		item.ProviderMetadata = model.ProviderMetadata{
			Namespace: BatchItemMetadata{
				PromptFeedback: BatchPromptFeedback{
					BlockReason: blockReason,
				},
			},
		}
	}

	return item, true
}

// generateResult converts a generateContent response like a language model
// call without tools, with metadata under "google".
func (b *Batch) generateResult(response generateContentResponse) (*model.GenerateResult, error) {
	call := preparedCall{
		toolNames: toolnames.New(nil, map[string]string{
			codeExecutionToolID: codeExecutionToolName,
		}),
		generateID: b.provider.generateID,
		metadataNamespaces: []string{
			Namespace,
		},
	}

	converted, err := convertResponse(call, response, response.UsageMetadata)
	if err != nil {
		return nil, err
	}

	return &model.GenerateResult{
		Content:          converted.content,
		FinishReason:     converted.finishReason,
		Usage:            convertUsage(response.UsageMetadata),
		ProviderMetadata: wrapMetadata(call.metadataNamespaces, converted.metadata),
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID: response.ResponseID,
			},
		},
	}, nil
}

// imageBatchItem reports a response with inline images as an image result.
func imageBatchItem(key string, result *model.GenerateResult) (model.BatchItemResult, bool) {
	images := generatedImages(result.Content)
	if len(images) == 0 {
		return nil, false
	}

	languageMetadata, _ := result.ProviderMetadata[Namespace].(ChatMetadata)

	return model.ImageBatchItemResult{
		ID:     key,
		Status: model.BatchItemSucceeded,
		Result: &model.ImageResult{
			Images: images,
			Usage:  imageUsage(result.Usage),
			ProviderMetadata: model.ProviderMetadata{
				Namespace: ImageMetadata{
					ChatMetadata: languageMetadata,
					Images:       make([]struct{}, len(images)),
				},
			},
			Response: model.ResponseInfo{
				ResponseMetadata: model.ResponseMetadata{
					Timestamp: time.Now(),
				},
			},
		},
	}, true
}

// unsupportedBatchContent returns the upstream type name of the first
// content text batches cannot carry: anything but text, reasoning,
// sources, tool calls and tool results.
func unsupportedBatchContent(contents []model.Content) (string, bool) {
	for _, item := range contents {
		switch item.(type) {
		case model.Text, model.Reasoning, model.Source, model.ToolCall, model.ToolResult:
			continue
		case model.File:
			return "file", true
		case model.ReasoningFile:
			return "reasoning-file", true
		case model.Custom:
			return "custom", true
		case model.ToolApprovalRequest:
			return "tool-approval-request", true
		default:
			return fmt.Sprintf("%T", item), true
		}
	}

	return "", false
}

func invalidBatchResponse(key string) model.BatchItemResult {
	return failedTextItem(key, &model.BatchError{
		Message: "Google returned an invalid GenerateContent batch result.",
		Code:    "invalid_response",
	})
}

func failedTextItem(key string, batchErr *model.BatchError) model.BatchItemResult {
	return model.TextBatchItemResult{
		ID:     key,
		Status: model.BatchItemFailed,
		Error:  batchErr,
	}
}
