package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/urlpath"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// batchEndpoint is the API endpoint every batch request targets.
	batchEndpoint = "/v1/responses"

	batchInputFilename = "batch.jsonl"

	// batchDefaultInputFileExpiresAfter is the default lifetime of the
	// uploaded input file, in seconds.
	batchDefaultInputFileExpiresAfter = 48 * 60 * 60

	batchMinInputFileExpiresAfter = 60 * 60
	batchMaxInputFileExpiresAfter = 30 * 24 * 60 * 60
)

// batchConvertibleProviderTools are the provider tools whose output batch
// results can convert.
var batchConvertibleProviderTools = []string{
	codeInterpreterToolID,
	customToolID,
	fileSearchToolID,
	webSearchToolID,
	webSearchPreviewToolID,
}

// BatchOptions are the batch settings read from the "openai" provider
// options namespace.
type BatchOptions struct {
	// InputFileExpiresAfter is the lifetime of the uploaded input file in
	// seconds, from 3600 (1 hour) to 2592000 (30 days). The default is 48
	// hours.
	InputFileExpiresAfter *int `json:"inputFileExpiresAfter,omitempty"`
}

// BatchMetadata is the "openai" provider metadata of a started batch.
type BatchMetadata struct {
	InputFileID string `json:"inputFileId"`

	// InputFileExpiresAt is zero when OpenAI did not report it.
	InputFileExpiresAt time.Time `json:"inputFileExpiresAt,omitzero"`
}

// Batch processes text requests with the OpenAI Batch API through the
// Responses endpoint. Every request of a batch must use the same model.
// The batch contract is experimental and may change.
type Batch struct {
	config config
}

// Provider returns the provider identifier, such as "openai.batch".
func (b *Batch) Provider() string {
	return b.config.modelProvider("batch")
}

// SupportedURLs reports the URLs the Responses API downloads itself.
func (b *Batch) SupportedURLs() map[string][]*regexp.Regexp {
	return newResponsesModel("", b.config).SupportedURLs()
}

// StartBatch uploads the requests as a JSON Lines file and creates the
// batch.
func (b *Batch) StartBatch(ctx context.Context, options model.BatchStartOptions) (*model.BatchStartResult, error) {
	requests, err := textBatchRequests(options.Requests)
	if err != nil {
		return nil, err
	}

	expiresAfter, err := b.inputFileExpiresAfter(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	input, warnings, err := b.inputFile(requests)
	if err != nil {
		return nil, err
	}

	if options.WebhookURL != "" {
		warnings = append([]model.BatchWarning{
			{
				Warning: model.UnsupportedWarning(
					"webhookUrl",
					"The OpenAI Batch API does not support per-batch webhook URLs.",
				),
			},
		}, warnings...)
	}

	file, err := b.uploadInputFile(ctx, input, expiresAfter, options)
	if err != nil {
		return nil, err
	}

	batch, err := b.createBatch(ctx, file.ID, options)
	if err != nil {
		return nil, err
	}

	status := batch.status()
	status.ProviderMetadata = model.ProviderMetadata{
		Namespace: BatchMetadata{
			InputFileID:        file.ID,
			InputFileExpiresAt: unixTime(file.ExpiresAt),
		},
	}

	return &model.BatchStartResult{
		BatchID:     batch.ID,
		BatchStatus: status,
		Warnings:    warnings,
	}, nil
}

// textBatchRequests accepts only text requests, all for the same model.
func textBatchRequests(requests []model.BatchRequest) ([]model.TextBatchRequest, error) {
	textRequests := make([]model.TextBatchRequest, 0, len(requests))

	for _, request := range requests {
		textRequest, ok := request.(model.TextBatchRequest)
		if !ok {
			return nil, fmt.Errorf("%w: the OpenAI Batch API does not support batch requests of type %T", model.ErrUnsupported, request)
		}

		if len(textRequests) > 0 && textRequest.ModelID != textRequests[0].ModelID {
			return nil, fmt.Errorf(
				"%w: the OpenAI Batch API requires all requests in a batch to use the same model; found %q and %q",
				model.ErrInvalidArgument,
				textRequests[0].ModelID,
				textRequest.ModelID,
			)
		}

		textRequests = append(textRequests, textRequest)
	}

	return textRequests, nil
}

// inputFileExpiresAfter reads the input file lifetime from the "openai"
// provider options.
func (b *Batch) inputFileExpiresAfter(providerOptions model.ProviderOptions) (int, error) {
	options, _, err := model.ProviderValue[BatchOptions](providerOptions, Namespace)
	if err != nil {
		return 0, err
	}

	if options.InputFileExpiresAfter == nil {
		return batchDefaultInputFileExpiresAfter, nil
	}

	expiresAfter := *options.InputFileExpiresAfter
	if expiresAfter < batchMinInputFileExpiresAfter || expiresAfter > batchMaxInputFileExpiresAfter {
		return 0, fmt.Errorf(
			"%w: providerOptions.openai.inputFileExpiresAfter must be between %d and %d",
			model.ErrInvalidArgument,
			batchMinInputFileExpiresAfter,
			batchMaxInputFileExpiresAfter,
		)
	}

	return expiresAfter, nil
}

// batchInputLine is one request line of the input file.
type batchInputLine struct {
	CustomID string           `json:"custom_id"`
	Method   string           `json:"method"`
	URL      string           `json:"url"`
	Body     responsesRequest `json:"body"`
}

// inputFile prepares every request like a Responses API call and encodes
// them as JSON Lines.
func (b *Batch) inputFile(requests []model.TextBatchRequest) ([]byte, []model.BatchWarning, error) {
	var input []byte
	var warnings []model.BatchWarning

	for _, request := range requests {
		modelConfig := b.config.forModel(request.ModelID, nil)
		responsesModel := newResponsesModel(request.ModelID, modelConfig)

		call, err := responsesModel.prepare(request.Options)
		if err != nil {
			return nil, nil, err
		}

		line, err := json.Marshal(batchInputLine{
			CustomID: request.ID,
			Method:   "POST",
			URL:      batchEndpoint,
			Body:     call.body,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("encode batch request %q: %w", request.ID, err)
		}

		input = append(input, line...)
		input = append(input, '\n')

		for _, warning := range call.warnings {
			warnings = append(warnings, model.BatchWarning{
				RequestID: request.ID,
				Warning:   warning,
			})
		}

		warnings = append(warnings, unconvertibleToolWarnings(request)...)
	}

	return input, warnings, nil
}

// unconvertibleToolWarnings warns about provider tools whose output batch
// results cannot convert.
func unconvertibleToolWarnings(request model.TextBatchRequest) []model.BatchWarning {
	var warnings []model.BatchWarning

	for _, tool := range request.Options.Tools {
		providerTool, ok := tool.(model.ProviderTool)
		if !ok || slices.Contains(batchConvertibleProviderTools, providerTool.ID) {
			continue
		}

		warnings = append(warnings, model.BatchWarning{
			RequestID: request.ID,
			Warning: model.UnsupportedWarning(
				fmt.Sprintf("batch result conversion for tool %q", providerTool.Name),
				"OpenAI may return output for this tool that AI SDK text batches cannot currently convert.",
			),
		})
	}

	return warnings
}

func (b *Batch) uploadInputFile(ctx context.Context, input []byte, expiresAfter int, options model.BatchStartOptions) (fileResponse, error) {
	request, err := b.config.request(ctx, "/files", options.Headers)
	if err != nil {
		return fileResponse{}, err
	}

	response, err := apicall.PostForm(ctx, request, []apicall.FormField{
		{
			Name:      "file",
			Filename:  batchInputFilename,
			MediaType: "application/jsonl",
			Data:      input,
		},
		{
			Name:  "purpose",
			Value: "batch",
		},
		{
			Name:  "expires_after[anchor]",
			Value: "created_at",
		},
		{
			Name:  "expires_after[seconds]",
			Value: strconv.Itoa(expiresAfter),
		},
	})
	if err != nil {
		return fileResponse{}, err
	}

	return decodeFileResponse(response.Body)
}

// batchCreateRequest is the body of a batch creation.
type batchCreateRequest struct {
	InputFileID      string `json:"input_file_id"`
	Endpoint         string `json:"endpoint"`
	CompletionWindow string `json:"completion_window"`
}

func (b *Batch) createBatch(ctx context.Context, inputFileID string, options model.BatchStartOptions) (*batchResponse, error) {
	request, err := b.config.request(ctx, "/batches", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, batchCreateRequest{
		InputFileID:      inputFileID,
		Endpoint:         batchEndpoint,
		CompletionWindow: "24h",
	})
	if err != nil {
		return nil, err
	}

	return decodeBatchResponse(response.Body)
}

// BatchStatus returns the latest status of a batch.
func (b *Batch) BatchStatus(ctx context.Context, options model.BatchOperationOptions) (*model.BatchStatus, error) {
	batch, err := b.retrieveBatch(ctx, options)
	if err != nil {
		return nil, err
	}

	status := batch.status()

	return &status, nil
}

// CancelBatch requests the cancellation of a batch.
func (b *Batch) CancelBatch(ctx context.Context, options model.BatchOperationOptions) (*model.BatchCancelResult, error) {
	request, err := b.config.request(ctx, "/batches/"+urlpath.Escape(options.BatchID)+"/cancel", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, struct{}{})
	if err != nil {
		return nil, err
	}

	if _, err := decodeBatchResponse(response.Body); err != nil {
		return nil, err
	}

	return &model.BatchCancelResult{}, nil
}

// batchListResponse is one page of batches.
type batchListResponse struct {
	Data    []batchResponse `json:"data"`
	HasMore bool            `json:"has_more"`
	LastID  *string         `json:"last_id"`
}

// ListBatches returns one page of batches, newest first.
func (b *Batch) ListBatches(ctx context.Context, options model.BatchListOptions) (*model.BatchListResult, error) {
	query := url.Values{}

	if options.Limit > 0 {
		query.Set("limit", strconv.Itoa(options.Limit))
	}

	if options.Cursor != "" {
		query.Set("after", options.Cursor)
	}

	path := "/batches"
	if len(query) > 0 {
		path += "?" + query.Encode()
	}

	request, err := b.config.request(ctx, path, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, err
	}

	var page batchListResponse
	if err := apicall.DecodeJSON(response.Body, &page); err != nil {
		return nil, err
	}

	result := &model.BatchListResult{
		Batches: make([]model.BatchListItem, len(page.Data)),
	}

	for index, batch := range page.Data {
		result.Batches[index] = model.BatchListItem{
			BatchID:     batch.ID,
			BatchStatus: batch.status(),
		}
	}

	if page.HasMore && page.LastID != nil {
		result.NextCursor = *page.LastID
	}

	return result, nil
}

func (b *Batch) retrieveBatch(ctx context.Context, options model.BatchOperationOptions) (*batchResponse, error) {
	request, err := b.config.request(ctx, "/batches/"+urlpath.Escape(options.BatchID), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, err
	}

	return decodeBatchResponse(response.Body)
}

// batchResponse is a batch object.
type batchResponse struct {
	ID            string              `json:"id"`
	Status        string              `json:"status"`
	OutputFileID  *string             `json:"output_file_id"`
	ErrorFileID   *string             `json:"error_file_id"`
	CreatedAt     *int64              `json:"created_at"`
	ExpiresAt     *int64              `json:"expires_at"`
	RequestCounts *batchRequestCounts `json:"request_counts"`
	Errors        *batchErrors        `json:"errors"`
}

type batchRequestCounts struct {
	Total     *int `json:"total"`
	Completed *int `json:"completed"`
	Failed    *int `json:"failed"`
}

type batchErrors struct {
	Data []batchErrorEntry `json:"data"`
}

type batchErrorEntry struct {
	Code    *string `json:"code"`
	Message *string `json:"message"`
}

func decodeBatchResponse(body []byte) (*batchResponse, error) {
	var batch batchResponse
	if err := apicall.DecodeJSON(body, &batch); err != nil {
		return nil, err
	}

	if batch.ID == "" || batch.Status == "" {
		return nil, fmt.Errorf("%w: batch response has no id or status", model.ErrInvalidResponse)
	}

	return &batch, nil
}

// status converts the batch into a normalized status.
func (r batchResponse) status() model.BatchStatus {
	status := model.BatchStatus{
		State:         batchState(r.Status),
		RawStatus:     r.Status,
		RequestCounts: r.requestCounts(),
		CreatedAt:     unixTime(r.CreatedAt),
		ExpiresAt:     unixTime(r.ExpiresAt),
	}

	if r.Errors != nil && len(r.Errors.Data) > 0 {
		first := r.Errors.Data[0]

		status.Error = &model.BatchError{
			Message: valueOr(first.Message, "OpenAI batch failed."),
			Code:    valueOr(first.Code, ""),
		}
	}

	return status
}

// batchState maps the API status. Unknown states are treated as pending,
// so that callers do not fetch incomplete results.
func batchState(rawStatus string) model.BatchState {
	switch rawStatus {
	case "completed":
		return model.BatchCompleted
	case "failed", "expired", "cancelled":
		return model.BatchFailed
	default:
		return model.BatchPending
	}
}

func (r batchResponse) requestCounts() *model.BatchRequestCounts {
	counts := r.RequestCounts
	if counts == nil {
		return nil
	}

	var pending *int
	if counts.Total != nil && counts.Completed != nil && counts.Failed != nil {
		pending = new(*counts.Total - *counts.Completed - *counts.Failed)
	}

	return model.NormalizeBatchRequestCounts(counts.Total, pending, counts.Completed, counts.Failed)
}
