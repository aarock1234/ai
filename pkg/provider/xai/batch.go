package xai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"regexp"
	"strconv"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// batchName names every batch the SDK creates.
	batchName = "ai-sdk-text-batch"

	batchInputFilename = "batch.jsonl"

	// Endpoints that batch request lines target.
	batchResponsesEndpoint       = "/v1/responses"
	batchImageGenerationEndpoint = "/v1/images/generations"
	batchImageEditEndpoint       = "/v1/images/edits"

	batchMinInputFileExpiresAfter = 60 * 60
	batchMaxInputFileExpiresAfter = 30 * 24 * 60 * 60
)

var (
	_ model.Batch          = (*Batch)(nil)
	_ model.BatchCanceller = (*Batch)(nil)
	_ model.BatchLister    = (*Batch)(nil)
	_ model.BatchProvider  = (*Provider)(nil)
)

// BatchOptions are the batch settings read from the "xai" provider options
// namespace.
type BatchOptions struct {
	// InputFileExpiresAfter is the lifetime of the uploaded input file in
	// seconds, from 3600 (1 hour) to 2592000 (30 days). Without it the file
	// does not expire.
	InputFileExpiresAfter *int `json:"inputFileExpiresAfter,omitempty"`
}

// BatchMetadata is the "xai" provider metadata of a started batch.
type BatchMetadata struct {
	InputFileID string `json:"inputFileId"`

	// InputFileExpiresAt is zero when xAI did not report it.
	InputFileExpiresAt time.Time `json:"inputFileExpiresAt,omitzero"`
}

// BatchTextMetadata is the "xai" provider metadata of a text batch result.
type BatchTextMetadata struct {
	CostInUSDTicks *float64 `json:"costInUsdTicks,omitempty"`
	ServiceTier    string   `json:"serviceTier,omitempty"`
}

// Batch returns the batch interface, an *Batch that processes Responses
// API and image requests with the xAI Batch API. Batches are experimental.
func (p *Provider) Batch() model.Batch {
	return &Batch{
		provider: p,
	}
}

// Batch processes text and image requests with the xAI Batch API. The
// batch contract is experimental and may change.
type Batch struct {
	provider *Provider
}

// Provider returns the provider identifier, "xai.batch" by default.
func (b *Batch) Provider() string {
	return b.provider.name + ".batch"
}

// SupportedURLs reports the URLs the Responses API reads itself.
func (b *Batch) SupportedURLs() map[string][]*regexp.Regexp {
	return supportedURLs
}

// StartBatch uploads the requests as a JSON Lines file and creates the
// batch.
func (b *Batch) StartBatch(ctx context.Context, options model.BatchStartOptions) (*model.BatchStartResult, error) {
	expiresAfter, err := inputFileExpiresAfter(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	var warnings []model.BatchWarning
	if options.WebhookURL != "" {
		warnings = append(warnings, model.BatchWarning{
			Warning: model.UnsupportedWarning(
				"webhookUrl",
				"The xAI Batch API does not support per-batch webhook URLs.",
			),
		})
	}

	input, inputWarnings, err := b.inputFile(options.Requests)
	if err != nil {
		return nil, err
	}

	warnings = append(warnings, inputWarnings...)

	file, err := b.uploadInputFile(ctx, input, expiresAfter, options)
	if err != nil {
		return nil, err
	}

	batch, err := b.createBatch(ctx, file.ID, options)
	if err != nil {
		return nil, err
	}

	status, err := batch.status()
	if err != nil {
		return nil, err
	}

	metadata := BatchMetadata{
		InputFileID: file.ID,
	}

	if file.ExpiresAt != nil {
		metadata.InputFileExpiresAt = time.Unix(*file.ExpiresAt, 0).UTC()
	}

	status.ProviderMetadata = model.ProviderMetadata{
		Namespace: metadata,
	}

	return &model.BatchStartResult{
		BatchID:     batch.BatchID,
		BatchStatus: status,
		Warnings:    warnings,
	}, nil
}

// inputFileExpiresAfter reads the optional input file lifetime.
func inputFileExpiresAfter(providerOptions model.ProviderOptions) (*int, error) {
	options, _, err := model.ProviderValue[BatchOptions](providerOptions, Namespace)
	if err != nil {
		return nil, err
	}

	expiresAfter := options.InputFileExpiresAfter
	if expiresAfter == nil {
		return nil, nil
	}

	if *expiresAfter < batchMinInputFileExpiresAfter || *expiresAfter > batchMaxInputFileExpiresAfter {
		return nil, fmt.Errorf(
			"%w: providerOptions.xai.inputFileExpiresAfter must be between %d and %d",
			model.ErrInvalidArgument,
			batchMinInputFileExpiresAfter,
			batchMaxInputFileExpiresAfter,
		)
	}

	return expiresAfter, nil
}

// batchInputLine is one request line of the input file.
type batchInputLine struct {
	CustomID string `json:"custom_id"`
	Method   string `json:"method"`
	URL      string `json:"url"`
	Body     any    `json:"body"`
}

// preparedBatchRequest is the endpoint and body of one batch request.
type preparedBatchRequest struct {
	endpoint string
	body     any
	warnings []model.Warning
}

// inputFile prepares every request like a direct call and encodes them as
// JSON Lines.
func (b *Batch) inputFile(requests []model.BatchRequest) ([]byte, []model.BatchWarning, error) {
	var input []byte
	var warnings []model.BatchWarning

	for _, request := range requests {
		prepared, err := b.prepareRequest(request)
		if err != nil {
			return nil, nil, err
		}

		line, err := json.Marshal(batchInputLine{
			CustomID: request.BatchRequestID(),
			Method:   "POST",
			URL:      prepared.endpoint,
			Body:     prepared.body,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("encode batch request %q: %w", request.BatchRequestID(), err)
		}

		input = append(input, line...)
		input = append(input, '\n')

		for _, warning := range prepared.warnings {
			warnings = append(warnings, model.BatchWarning{
				RequestID: request.BatchRequestID(),
				Warning:   warning,
			})
		}
	}

	return input, warnings, nil
}

func (b *Batch) prepareRequest(request model.BatchRequest) (preparedBatchRequest, error) {
	switch typed := request.(type) {
	case model.TextBatchRequest:
		prepared, err := b.provider.Responses(typed.ModelID).prepareRequest(typed.Options)
		if err != nil {
			return preparedBatchRequest{}, err
		}

		return preparedBatchRequest{
			endpoint: batchResponsesEndpoint,
			body:     prepared.body,
			warnings: prepared.warnings,
		}, nil

	case model.ImageBatchRequest:
		return b.prepareImageRequest(typed)

	default:
		return preparedBatchRequest{}, fmt.Errorf(
			"%w: the xAI Batch API does not support batch requests of type %T",
			model.ErrUnsupported,
			request,
		)
	}
}

// prepareImageRequest builds an image generation request, or an edit
// request when the request has input files.
func (b *Batch) prepareImageRequest(request model.ImageBatchRequest) (preparedBatchRequest, error) {
	imageOptions, err := readImageOptions(request.Options.ProviderOptions)
	if err != nil {
		return preparedBatchRequest{}, err
	}

	endpoint := batchImageGenerationEndpoint
	if len(request.Options.Files) > 0 {
		endpoint = batchImageEditEndpoint
	}

	return preparedBatchRequest{
		endpoint: endpoint,
		body:     b.provider.Image(request.ModelID).requestBody(request.Options, imageOptions),
		warnings: unsupportedImageSettings(request.Options),
	}, nil
}

func (b *Batch) uploadInputFile(ctx context.Context, input []byte, expiresAfter *int, options model.BatchStartOptions) (fileResponse, error) {
	request, err := b.provider.apiRequest("/files", options.Headers)
	if err != nil {
		return fileResponse{}, err
	}

	// xAI rejects uploads whose fields follow the file part.
	var fields []apicall.FormField
	if expiresAfter != nil {
		fields = append(fields, apicall.FormField{
			Name:  "expires_after",
			Value: strconv.Itoa(*expiresAfter),
		})
	}

	fields = append(fields, apicall.FormField{
		Name:      "file",
		Filename:  batchInputFilename,
		MediaType: "application/jsonl",
		Data:      input,
	})

	response, err := apicall.PostForm(ctx, request, fields)
	if err != nil {
		return fileResponse{}, err
	}

	return decodeFileResponse(response.Body)
}

// batchCreateRequest is the body of a batch creation.
type batchCreateRequest struct {
	Name        string `json:"name"`
	InputFileID string `json:"input_file_id"`
}

func (b *Batch) createBatch(ctx context.Context, inputFileID string, options model.BatchStartOptions) (*batchResponse, error) {
	request, err := b.provider.apiRequest("/batches", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, batchCreateRequest{
		Name:        batchName,
		InputFileID: inputFileID,
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

	status, err := batch.status()
	if err != nil {
		return nil, err
	}

	return &status, nil
}

// CancelBatch requests the cancellation of a batch.
func (b *Batch) CancelBatch(ctx context.Context, options model.BatchOperationOptions) (*model.BatchCancelResult, error) {
	request, err := b.provider.apiRequest("/batches/"+url.PathEscape(options.BatchID)+":cancel", options.Headers)
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
	Batches         []batchResponse `json:"batches"`
	PaginationToken *string         `json:"pagination_token"`
}

// ListBatches returns one page of batches.
func (b *Batch) ListBatches(ctx context.Context, options model.BatchListOptions) (*model.BatchListResult, error) {
	query := url.Values{}

	if options.Limit > 0 {
		query.Set("limit", strconv.Itoa(options.Limit))
	}

	if options.Cursor != "" {
		query.Set("pagination_token", options.Cursor)
	}

	path := "/batches"
	if len(query) > 0 {
		path += "?" + query.Encode()
	}

	request, err := b.provider.apiRequest(path, options.Headers)
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
		Batches: make([]model.BatchListItem, len(page.Batches)),
	}

	for index, batch := range page.Batches {
		if err := batch.validate(); err != nil {
			return nil, err
		}

		status, err := batch.status()
		if err != nil {
			return nil, err
		}

		result.Batches[index] = model.BatchListItem{
			BatchID:     batch.BatchID,
			BatchStatus: status,
		}
	}

	if page.PaginationToken != nil {
		result.NextCursor = *page.PaginationToken
	}

	return result, nil
}

func (b *Batch) retrieveBatch(ctx context.Context, options model.BatchOperationOptions) (*batchResponse, error) {
	request, err := b.provider.apiRequest("/batches/"+url.PathEscape(options.BatchID), options.Headers)
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
	BatchID            string      `json:"batch_id"`
	Name               *string     `json:"name"`
	CreateTime         *string     `json:"create_time"`
	ExpireTime         *string     `json:"expire_time"`
	CancelTime         *string     `json:"cancel_time"`
	CancelByXAIMessage *string     `json:"cancel_by_xai_message"`
	State              *batchState `json:"state"`
}

// batchState counts the requests of a batch.
type batchState struct {
	NumRequests  *int `json:"num_requests"`
	NumPending   *int `json:"num_pending"`
	NumSuccess   *int `json:"num_success"`
	NumError     *int `json:"num_error"`
	NumCancelled *int `json:"num_cancelled"`
}

func decodeBatchResponse(body []byte) (*batchResponse, error) {
	var batch batchResponse
	if err := apicall.DecodeJSON(body, &batch); err != nil {
		return nil, err
	}

	if err := batch.validate(); err != nil {
		return nil, err
	}

	return &batch, nil
}

func (r batchResponse) validate() error {
	if r.BatchID == "" {
		return fmt.Errorf("%w: batch response has no batch_id", model.ErrInvalidResponse)
	}

	return nil
}

// status converts the batch into a normalized status. Cancelled and expired
// batches have failed; batches without pending requests have completed.
func (r batchResponse) status() (model.BatchStatus, error) {
	createdAt, err := batchTime("create_time", r.CreateTime)
	if err != nil {
		return model.BatchStatus{}, err
	}

	expiresAt, err := batchTime("expire_time", r.ExpireTime)
	if err != nil {
		return model.BatchStatus{}, err
	}

	counts := r.requestCounts()

	status := model.BatchStatus{
		State:         model.BatchPending,
		RequestCounts: counts,
		CreatedAt:     createdAt,
		ExpiresAt:     expiresAt,
	}

	isCancelled := r.CancelTime != nil || r.CancelByXAIMessage != nil
	isExpired := !expiresAt.IsZero() && !expiresAt.After(time.Now())

	switch {
	case isCancelled:
		message := fmt.Sprintf("xAI batch %q was cancelled.", r.BatchID)
		if r.CancelByXAIMessage != nil {
			message = *r.CancelByXAIMessage
		}

		status.State = model.BatchFailed
		status.Error = &model.BatchError{
			Message: message,
			Code:    "batch_cancelled",
		}

	case isExpired:
		status.State = model.BatchFailed
		status.Error = &model.BatchError{
			Message: fmt.Sprintf("xAI batch %q expired.", r.BatchID),
			Code:    "batch_expired",
		}

	case counts != nil && counts.Total > 0 && counts.Pending == 0:
		status.State = model.BatchCompleted
	}

	return status, nil
}

// batchTime parses an optional RFC 3339 timestamp of a batch object.
func batchTime(name string, value *string) (time.Time, error) {
	if value == nil {
		return time.Time{}, nil
	}

	parsed, err := time.Parse(time.RFC3339Nano, *value)
	if err != nil {
		return time.Time{}, fmt.Errorf("%w: batch %s: %w", model.ErrInvalidResponse, name, err)
	}

	return parsed, nil
}

func (r batchResponse) requestCounts() *model.BatchRequestCounts {
	state := r.State
	if state == nil {
		return nil
	}

	var failed *int
	if state.NumError != nil && state.NumCancelled != nil {
		failed = new(*state.NumError + *state.NumCancelled)
	}

	return model.NormalizeBatchRequestCounts(state.NumRequests, state.NumPending, state.NumSuccess, failed)
}
