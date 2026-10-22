package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/urlpath"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// batchInputFileMaxBytes bounds an uploaded input file: 2 GB.
	batchInputFileMaxBytes = 2 * 1024 * 1024 * 1024

	// batchInlineCreationMaxBytes bounds a batch creation with inline
	// requests. Larger batches upload their requests as a JSON Lines file.
	batchInlineCreationMaxBytes = 20_000_000

	batchInputMediaType = "application/jsonl"
)

// batchNamePattern matches batch IDs, the names of batch operations.
var batchNamePattern = regexp.MustCompile(`^batches/([^/]+)$`)

var (
	_ model.Batch          = (*Batch)(nil)
	_ model.BatchCanceller = (*Batch)(nil)
	_ model.BatchLister    = (*Batch)(nil)
	_ model.BatchProvider  = (*Provider)(nil)
)

// batchStatePrefix starts the Batch API states, such as
// "BATCH_STATE_RUNNING", and the legacy job states.
var batchStatePrefix = regexp.MustCompile(`^(?:BATCH|JOB)_STATE_`)

// BatchMetadata is the Google metadata of a batch started from an uploaded
// input file, stored under ProviderMetadata["google"].
type BatchMetadata struct {
	InputFileID string `json:"inputFileId"`

	// InputFileExpiresAt is the expiration time Google reported, if any.
	InputFileExpiresAt string `json:"inputFileExpiresAt,omitempty"`
}

// Batch processes text and image generation requests with the Gemini Batch
// API. Every request of a batch must use the same model, which is part of
// the batch endpoint. Small batches are sent inline; larger ones upload a
// JSON Lines input file. The batch contract is experimental and may change.
type Batch struct {
	provider *Provider
}

// WithBatchResultMaxLineBytes bounds one line of a downloaded batch result
// file. The default is 64 MiB.
func WithBatchResultMaxLineBytes(maxLineBytes int) Option {
	return func(p *Provider) {
		p.batchMaxLineBytes = maxLineBytes
	}
}

// Batch returns the batch interface, an *Batch. Batches are experimental.
func (p *Provider) Batch() model.Batch {
	return &Batch{
		provider: p,
	}
}

// Provider returns the provider identifier, "google.batch" by default.
func (b *Batch) Provider() string {
	return strings.TrimSuffix(b.provider.name, defaultProviderSuffix) + ".batch"
}

// SupportedURLs returns the URLs every batch model downloads itself: Google
// Files API files and YouTube videos. Batch requests are converted before a
// model is known, so model-specific external URLs are not included.
func (b *Batch) SupportedURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{
		"*": b.provider.geminiFileURLPatterns(),
	}
}

// batchCreateRequest creates a batch.
type batchCreateRequest struct {
	Batch batchDefinition `json:"batch"`
}

type batchDefinition struct {
	DisplayName   string              `json:"displayName"`
	WebhookConfig *batchWebhookConfig `json:"webhookConfig,omitempty"`
	InputConfig   batchInputConfig    `json:"inputConfig"`
}

type batchWebhookConfig struct {
	URIs []string `json:"uris"`
}

// batchInputConfig holds inline requests or the name of an uploaded input
// file.
type batchInputConfig struct {
	Requests *batchInlineRequests `json:"requests,omitempty"`
	FileName string               `json:"fileName,omitempty"`
}

type batchInlineRequests struct {
	Requests []batchInlineRequest `json:"requests"`
}

type batchInlineRequest struct {
	Request  json.RawMessage      `json:"request"`
	Metadata batchRequestMetadata `json:"metadata"`
}

type batchRequestMetadata struct {
	Key string `json:"key"`
}

// batchInputLine is one request line of an input file.
type batchInputLine struct {
	Key     string          `json:"key"`
	Request json.RawMessage `json:"request"`
}

// batchUploadResponse is the finalized upload of an input file.
type batchUploadResponse struct {
	File *batchUploadedFile `json:"file"`
}

type batchUploadedFile struct {
	Name           string  `json:"name"`
	ExpirationTime *string `json:"expirationTime"`
}

// StartBatch converts every request like a generateContent call and creates
// the batch, inline while the creation stays below 20 MB and from an
// uploaded JSON Lines file otherwise.
func (b *Batch) StartBatch(ctx context.Context, options model.BatchStartOptions) (*model.BatchStartResult, error) {
	modelID, err := batchModelID(options.Requests)
	if err != nil {
		return nil, err
	}

	prepared, warnings, err := b.prepareRequests(ctx, options.Requests)
	if err != nil {
		return nil, err
	}

	definition := batchDefinition{
		DisplayName: "ai-sdk-batch-" + b.provider.generateID(),
	}

	if options.WebhookURL != "" {
		definition.WebhookConfig = &batchWebhookConfig{
			URIs: []string{
				options.WebhookURL,
			},
		}
	}

	inline, fits, err := inlineBatchRequests(definition, prepared)
	if err != nil {
		return nil, err
	}

	var result *model.BatchStartResult
	if fits {
		result, err = b.startInline(ctx, modelID, definition, inline, options.Headers)
	} else {
		result, err = b.startFromFile(ctx, modelID, definition, prepared, options.Headers)
	}

	if err != nil {
		return nil, err
	}

	result.Warnings = warnings

	return result, nil
}

// startInline creates a batch whose requests are part of the creation.
func (b *Batch) startInline(
	ctx context.Context,
	modelID string,
	definition batchDefinition,
	inline *batchInlineRequests,
	headers http.Header,
) (*model.BatchStartResult, error) {
	definition.InputConfig.Requests = inline

	return b.createBatch(ctx, modelID, definition, headers)
}

// startFromFile uploads the requests as an input file and creates a batch
// that reads them, reporting the file in the provider metadata.
func (b *Batch) startFromFile(
	ctx context.Context,
	modelID string,
	definition batchDefinition,
	prepared []preparedBatchRequest,
	headers http.Header,
) (*model.BatchStartResult, error) {
	file, err := b.uploadInputFile(ctx, definition.DisplayName, prepared, headers)
	if err != nil {
		return nil, err
	}

	definition.InputConfig.FileName = file.Name

	metadata := BatchMetadata{
		InputFileID: file.Name,
	}

	if file.ExpirationTime != nil {
		metadata.InputFileExpiresAt = *file.ExpirationTime
	}

	result, err := b.createBatch(ctx, modelID, definition, headers)
	if err != nil {
		return nil, err
	}

	result.ProviderMetadata = model.ProviderMetadata{
		Namespace: metadata,
	}

	return result, nil
}

// batchModelID returns the model of the batch, which every request shares,
// and rejects request types the Batch API does not support.
func batchModelID(requests []model.BatchRequest) (string, error) {
	for _, request := range requests {
		switch request.(type) {
		case model.TextBatchRequest, model.ImageBatchRequest:
		default:
			return "", fmt.Errorf("%w: the Google Batch API does not support batch requests of type %T", model.ErrUnsupported, request)
		}
	}

	if len(requests) == 0 {
		return "", fmt.Errorf("%w: requests: Google batches require at least one request", model.ErrInvalidArgument)
	}

	modelID := requests[0].BatchModelID()

	for _, request := range requests {
		if request.BatchModelID() != modelID {
			return "", fmt.Errorf(
				"%w: requests: Google batches require every request to use the same model because the model is part of the batch endpoint",
				model.ErrInvalidArgument,
			)
		}
	}

	return modelID, nil
}

// preparedBatchRequest is one converted request.
type preparedBatchRequest struct {
	key     string
	request json.RawMessage
}

func (b *Batch) prepareRequests(ctx context.Context, requests []model.BatchRequest) ([]preparedBatchRequest, []model.BatchWarning, error) {
	prepared := make([]preparedBatchRequest, 0, len(requests))

	var warnings []model.BatchWarning

	for _, request := range requests {
		body, requestWarnings, err := b.prepareRequest(ctx, request)
		if err != nil {
			return nil, nil, err
		}

		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, nil, fmt.Errorf("encode batch request %q: %w", request.BatchRequestID(), err)
		}

		prepared = append(prepared, preparedBatchRequest{
			key:     request.BatchRequestID(),
			request: encoded,
		})

		for _, warning := range requestWarnings {
			warnings = append(warnings, model.BatchWarning{
				RequestID: request.BatchRequestID(),
				Warning:   warning,
			})
		}
	}

	return prepared, warnings, nil
}

// prepareRequest converts a text request like a generateContent call, and
// an image request like a Gemini image call.
func (b *Batch) prepareRequest(ctx context.Context, request model.BatchRequest) (generateContentRequest, []model.Warning, error) {
	switch typed := request.(type) {
	case model.TextBatchRequest:
		call, err := b.provider.Chat(typed.ModelID).prepareRequest(ctx, typed.Options)
		if err != nil {
			return generateContentRequest{}, nil, err
		}

		return call.body, call.warnings, nil

	case model.ImageBatchRequest:
		return b.prepareImageRequest(ctx, typed)

	default:
		return generateContentRequest{}, nil, fmt.Errorf("%w: batch request of type %T", model.ErrUnsupported, request)
	}
}

func (b *Batch) prepareImageRequest(ctx context.Context, request model.ImageBatchRequest) (generateContentRequest, []model.Warning, error) {
	options := request.Options

	if options.Mask != nil {
		return generateContentRequest{}, nil, fmt.Errorf("%w: mask-based image editing in Google batches", model.ErrUnsupported)
	}

	if options.N > 1 {
		return generateContentRequest{}, nil, fmt.Errorf("%w: multiple images per Google batch request", model.ErrUnsupported)
	}

	var warnings []model.Warning
	if options.Size != "" {
		warnings = append(warnings, model.UnsupportedWarning("size", "This model does not support the `size` option. Use `aspectRatio` instead."))
	}

	imageModel := b.provider.Image(request.ModelID)

	callOptions, err := imageModel.languageModelCall(options)
	if err != nil {
		return generateContentRequest{}, nil, err
	}

	call, err := b.provider.Chat(request.ModelID).prepareRequest(ctx, callOptions)
	if err != nil {
		return generateContentRequest{}, nil, err
	}

	return call.body, append(warnings, call.warnings...), nil
}

// inlineBatchRequests collects the requests of an inline creation. It
// reports false when the creation would reach the inline limit.
func inlineBatchRequests(definition batchDefinition, prepared []preparedBatchRequest) (*batchInlineRequests, bool, error) {
	inline := &batchInlineRequests{
		Requests: []batchInlineRequest{},
	}

	definition.InputConfig.Requests = inline

	empty, err := json.Marshal(batchCreateRequest{
		Batch: definition,
	})
	if err != nil {
		return nil, false, fmt.Errorf("encode batch creation: %w", err)
	}

	size := len(empty)

	for index, request := range prepared {
		entry := batchInlineRequest{
			Request: request.request,
			Metadata: batchRequestMetadata{
				Key: request.key,
			},
		}

		encoded, err := json.Marshal(entry)
		if err != nil {
			return nil, false, fmt.Errorf("encode batch request %q: %w", request.key, err)
		}

		size += len(encoded)
		if index > 0 {
			size++
		}

		if size >= batchInlineCreationMaxBytes {
			return nil, false, nil
		}

		inline.Requests = append(inline.Requests, entry)
	}

	return inline, true, nil
}

// uploadInputFile uploads the requests as a JSON Lines file with the
// resumable upload protocol.
func (b *Batch) uploadInputFile(ctx context.Context, displayName string, prepared []preparedBatchRequest, headers http.Header) (*batchUploadedFile, error) {
	input, err := batchInputFile(prepared)
	if err != nil {
		return nil, err
	}

	if len(input) > batchInputFileMaxBytes {
		return nil, fmt.Errorf("%w: requests: Google batch input files must not exceed 2 GB", model.ErrInvalidArgument)
	}

	inputName := displayName + "-input"

	uploaded, err := uploadResumable[batchUploadResponse](ctx, b.provider, resumableUpload{
		DisplayName: &inputName,
		MediaType:   batchInputMediaType,
		Content:     input,
		ContentType: batchInputMediaType,
		Headers:     headers,
	})
	if err != nil {
		return nil, err
	}

	if uploaded.File == nil || uploaded.File.Name == "" {
		return nil, fmt.Errorf("%w: batch input file upload response has no file name", model.ErrInvalidResponse)
	}

	return uploaded.File, nil
}

func batchInputFile(prepared []preparedBatchRequest) ([]byte, error) {
	var input []byte

	for _, request := range prepared {
		line, err := json.Marshal(batchInputLine{
			Key:     request.key,
			Request: request.request,
		})
		if err != nil {
			return nil, fmt.Errorf("encode batch request %q: %w", request.key, err)
		}

		input = append(input, line...)
		input = append(input, '\n')
	}

	return input, nil
}

// createBatch creates the batch on the endpoint of its model.
func (b *Batch) createBatch(ctx context.Context, modelID string, definition batchDefinition, headers http.Header) (*model.BatchStartResult, error) {
	request, err := b.provider.request(ctx, modelPath(modelID)+":batchGenerateContent", headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, batchCreateRequest{
		Batch: definition,
	})
	if err != nil {
		return nil, err
	}

	operation, err := decodeBatchOperation(response.Body)
	if err != nil {
		return nil, err
	}

	return &model.BatchStartResult{
		BatchID:     operation.Name,
		BatchStatus: operation.status(),
	}, nil
}

// BatchStatus returns the latest status of a batch.
func (b *Batch) BatchStatus(ctx context.Context, options model.BatchOperationOptions) (*model.BatchStatus, error) {
	operation, err := b.retrieveBatch(ctx, options)
	if err != nil {
		return nil, err
	}

	status := operation.status()

	return &status, nil
}

// CancelBatch requests the cancellation of a batch.
func (b *Batch) CancelBatch(ctx context.Context, options model.BatchOperationOptions) (*model.BatchCancelResult, error) {
	path, err := batchPath(options.BatchID)
	if err != nil {
		return nil, err
	}

	request, err := b.provider.request(ctx, path+":cancel", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, emptyObject{})
	if err != nil {
		return nil, err
	}

	var cancelled emptyObject
	if err := apicall.DecodeJSON(response.Body, &cancelled); err != nil {
		return nil, err
	}

	return &model.BatchCancelResult{}, nil
}

// batchListResponse is one page of batches.
type batchListResponse struct {
	Operations    []batchOperation `json:"operations"`
	NextPageToken *string          `json:"nextPageToken"`
}

// ListBatches returns one page of batches.
func (b *Batch) ListBatches(ctx context.Context, options model.BatchListOptions) (*model.BatchListResult, error) {
	query := url.Values{}

	if options.Limit > 0 {
		query.Set("pageSize", strconv.Itoa(options.Limit))
	}

	if options.Cursor != "" {
		query.Set("pageToken", options.Cursor)
	}

	path := "batches"
	if len(query) > 0 {
		path += "?" + query.Encode()
	}

	request, err := b.provider.request(ctx, path, options.Headers)
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
		Batches: make([]model.BatchListItem, len(page.Operations)),
	}

	for index, operation := range page.Operations {
		if err := operation.validate(); err != nil {
			return nil, err
		}

		result.Batches[index] = model.BatchListItem{
			BatchID:     operation.Name,
			BatchStatus: operation.status(),
		}
	}

	if page.NextPageToken != nil {
		result.NextCursor = *page.NextPageToken
	}

	return result, nil
}

func (b *Batch) retrieveBatch(ctx context.Context, options model.BatchOperationOptions) (*batchOperation, error) {
	path, err := batchPath(options.BatchID)
	if err != nil {
		return nil, err
	}

	request, err := b.provider.request(ctx, path, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, err
	}

	return decodeBatchOperation(response.Body)
}

// batchPath validates a batch ID, such as "batches/abc-123", and escapes
// its ID segment.
func batchPath(batchID string) (string, error) {
	match := batchNamePattern.FindStringSubmatch(batchID)
	if match == nil {
		return "", fmt.Errorf(
			"%w: batchId: Google batch IDs have the form batches/{id}, got %q",
			model.ErrInvalidArgument,
			batchID,
		)
	}

	return "batches/" + urlpath.Escape(match[1]), nil
}

// batchOperation is a batch long-running operation.
type batchOperation struct {
	Name     string                  `json:"name"`
	Metadata *batchOperationMetadata `json:"metadata"`
	Done     *bool                   `json:"done"`
	Error    *rpcStatus              `json:"error"`
	Response *batchOutput            `json:"response"`
}

type batchOperationMetadata struct {
	State      *string      `json:"state"`
	CreateTime *string      `json:"createTime"`
	BatchStats *batchStats  `json:"batchStats"`
	Output     *batchOutput `json:"output"`
}

type batchStats struct {
	RequestCount           *batchCount `json:"requestCount"`
	SuccessfulRequestCount *batchCount `json:"successfulRequestCount"`
	FailedRequestCount     *batchCount `json:"failedRequestCount"`
	PendingRequestCount    *batchCount `json:"pendingRequestCount"`
}

// batchOutput is the output of a finished batch: inline responses or the
// name of a responses file.
type batchOutput struct {
	ResponsesFile    *string                `json:"responsesFile"`
	InlinedResponses *batchInlinedResponses `json:"inlinedResponses"`
}

type batchInlinedResponses struct {
	InlinedResponses []batchInlinedResponse `json:"inlinedResponses"`
}

type batchInlinedResponse struct {
	Metadata batchRequestMetadata `json:"metadata"`
	Response json.RawMessage      `json:"response"`
	Error    *rpcStatus           `json:"error"`
}

// rpcStatus is a google.rpc.Status error.
type rpcStatus struct {
	Code    rpcCode `json:"code"`
	Message *string `json:"message"`
	Status  *string `json:"status"`
}

// rpcCode is a status code sent as a number or a string.
type rpcCode string

// UnmarshalJSON accepts a JSON number or string.
func (c *rpcCode) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*c = rpcCode(text)

		return nil
	}

	var number json.Number
	if err := json.Unmarshal(data, &number); err != nil {
		return fmt.Errorf("rpc status code must be a number or a string: %w", err)
	}

	*c = rpcCode(number.String())

	return nil
}

// batchCount is a request count sent as a number or a decimal string. Value
// is nil for counts that are not safe nonnegative integers.
type batchCount struct {
	value *int
}

// maxSafeInteger is the largest integer a JSON number represents exactly.
const maxSafeInteger = 1<<53 - 1

// UnmarshalJSON accepts a JSON number or string.
func (c *batchCount) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		c.value = parseBatchCount(text, true)

		return nil
	}

	var number json.Number
	if err := json.Unmarshal(data, &number); err != nil {
		return fmt.Errorf("batch request count must be a number or a string: %w", err)
	}

	c.value = parseBatchCount(number.String(), false)

	return nil
}

// parseBatchCount parses a count. Strings must hold only digits.
func parseBatchCount(text string, digitsOnly bool) *int {
	if digitsOnly && strings.TrimLeft(text, "0123456789") != "" {
		return nil
	}

	count, err := strconv.ParseInt(text, 10, 64)
	if err != nil || count < 0 || count > maxSafeInteger {
		return nil
	}

	return new(int(count))
}

// countOr returns the count, or fallback when it was not reported.
func countOr(count *batchCount, fallback int) *int {
	if count == nil {
		return new(fallback)
	}

	return count.value
}

func decodeBatchOperation(body []byte) (*batchOperation, error) {
	var operation batchOperation
	if err := apicall.DecodeJSON(body, &operation); err != nil {
		return nil, err
	}

	if err := operation.validate(); err != nil {
		return nil, err
	}

	return &operation, nil
}

func (o batchOperation) validate() error {
	if o.Name == "" {
		return fmt.Errorf("%w: batch operation has no name", model.ErrInvalidResponse)
	}

	return nil
}

// status converts the operation into a normalized status.
func (o batchOperation) status() model.BatchStatus {
	status := model.BatchStatus{}

	var state *string

	if metadata := o.Metadata; metadata != nil {
		state = metadata.State
		status.RequestCounts = metadata.BatchStats.requestCounts()

		if metadata.CreateTime != nil {
			status.CreatedAt, _ = time.Parse(time.RFC3339Nano, *metadata.CreateTime)
		}
	}

	if state != nil {
		status.RawStatus = *state
	}

	if o.Error != nil {
		status.Error = o.Error.batchError("Google batch failed.")
	}

	status.State = batchState(state, o.Done != nil && *o.Done, status.Error != nil)

	return status
}

// batchState maps the operation state. Unknown states are pending, so that
// callers do not fetch incomplete results.
func batchState(rawStatus *string, done bool, hasError bool) model.BatchState {
	if hasError {
		return model.BatchFailed
	}

	if rawStatus == nil {
		if done {
			return model.BatchCompleted
		}

		return model.BatchPending
	}

	switch batchStatePrefix.ReplaceAllString(*rawStatus, "") {
	case "SUCCEEDED":
		return model.BatchCompleted
	case "FAILED", "CANCELLED", "EXPIRED":
		return model.BatchFailed
	default:
		return model.BatchPending
	}
}

func (s *batchStats) requestCounts() *model.BatchRequestCounts {
	if s == nil {
		return nil
	}

	var total *int
	if s.RequestCount != nil {
		total = s.RequestCount.value
	}

	return model.NormalizeBatchRequestCounts(
		total,
		countOr(s.PendingRequestCount, 0),
		countOr(s.SuccessfulRequestCount, 0),
		countOr(s.FailedRequestCount, 0),
	)
}

// batchError converts the status, with a fallback message.
func (s *rpcStatus) batchError(fallbackMessage string) *model.BatchError {
	batchErr := &model.BatchError{
		Message: fallbackMessage,
		Code:    string(s.Code),
	}

	if s.Message != nil {
		batchErr.Message = *s.Message
	}

	if s.Status != nil {
		batchErr.Type = *s.Status
	}

	return batchErr
}
