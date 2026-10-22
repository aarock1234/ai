package ai

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// batchReferenceVersion is the version of persisted batch references.
const batchReferenceVersion = 2

// BatchReference identifies a batch. Persist it to check the batch later,
// from any process.
type BatchReference struct {
	Version  int    `json:"version"`
	ID       string `json:"id"`
	Provider string `json:"provider"`
}

// Batch is a batch with its latest status.
type Batch struct {
	BatchReference
	model.BatchStatus
}

// BatchRequest is one request of a batch: TextBatchRequest or
// ImageBatchRequest.
type BatchRequest interface {
	batchRequestID() string
}

// TextBatchRequest is one text generation request of a batch.
type TextBatchRequest struct {
	// ID correlates the request with its result. It must be nonempty and
	// unique within the batch.
	ID string

	// Model is the provider model ID.
	Model string

	// CallSettings are the generation settings of the request. MaxRetries
	// and Headers apply to whole batch operations and must be left unset.
	CallSettings
	Prompt

	// Tools are the tools the model may call. A tool name must have the same
	// definition in every request of the batch. Execute functions are never
	// called.
	Tools []Tool

	// ToolChoice controls tool selection. Nil selects automatic choice.
	ToolChoice *model.ToolChoice

	// ToolOrder lists tool names to send first; the other tools follow in
	// name order. Nil keeps declaration order.
	ToolOrder []string

	ProviderOptions model.ProviderOptions
}

// ImageBatchRequest is one image generation request of a batch.
type ImageBatchRequest struct {
	// ID correlates the request with its result. It must be nonempty and
	// unique within the batch.
	ID string

	// Model is the provider model ID.
	Model string

	// Prompt describes the image to generate or the edit to make.
	Prompt string

	// Images are input images for editing, and Mask optionally marks the
	// region to edit.
	Images []model.FileData
	Mask   model.FileData

	// N is the number of images. Zero means one.
	N int

	// Size has the form "{width}x{height}" and AspectRatio the form
	// "{width}:{height}".
	Size        string
	AspectRatio string

	Seed *int

	ProviderOptions model.ProviderOptions
}

func (r TextBatchRequest) batchRequestID() string {
	return r.ID
}

func (r ImageBatchRequest) batchRequestID() string {
	return r.ID
}

// StartBatchOptions configures StartBatch.
type StartBatchOptions struct {
	// API is the provider's batch interface, such as the result of a
	// provider's Batch method or Registry.Batch.
	API model.Batch

	Requests []BatchRequest

	// WebhookURL is notified when the batch reaches a terminal state.
	// Providers without completion webhooks return an unsupported warning.
	WebhookURL string

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// LogWarnings receives the warnings of the batch, per request. Nil
	// discards them; they are always returned on the result.
	LogWarnings LogWarningsFunc
}

// StartBatchResult is the acknowledged batch with the warnings produced
// while starting it.
type StartBatchResult struct {
	Batch

	Warnings []model.BatchWarning
}

// GetBatchStatusOptions configures GetBatchStatus.
type GetBatchStatusOptions struct {
	// API is the batch interface the batch was started on.
	API model.Batch

	// Reference is the reference returned by StartBatch.
	Reference BatchReference

	// MaxRetries is the number of retries of the status call. Nil selects 2;
	// zero disables retries.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// GetBatchResultsOptions configures GetBatchResults.
type GetBatchResultsOptions struct {
	// API is the batch interface the batch was started on.
	API model.Batch

	// Reference is the reference returned by StartBatch.
	Reference BatchReference

	// Tools are the tools given to the text requests. They validate and
	// normalize the returned tool calls; Execute functions are never
	// called.
	Tools []Tool

	// MaxRetries is the number of retries of the results call. Nil selects
	// 2; zero disables retries.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// CancelBatchOptions configures CancelBatch.
type CancelBatchOptions struct {
	// API is the batch interface the batch was started on. It must
	// implement model.BatchCanceller.
	API model.Batch

	// Reference is the reference returned by StartBatch.
	Reference BatchReference

	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// ListBatchesOptions configures ListBatches.
type ListBatchesOptions struct {
	// API is the batch interface to list. It must implement
	// model.BatchLister.
	API model.Batch

	// Limit is the page size, or zero for the provider default.
	Limit int

	// Cursor continues a previous listing.
	Cursor string

	// MaxRetries is the number of retries of the list call. Nil selects 2;
	// zero disables retries.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// ListBatchesResult is one page of batches.
type ListBatchesResult struct {
	Batches []Batch

	// NextCursor continues the listing, and is empty on the last page.
	NextCursor string

	ProviderMetadata model.ProviderMetadata
}

// StartBatch prepares the requests like GenerateText and GenerateImage
// prepare a call, and submits them as one batch. The batch API is
// experimental and may change.
//
// Cancellation and deadlines come from ctx.
func StartBatch(ctx context.Context, options StartBatchOptions) (*StartBatchResult, error) {
	if options.API == nil {
		return nil, missingBatchAPI()
	}

	if err := validateBatchRequests(options.Requests); err != nil {
		return nil, err
	}

	requests, err := prepareBatchRequests(ctx, options.API, options.Requests)
	if err != nil {
		return nil, err
	}

	started, err := options.API.StartBatch(ctx, model.BatchStartOptions{
		Requests:        requests,
		WebhookURL:      options.WebhookURL,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	})
	if err != nil {
		return nil, err
	}

	logBatchWarnings(ctx, options, requests, started.Warnings)

	return &StartBatchResult{
		BatchReference: newBatchReference(options.API, started.BatchID),
		BatchStatus:    started.BatchStatus,
		Warnings:       started.Warnings,
	}, nil
}

// GetBatchStatus returns a batch with its latest status. The batch API is
// experimental and may change.
func GetBatchStatus(ctx context.Context, options GetBatchStatusOptions) (*Batch, error) {
	if err := validateBatchReference(options.API, options.Reference); err != nil {
		return nil, err
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	status, err := retryCall(ctx, retry, func(ctx context.Context) (*model.BatchStatus, error) {
		return options.API.BatchStatus(ctx, model.BatchOperationOptions{
			BatchID:         options.Reference.ID,
			Headers:         options.Headers,
			ProviderOptions: options.ProviderOptions,
		})
	})
	if err != nil {
		return nil, err
	}

	if status == nil {
		return nil, fmt.Errorf("%w: batch interface returned no status", model.ErrInvalidResponse)
	}

	return &Batch{
		BatchReference: options.Reference,
		BatchStatus:    *status,
	}, nil
}

// GetBatchResults returns the terminal result of every request of a
// finished batch. Text results are normalized like GenerateText content and
// image results like GenerateImage images, while the items are iterated.
// The batch API is experimental and may change.
//
// Ctx governs the results call and the iteration; it must stay valid until
// the results are consumed or closed.
func GetBatchResults(ctx context.Context, options GetBatchResultsOptions) (*BatchResults, error) {
	if err := validateBatchReference(options.API, options.Reference); err != nil {
		return nil, err
	}

	if err := validateTools(options.Tools); err != nil {
		return nil, err
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	stream, err := retryCall(ctx, retry, func(ctx context.Context) (*model.BatchResultStream, error) {
		return options.API.BatchResults(ctx, model.BatchOperationOptions{
			BatchID:         options.Reference.ID,
			Headers:         options.Headers,
			ProviderOptions: options.ProviderOptions,
		})
	})
	if err != nil {
		return nil, err
	}

	return &BatchResults{
		ctx:    ctx,
		stream: stream,
		tools:  options.Tools,
	}, nil
}

// CancelBatch requests the cancellation of a batch. The batch API is
// experimental and may change.
func CancelBatch(ctx context.Context, options CancelBatchOptions) (*model.BatchCancelResult, error) {
	if err := validateBatchReference(options.API, options.Reference); err != nil {
		return nil, err
	}

	canceller, ok := options.API.(model.BatchCanceller)
	if !ok {
		return nil, fmt.Errorf("%w: the provider does not support batch cancellation", model.ErrUnsupported)
	}

	return canceller.CancelBatch(ctx, model.BatchOperationOptions{
		BatchID:         options.Reference.ID,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	})
}

// ListBatches returns one page of batches. The batch API is experimental
// and may change.
func ListBatches(ctx context.Context, options ListBatchesOptions) (*ListBatchesResult, error) {
	if options.API == nil {
		return nil, missingBatchAPI()
	}

	lister, ok := options.API.(model.BatchLister)
	if !ok {
		return nil, fmt.Errorf("%w: the provider does not support listing batches", model.ErrUnsupported)
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	page, err := retryCall(ctx, retry, func(ctx context.Context) (*model.BatchListResult, error) {
		return lister.ListBatches(ctx, model.BatchListOptions{
			Limit:           options.Limit,
			Cursor:          options.Cursor,
			Headers:         options.Headers,
			ProviderOptions: options.ProviderOptions,
		})
	})
	if err != nil {
		return nil, err
	}

	batches := make([]Batch, len(page.Batches))
	for index, listed := range page.Batches {
		batches[index] = Batch{
			BatchReference: newBatchReference(options.API, listed.BatchID),
			BatchStatus:    listed.BatchStatus,
		}
	}

	return &ListBatchesResult{
		Batches:          batches,
		NextCursor:       page.NextCursor,
		ProviderMetadata: page.ProviderMetadata,
	}, nil
}

// BatchResults streams the normalized item results of a batch. It owns the
// provider's downloads: ranging over Items releases them when iteration
// ends, including an early break, and Close releases them without
// iterating.
type BatchResults struct {
	ctx    context.Context
	stream *model.BatchResultStream
	tools  []Tool
}

// Items returns the item results in provider order: TextBatchItemResult and
// ImageBatchItemResult values. A non-nil error ends the sequence. The
// sequence can be ranged over once.
func (r *BatchResults) Items() iter.Seq2[BatchItemResult, error] {
	return func(yield func(BatchItemResult, error) bool) {
		for item, err := range r.stream.Items() {
			if err != nil {
				yield(nil, err)

				return
			}

			converted, err := convertBatchItemResult(r.ctx, item, r.tools)
			if !yield(converted, err) || err != nil {
				return
			}
		}
	}
}

// Close releases the results. It is safe to call more than once.
func (r *BatchResults) Close() error {
	return r.stream.Close()
}

func newBatchReference(batch model.Batch, id string) BatchReference {
	return BatchReference{
		Version:  batchReferenceVersion,
		ID:       id,
		Provider: batch.Provider(),
	}
}

func missingBatchAPI() error {
	return &InvalidArgumentError{
		Parameter: "api",
		Message:   "api is required; the provider must offer a batch interface",
	}
}

// validateBatchReference checks that the reference is supported and
// belongs to the batch API.
func validateBatchReference(batch model.Batch, reference BatchReference) error {
	if batch == nil {
		return missingBatchAPI()
	}

	if reference.Version != batchReferenceVersion {
		return &InvalidArgumentError{
			Parameter: "reference",
			Value:     reference,
			Message:   "batch must be a supported batch reference",
		}
	}

	if reference.Provider != batch.Provider() {
		return &InvalidArgumentError{
			Parameter: "api",
			Value:     batch.Provider(),
			Message: fmt.Sprintf(
				"provider %s is not compatible with batch provider %s",
				batch.Provider(),
				reference.Provider,
			),
		}
	}

	return nil
}

// validateBatchRequests requires at least one request and unique, nonempty
// request IDs.
func validateBatchRequests(requests []BatchRequest) error {
	if len(requests) == 0 {
		return invalidBatchRequests("requests must not be empty")
	}

	ids := make(map[string]bool, len(requests))

	for _, request := range requests {
		if request == nil {
			return invalidBatchRequests("requests must not be nil")
		}

		id := request.batchRequestID()
		if strings.TrimSpace(id) == "" {
			return invalidBatchRequests("request IDs must not be empty")
		}

		if ids[id] {
			return invalidBatchRequests(fmt.Sprintf("request IDs must be unique; duplicate ID %q", id))
		}

		ids[id] = true
	}

	return nil
}

func invalidBatchRequests(message string) error {
	return &InvalidArgumentError{
		Parameter: "requests",
		Message:   message,
	}
}

// logBatchWarnings passes the warnings of a started batch to LogWarnings,
// attributing request warnings to the request's model.
func logBatchWarnings(ctx context.Context, options StartBatchOptions, requests []model.BatchRequest, warnings []model.BatchWarning) {
	if options.LogWarnings == nil {
		return
	}

	modelIDs := make(map[string]string, len(requests))
	for _, request := range requests {
		modelIDs[request.BatchRequestID()] = request.BatchModelID()
	}

	for _, warning := range warnings {
		options.LogWarnings.log(ctx, WarningsEvent{
			Warnings: []model.Warning{
				warning.Warning,
			},
			Provider: options.API.Provider(),
			ModelID:  modelIDs[warning.RequestID],
		})
	}
}
