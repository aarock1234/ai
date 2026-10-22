package model

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"regexp"
	"time"
)

// Batch processes many generation requests asynchronously, usually at a
// lower price, through a provider's batch API. The batch contract is
// experimental and may change.
//
// Batches that can be cancelled implement BatchCanceller, and batches that
// can be listed implement BatchLister.
type Batch interface {
	// Provider returns the provider identifier, such as "openai.batch".
	Provider() string

	// SupportedURLs maps media type patterns to the URL patterns the
	// provider downloads itself, as LanguageModel.SupportedURLs does.
	SupportedURLs() map[string][]*regexp.Regexp

	// StartBatch submits the requests as one batch.
	StartBatch(ctx context.Context, options BatchStartOptions) (*BatchStartResult, error)

	// BatchStatus returns the latest status of a batch.
	BatchStatus(ctx context.Context, options BatchOperationOptions) (*BatchStatus, error)

	// BatchResults streams the terminal result of every request of a
	// finished batch.
	BatchResults(ctx context.Context, options BatchOperationOptions) (*BatchResultStream, error)
}

// BatchCanceller is implemented by batches that can be cancelled.
type BatchCanceller interface {
	CancelBatch(ctx context.Context, options BatchOperationOptions) (*BatchCancelResult, error)
}

// BatchLister is implemented by batches that can be listed.
type BatchLister interface {
	ListBatches(ctx context.Context, options BatchListOptions) (*BatchListResult, error)
}

// BatchProvider is implemented by providers that offer a batch API. It is
// separate from Provider because most providers have none.
type BatchProvider interface {
	Batch() Batch
}

// BatchRequest is one request of a batch: TextBatchRequest or
// ImageBatchRequest.
type BatchRequest interface {
	// BatchRequestID returns the application identifier that correlates the
	// request with its result.
	BatchRequestID() string

	// BatchModelID returns the provider model ID of the request.
	BatchModelID() string
}

// TextBatchRequest is one text generation request of a batch.
type TextBatchRequest struct {
	// ID correlates the request with its result. It is unique within the
	// batch.
	ID string

	ModelID string

	// Options are the call options of the request. Headers and
	// IncludeRawChunks do not apply to batch requests and are ignored.
	Options CallOptions
}

// ImageBatchRequest is one image generation request of a batch.
type ImageBatchRequest struct {
	// ID correlates the request with its result. It is unique within the
	// batch.
	ID string

	ModelID string

	// Options are the call options of the request. Headers do not apply to
	// batch requests and are ignored.
	Options ImageOptions
}

// BatchRequestID returns the request ID.
func (r TextBatchRequest) BatchRequestID() string {
	return r.ID
}

// BatchModelID returns the model ID.
func (r TextBatchRequest) BatchModelID() string {
	return r.ModelID
}

// BatchRequestID returns the request ID.
func (r ImageBatchRequest) BatchRequestID() string {
	return r.ID
}

// BatchModelID returns the model ID.
func (r ImageBatchRequest) BatchModelID() string {
	return r.ModelID
}

// BatchStartOptions are the inputs of StartBatch.
type BatchStartOptions struct {
	Requests []BatchRequest

	// WebhookURL is notified when the batch reaches a terminal state.
	// Providers without completion webhooks return an unsupported warning.
	WebhookURL string

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// BatchOperationOptions identify one batch for a status, results or cancel
// operation.
type BatchOperationOptions struct {
	BatchID string

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// BatchListOptions are the inputs of ListBatches.
type BatchListOptions struct {
	// Limit is the page size, or zero for the provider default.
	Limit int

	// Cursor continues a previous listing.
	Cursor string

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// BatchState is the normalized lifecycle state of a batch.
type BatchState string

// Batch states.
const (
	BatchPending   BatchState = "pending"
	BatchCompleted BatchState = "completed"
	BatchFailed    BatchState = "failed"
)

// BatchStatus is the normalized lifecycle status of a batch. Empty fields
// were not reported by the provider.
type BatchStatus struct {
	State BatchState

	// RawStatus is the provider's own status, such as "in_progress".
	RawStatus string

	// RequestCounts is nil when the provider did not report complete counts.
	RequestCounts *BatchRequestCounts

	// Error describes why the batch failed.
	Error *BatchError

	CreatedAt time.Time
	ExpiresAt time.Time

	ProviderMetadata ProviderMetadata
}

// BatchRequestCounts counts the requests of a batch by state.
type BatchRequestCounts struct {
	Total     int
	Pending   int
	Completed int
	Failed    int
}

// NormalizeBatchRequestCounts returns complete request counts, or nil when
// any count is missing or negative, or when the counts do not add up to the
// total.
func NormalizeBatchRequestCounts(total, pending, completed, failed *int) *BatchRequestCounts {
	counts := []*int{
		total,
		pending,
		completed,
		failed,
	}

	for _, count := range counts {
		if count == nil || *count < 0 {
			return nil
		}
	}

	if *pending+*completed+*failed != *total {
		return nil
	}

	return &BatchRequestCounts{
		Total:     *total,
		Pending:   *pending,
		Completed: *completed,
		Failed:    *failed,
	}
}

// BatchError describes the failure of a batch or of one of its requests.
// Empty fields were not reported by the provider.
type BatchError struct {
	Message string
	Type    string
	Code    string

	// StatusCode is the HTTP status of a failed request, or zero.
	StatusCode int
}

// Error returns the message with the code, if any.
func (e *BatchError) Error() string {
	if e.Code == "" {
		return e.Message
	}

	return fmt.Sprintf("%s (%s)", e.Message, e.Code)
}

// BatchStartResult is the outcome of StartBatch.
type BatchStartResult struct {
	BatchID string

	BatchStatus

	Warnings []BatchWarning
}

// BatchWarning is a warning about the batch or about one of its requests.
type BatchWarning struct {
	// RequestID is empty for warnings about the whole batch.
	RequestID string

	Warning Warning
}

// BatchCancelResult is the outcome of CancelBatch.
type BatchCancelResult struct {
	ProviderMetadata ProviderMetadata
}

// BatchListResult is one page of batches.
type BatchListResult struct {
	Batches []BatchListItem

	// NextCursor continues the listing, and is empty on the last page.
	NextCursor string

	ProviderMetadata ProviderMetadata
}

// BatchListItem is one listed batch.
type BatchListItem struct {
	BatchID string

	BatchStatus
}

// BatchItemStatus is the terminal state of one request of a batch.
type BatchItemStatus string

// Batch item states.
const (
	BatchItemSucceeded BatchItemStatus = "succeeded"
	BatchItemFailed    BatchItemStatus = "failed"
	BatchItemCancelled BatchItemStatus = "cancelled"
	BatchItemExpired   BatchItemStatus = "expired"
)

// BatchItemResult is the terminal result of one request:
// TextBatchItemResult or ImageBatchItemResult.
type BatchItemResult interface {
	isBatchItemResult()
}

// TextBatchItemResult is the result of a text request. Result is set when
// the request succeeded; Error is set when it failed and may be set when it
// was cancelled or expired.
type TextBatchItemResult struct {
	ID     string
	Status BatchItemStatus

	Result *GenerateResult
	Error  *BatchError

	ProviderMetadata ProviderMetadata
}

// ImageBatchItemResult is the result of an image request. Result is set
// when the request succeeded; Error is set when it failed and may be set
// when it was cancelled or expired.
type ImageBatchItemResult struct {
	ID     string
	Status BatchItemStatus

	Result *ImageResult
	Error  *BatchError

	ProviderMetadata ProviderMetadata
}

func (TextBatchItemResult) isBatchItemResult()  {}
func (ImageBatchItemResult) isBatchItemResult() {}

// BatchResultStream streams the item results of a batch. It owns the
// underlying downloads: ranging over Items releases them when iteration
// ends, including an early break, and Close releases them without
// iterating.
type BatchResultStream struct {
	items  iter.Seq2[BatchItemResult, error]
	stream singleUseStream
}

// NewBatchResultStream returns a stream over items. A non-nil error ends
// the stream. Release, which may be nil, is called exactly once when the
// stream ends or is closed.
func NewBatchResultStream(items iter.Seq2[BatchItemResult, error], release func() error) *BatchResultStream {
	return &BatchResultStream{
		items: items,
		stream: singleUseStream{
			release: release,
		},
	}
}

// Items returns the item results in provider order. The sequence can be
// ranged over once; a second iteration yields a single error wrapping
// ErrStreamConsumed.
func (s *BatchResultStream) Items() iter.Seq2[BatchItemResult, error] {
	return func(yield func(BatchItemResult, error) bool) {
		if !s.stream.claim() {
			yield(nil, ErrStreamConsumed)

			return
		}

		defer s.closeAfterIteration()

		for item, err := range s.items {
			if !yield(item, err) || err != nil {
				return
			}
		}
	}
}

// closeAfterIteration releases the stream when iteration ends. The release
// error is kept and returned by Close.
func (s *BatchResultStream) closeAfterIteration() {
	_ = s.Close()
}

// Close releases the stream. It is safe to call more than once and returns
// the release error, if any.
func (s *BatchResultStream) Close() error {
	return s.stream.close()
}
