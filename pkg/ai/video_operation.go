package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

// Default polling of the asynchronous start/status flow.
const (
	defaultVideoPollInterval = 5 * time.Second
	defaultVideoPollTimeout  = 10 * time.Minute
)

// idempotencyKeyHeader carries the key that makes a billable start safe to
// retry.
const idempotencyKeyHeader = "Idempotency-Key"

// VideoPollOptions configure the asynchronous start/status flow. With a
// webhook, Timeout limits the wait for the notification; without one, the
// settings control polling.
type VideoPollOptions struct {
	// Interval is the time between status checks. Zero selects 5 seconds.
	Interval time.Duration

	// Timeout is the longest time to wait for completion. Zero selects 10
	// minutes.
	Timeout time.Duration

	// Delay waits between status checks and for the webhook timeout, for
	// example with a durable workflow's sleep. It returns an error when ctx
	// ends first. Nil uses a timer.
	Delay func(ctx context.Context, delay time.Duration) error
}

// resolvePollOptions fills in the defaults of absent settings.
func resolvePollOptions(options *VideoPollOptions) VideoPollOptions {
	resolved := VideoPollOptions{
		Interval: defaultVideoPollInterval,
		Timeout:  defaultVideoPollTimeout,
		Delay:    sleep,
	}

	if options == nil {
		return resolved
	}

	if options.Interval > 0 {
		resolved.Interval = options.Interval
	}

	if options.Timeout > 0 {
		resolved.Timeout = options.Timeout
	}

	if options.Delay != nil {
		resolved.Delay = options.Delay
	}

	return resolved
}

// videoOperation runs one call through the asynchronous start/status flow.
type videoOperation struct {
	model    model.VideoModel
	operator model.VideoOperator
	options  model.VideoOptions
	poll     VideoPollOptions
	webhook  model.VideoWebhookFactory
	retry    retryPolicy
}

// videoOperationState accumulates the warnings and metadata of the start
// and every status check.
type videoOperationState struct {
	warnings []model.Warning
	metadata model.ProviderMetadata
}

// run starts the generation and waits for its videos, through the webhook
// when the model handles one and by polling otherwise.
func (o videoOperation) run(ctx context.Context) (*model.VideoResult, error) {
	webhook, warnings, err := o.prepareWebhook(ctx)
	if err != nil {
		return nil, err
	}

	videoOptions := o.options
	videoOptions.Headers = withIdempotencyKey(o.options.Headers)

	startOptions := model.VideoStartOptions{
		VideoOptions: videoOptions,
	}

	if webhook != nil {
		startOptions.WebhookURL = webhook.URL
	}

	started, err := startVideoWithRetry(ctx, o.operator, o.retry, startOptions)
	if err != nil {
		return nil, err
	}

	state := videoOperationState{
		warnings: append(warnings, started.Warnings...),
		metadata: maps.Clone(started.ProviderMetadata),
	}

	if webhook != nil {
		return o.completeAfterWebhook(ctx, webhook, started.Operation, &state)
	}

	return o.pollUntilComplete(ctx, started.Operation, &state)
}

// prepareWebhook creates the webhook endpoint when one is requested and the
// model handles webhooks. Other models poll with an unsupported warning.
func (o videoOperation) prepareWebhook(ctx context.Context) (*model.VideoWebhook, []model.Warning, error) {
	if o.webhook == nil {
		return nil, nil, nil
	}

	handler, ok := o.model.(model.VideoWebhookHandler)
	if !ok {
		warnings := []model.Warning{
			model.UnsupportedWarning("webhook", "This model does not support webhooks. Falling back to polling."),
		}

		return nil, warnings, nil
	}

	webhook, err := handler.HandleWebhook(ctx, o.webhook)
	if err != nil {
		return nil, nil, err
	}

	if webhook == nil || webhook.Wait == nil {
		return nil, nil, fmt.Errorf("%w: video webhook has no wait function", model.ErrInvalidArgument)
	}

	return webhook, nil, nil
}

// completeAfterWebhook waits for the notification, then checks the status
// once. A generation that is still pending fails.
func (o videoOperation) completeAfterWebhook(
	ctx context.Context,
	webhook *model.VideoWebhook,
	operation json.RawMessage,
	state *videoOperationState,
) (*model.VideoResult, error) {
	if err := o.waitForWebhook(ctx, webhook); err != nil {
		return nil, err
	}

	status, err := videoStatusWithRetry(ctx, o.operator, o.retry, o.statusOptions(operation))
	if err != nil {
		return nil, err
	}

	result, err := state.apply(status)
	if err != nil {
		return nil, err
	}

	if result == nil {
		return nil, fmt.Errorf("%w: video generation did not complete after webhook notification", ErrVideoGenerationFailed)
	}

	return result, nil
}

// waitForWebhook waits for the notification until the poll timeout, which
// the delay function measures. The timer goroutine ends before it returns.
func (o videoOperation) waitForWebhook(ctx context.Context, webhook *model.VideoWebhook) error {
	waitCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	var timer sync.WaitGroup
	timer.Go(func() {
		if err := o.poll.Delay(waitCtx, o.poll.Timeout); err == nil {
			cancel(ErrVideoGenerationTimeout)
		}
	})

	_, err := webhook.Wait(waitCtx)

	cancel(nil)
	timer.Wait()

	if err == nil {
		return nil
	}

	if errors.Is(context.Cause(waitCtx), ErrVideoGenerationTimeout) {
		return o.timeoutError()
	}

	return err
}

// pollUntilComplete checks the status after every interval until the
// generation completes, fails or the timeout passes. Each check, including
// its retries, ends at the timeout.
func (o videoOperation) pollUntilComplete(
	ctx context.Context,
	operation json.RawMessage,
	state *videoOperationState,
) (*model.VideoResult, error) {
	deadline := time.Now().Add(o.poll.Timeout)

	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return nil, o.timeoutError()
		}

		if err := o.poll.Delay(ctx, min(o.poll.Interval, remaining)); err != nil {
			return nil, err
		}

		if time.Until(deadline) <= 0 {
			return nil, o.timeoutError()
		}

		status, err := o.statusBefore(ctx, operation, deadline)
		if err != nil {
			return nil, err
		}

		result, err := state.apply(status)
		if err != nil || result != nil {
			return result, err
		}
	}
}

// statusBefore checks the status, retrying until deadline at the latest.
func (o videoOperation) statusBefore(
	ctx context.Context,
	operation json.RawMessage,
	deadline time.Time,
) (*model.VideoStatusResult, error) {
	statusCtx, cancel := context.WithDeadlineCause(ctx, deadline, ErrVideoGenerationTimeout)
	defer cancel()

	status, err := videoStatusWithRetry(statusCtx, o.operator, o.retry, o.statusOptions(operation))
	if err != nil && errors.Is(context.Cause(statusCtx), ErrVideoGenerationTimeout) {
		return nil, o.timeoutError()
	}

	return status, err
}

func (o videoOperation) statusOptions(operation json.RawMessage) model.VideoStatusOptions {
	return model.VideoStatusOptions{
		Operation: operation,
		Headers:   o.options.Headers,
	}
}

func (o videoOperation) timeoutError() error {
	return fmt.Errorf("%w after %s", ErrVideoGenerationTimeout, o.poll.Timeout)
}

// apply records a status check. It returns the result of a completed
// generation, nil while it is pending, and an error when it failed.
func (s *videoOperationState) apply(status *model.VideoStatusResult) (*model.VideoResult, error) {
	if status.Status == model.VideoError {
		return nil, fmt.Errorf("%w: %s", ErrVideoGenerationFailed, status.Error)
	}

	s.warnings = append(s.warnings, status.Warnings...)

	if status.ProviderMetadata != nil {
		if s.metadata == nil {
			s.metadata = model.ProviderMetadata{}
		}

		mergeVideoProviderMetadata(s.metadata, status.ProviderMetadata)
	}

	if status.Status != model.VideoCompleted {
		return nil, nil
	}

	return &model.VideoResult{
		Videos:           status.Videos,
		ProviderMetadata: s.metadata,
		Warnings:         s.warnings,
		Response:         status.Response,
	}, nil
}

func startVideoWithRetry(
	ctx context.Context,
	operator model.VideoOperator,
	retry retryPolicy,
	options model.VideoStartOptions,
) (*model.VideoStartResult, error) {
	return retryCall(ctx, retry, func(ctx context.Context) (*model.VideoStartResult, error) {
		started, err := operator.Start(ctx, options)
		if err != nil {
			return nil, err
		}

		if started == nil {
			return nil, fmt.Errorf("%w: video model returned no start result", model.ErrInvalidResponse)
		}

		return started, nil
	})
}

func videoStatusWithRetry(
	ctx context.Context,
	operator model.VideoOperator,
	retry retryPolicy,
	options model.VideoStatusOptions,
) (*model.VideoStatusResult, error) {
	return retryCall(ctx, retry, func(ctx context.Context) (*model.VideoStatusResult, error) {
		status, err := operator.Status(ctx, options)
		if err != nil {
			return nil, err
		}

		if status == nil {
			return nil, fmt.Errorf("%w: video model returned no status", model.ErrInvalidResponse)
		}

		return status, nil
	})
}

// withIdempotencyKey adds one idempotency key per logical start, so that
// retried starts are not billed twice. A key set by the caller wins.
func withIdempotencyKey(headers http.Header) http.Header {
	for key := range headers {
		if strings.EqualFold(key, idempotencyKeyHeader) {
			return headers
		}
	}

	keyed := headers.Clone()
	if keyed == nil {
		keyed = make(http.Header)
	}

	keyed.Set(idempotencyKeyHeader, "aisdk_vid_"+id.New(""))

	return keyed
}
