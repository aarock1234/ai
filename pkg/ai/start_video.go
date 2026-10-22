package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// StartVideoOptions configures StartVideo.
type StartVideoOptions struct {
	VideoSettings

	// WebhookURL is where the provider should send a notification when the
	// generation completes.
	WebhookURL string
}

// StartVideoResult is the outcome of StartVideo.
type StartVideoResult struct {
	// Operation is an opaque JSON reference to the generation. Persist it
	// and pass it to GetVideoStatus, from any process.
	Operation json.RawMessage

	Warnings []model.Warning

	// ProviderMetadata carries the provider's own job identifiers.
	ProviderMetadata model.ProviderMetadata

	Response model.ResponseInfo
}

// GetVideoStatusOptions configures GetVideoStatus.
type GetVideoStatusOptions struct {
	// Model is the model the operation was started on.
	Model model.VideoModel

	// Operation is the reference returned by StartVideo.
	Operation json.RawMessage

	// MaxRetries is the number of retries of the status call. Nil selects 2;
	// zero disables retries.
	MaxRetries *int

	Headers http.Header
}

// GetVideoStatusResult is the status of a generation: pending, completed
// with its videos, or failed with an error message.
type GetVideoStatusResult = model.VideoStatusResult

// StartVideo starts an asynchronous video generation and returns without
// waiting for it, for fan-out, short-lived processes or webhook delivery.
// Check the outcome with GetVideoStatus. The model must implement
// model.VideoOperator, and N must not exceed its per-call limit: one
// operation covers every requested video. The video API is experimental and
// may change.
func StartVideo(ctx context.Context, options StartVideoOptions) (*StartVideoResult, error) {
	if err := options.validate(); err != nil {
		return nil, err
	}

	operator, ok := options.Model.(model.VideoOperator)
	if !ok {
		return nil, fmt.Errorf(
			"%w: video model %s does not implement Start; use GenerateVideo for models without an asynchronous start/status flow",
			model.ErrUnsupported,
			options.Model.ModelID(),
		)
	}

	count := max(options.N, 1)
	if limit := knownVideosPerCall(options.VideoSettings); limit > 0 && count > limit {
		return nil, &InvalidArgumentError{
			Parameter: "n",
			Value:     count,
			Message: fmt.Sprintf(
				"video model %s supports at most %d video(s) per call, but %d were requested; split the batch across multiple StartVideo calls",
				options.Model.ModelID(),
				limit,
				count,
			),
		}
	}

	inputs, err := normalizeVideoInputs(options.VideoSettings)
	if err != nil {
		return nil, err
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	videoOptions := inputs.callOptions(options.VideoSettings)
	videoOptions.N = count
	videoOptions.Headers = withIdempotencyKey(options.Headers)

	startOptions := model.VideoStartOptions{
		VideoOptions: videoOptions,
		WebhookURL:   options.WebhookURL,
	}

	started, err := startVideoWithRetry(ctx, operator, retry, startOptions)
	if err != nil {
		return nil, err
	}

	return &StartVideoResult{
		Operation:        started.Operation,
		Warnings:         slices.Concat(inputs.warnings, started.Warnings),
		ProviderMetadata: started.ProviderMetadata,
		Response:         started.Response,
	}, nil
}

// GetVideoStatus checks a generation started with StartVideo once, without
// polling. The model must implement model.VideoOperator. The video API is
// experimental and may change.
func GetVideoStatus(ctx context.Context, options GetVideoStatusOptions) (*GetVideoStatusResult, error) {
	if options.Model == nil {
		return nil, &InvalidArgumentError{
			Parameter: "model",
			Message:   "model is required",
		}
	}

	operator, ok := options.Model.(model.VideoOperator)
	if !ok {
		return nil, fmt.Errorf(
			"%w: video model %s does not implement Status",
			model.ErrUnsupported,
			options.Model.ModelID(),
		)
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	return videoStatusWithRetry(ctx, operator, retry, model.VideoStatusOptions{
		Operation: options.Operation,
		Headers:   options.Headers,
	})
}
