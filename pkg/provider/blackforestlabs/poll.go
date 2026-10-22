package blackforestlabs

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/origin"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	defaultPollInterval = 500 * time.Millisecond
	defaultPollTimeout  = 60 * time.Second
)

var (
	// ErrImageGenerationFailed reports an image generation that failed or
	// was moderated. The error names the reported status.
	ErrImageGenerationFailed = errors.New("black forest labs image generation failed")

	// ErrImageGenerationTimeout reports an image generation that was not
	// ready before the poll timeout.
	ErrImageGenerationTimeout = errors.New("black forest labs image generation timed out")
)

// pollSettings controls how a generation is awaited.
type pollSettings struct {
	interval time.Duration
	timeout  time.Duration
}

// pollResponse is the status of a queued generation. Older responses report
// the status as "state".
type pollResponse struct {
	Status *taskStatus  `json:"status"`
	State  *taskStatus  `json:"state"`
	Result *pollResults `json:"result"`
}

// pollResults describes a ready generation.
type pollResults struct {
	Sample    string   `json:"sample"`
	Seed      *float64 `json:"seed"`
	StartTime *float64 `json:"start_time"`
	EndTime   *float64 `json:"end_time"`
	Duration  *float64 `json:"duration"`
}

// pollSettings prefers the call options over the provider settings.
func (m *ImageModel) pollSettings(options ImageOptions) pollSettings {
	settings := pollSettings{
		interval: durationOr(m.provider.pollInterval, defaultPollInterval),
		timeout:  durationOr(m.provider.pollTimeout, defaultPollTimeout),
	}

	if options.PollIntervalMillis != nil {
		settings.interval = time.Duration(*options.PollIntervalMillis) * time.Millisecond
	}

	if options.PollTimeoutMillis != nil {
		settings.timeout = time.Duration(*options.PollTimeoutMillis) * time.Millisecond
	}

	return settings
}

// pollForResult checks the generation until it is ready, fails or the poll
// timeout passes.
func (m *ImageModel) pollForResult(
	ctx context.Context,
	submitted submitResponse,
	headers http.Header,
	settings pollSettings,
) (pollResults, error) {
	pollURL, err := statusURL(submitted)
	if err != nil {
		return pollResults{}, err
	}

	pollCtx, cancel := context.WithTimeoutCause(ctx, settings.timeout, ErrImageGenerationTimeout)
	defer cancel()

	for {
		result, done, err := m.checkStatus(pollCtx, pollURL, headers)
		if err == nil && !done {
			err = poll.Wait(pollCtx, settings.interval)
		}

		if err != nil {
			return pollResults{}, timeoutCause(ctx, pollCtx, settings, err)
		}

		if done {
			return result, nil
		}
	}
}

// statusURL returns the polling URL with the request ID.
func statusURL(submitted submitResponse) (string, error) {
	parsed, err := url.Parse(submitted.PollingURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("%w: invalid polling_url %q", model.ErrInvalidResponse, submitted.PollingURL)
	}

	query := parsed.Query()
	if !query.Has("id") {
		query.Set("id", submitted.ID)
		parsed.RawQuery = query.Encode()
	}

	return parsed.String(), nil
}

// timeoutCause reports the poll timeout, the cause pollCtx was created with,
// instead of the deadline error it caused, unless the caller's context ended
// first.
func timeoutCause(ctx context.Context, pollCtx context.Context, settings pollSettings, err error) error {
	if ctx.Err() != nil || pollCtx.Err() == nil {
		return err
	}

	cause := context.Cause(pollCtx)
	if errors.Is(cause, ErrImageGenerationTimeout) || errors.Is(cause, ErrVideoGenerationTimeout) {
		return fmt.Errorf("%w after %dms", cause, settings.timeout.Milliseconds())
	}

	return err
}

// checkStatus fetches the generation status. It reports true with the
// result when the generation is ready.
func (m *ImageModel) checkStatus(ctx context.Context, pollURL string, headers http.Header) (pollResults, bool, error) {
	body, err := m.provider.fetchStatus(ctx, pollURL, headers)
	if err != nil {
		return pollResults{}, false, err
	}

	var response pollResponse
	if err := apicall.DecodeJSON(body, &response); err != nil {
		return pollResults{}, false, err
	}

	status := response.Status
	if status == nil {
		status = response.State
	}

	if status == nil {
		return pollResults{}, false, fmt.Errorf("%w: missing status in Black Forest Labs poll response", model.ErrInvalidResponse)
	}

	switch *status {
	case statusReady:
		if response.Result == nil || response.Result.Sample == "" {
			return pollResults{}, false, fmt.Errorf("%w: Black Forest Labs poll response is Ready but missing result.sample", model.ErrInvalidResponse)
		}

		return *response.Result, true, nil

	case statusError, statusFailed, statusRequestModerated, statusContentModerated, statusTaskNotFound:
		return pollResults{}, false, fmt.Errorf("%w: %s", ErrImageGenerationFailed, *status)

	case statusPending, statusGenerating, statusReasoning:
		return pollResults{}, false, nil

	default:
		return pollResults{}, false, fmt.Errorf("%w: unknown Black Forest Labs status %q", model.ErrInvalidResponse, *status)
	}
}

// fetchStatus reads the polling URL, which comes from the response. Trusted
// Black Forest Labs hosts receive the credentials; other hosts are validated
// and receive none.
func (p *Provider) fetchStatus(ctx context.Context, pollURL string, headers http.Header) ([]byte, error) {
	if !p.isTrustedURL(pollURL) {
		downloaded, err := download.Fetch(ctx, download.Request{
			URL:           pollURL,
			TrustedOrigin: p.baseURL,
			Client:        p.httpClient,
		})
		if err != nil {
			return nil, err
		}

		return downloaded.Data, nil
	}

	if !origin.Same(pollURL, p.baseURL) {
		if err := download.ValidateURL(pollURL); err != nil {
			return nil, err
		}
	}

	response, err := apicall.Get(ctx, p.apiRequest(pollURL, headers))
	if err != nil {
		return nil, classifyError(err)
	}

	return response.Body, nil
}

// isTrustedURL reports whether location may receive the API key: the base
// URL origin, or an https host under bfl.ai. Polling URLs point at sibling
// cluster hosts such as api.us1.bfl.ai.
func (p *Provider) isTrustedURL(location string) bool {
	if origin.Same(location, p.baseURL) {
		return true
	}

	parsed, err := url.Parse(location)
	if err != nil {
		return false
	}

	return parsed.Scheme == "https" && isBFLHost(parsed.Hostname())
}

func isBFLHost(host string) bool {
	return host == "bfl.ai" || strings.HasSuffix(host, ".bfl.ai")
}

func hostname(location string) string {
	parsed, err := url.Parse(location)
	if err != nil {
		return ""
	}

	return parsed.Hostname()
}

// durationOr returns value, or fallback when value is not set.
func durationOr(value time.Duration, fallback time.Duration) time.Duration {
	if value <= 0 {
		return fallback
	}

	return value
}
