package topaz

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/origin"
	"github.com/aarock1234/ai/pkg/model"
)

// Video API paths.
const (
	videoExpressPath = "/video/express"
	videoPathPrefix  = "/video/"
	videoStatusPath  = "/status"
)

// Request statuses of video enhancement. Others, such as processing, are
// pending.
const (
	videoComplete = "complete"
	videoFailed   = "failed"
	videoCanceled = "canceled"
)

// maxUploadErrorBytes bounds the body read from a failed upload.
const maxUploadErrorBytes = 64 << 10

var (
	_ model.VideoModel    = (*VideoModel)(nil)
	_ model.VideoOperator = (*VideoModel)(nil)
	_ model.VideoProvider = (*Provider)(nil)
)

// VideoModel is a Topaz video enhancement model, such as "proteus" or
// "starlight-precise-2.6". It enhances the first video of the input
// references: Start creates an express request, which Topaz processes once
// it has the video, and Status checks it. Topaz fetches URL inputs itself;
// inline inputs are uploaded to the URL Topaz returns.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// VideoOperation identifies an express request. It is the JSON operation
// reference of Start and Status.
type VideoOperation struct {
	RequestID string `json:"requestId"`

	// OutputContainer is the container of the enhanced video, which selects
	// its media type.
	OutputContainer string `json:"outputContainer,omitempty"`
}

// VideoMetadata is the provider metadata of a video request, stored under
// the "topaz" namespace. Nil and empty fields were not reported.
type VideoMetadata struct {
	RequestID string `json:"requestId"`

	// Credits are the billed credits of a completed request: the lower
	// bound of the estimate.
	Credits *float64 `json:"credits,omitempty"`

	// EstimatedCredits are the lower and upper bounds of the cost estimate.
	// At start they are preliminary.
	EstimatedCredits []float64 `json:"estimatedCredits,omitempty"`

	// OutputSize and ExpiresAt are strings or numbers.
	OutputSize json.RawMessage `json:"outputSize,omitempty"`
	ExpiresAt  json.RawMessage `json:"expiresAt,omitempty"`

	ErrorCode *string `json:"errorCode,omitempty"`
}

type expressResponse struct {
	RequestID  *string         `json:"requestId"`
	UploadURLs []string        `json:"uploadUrls"`
	Estimates  *videoEstimates `json:"estimates"`
}

// videoEstimates are [lower, upper] bounds of the cost in credits and the
// time in seconds.
type videoEstimates struct {
	Cost []float64 `json:"cost"`
	Time []float64 `json:"time"`
}

type videoStatusResponse struct {
	Status    *string         `json:"status"`
	Message   *string         `json:"message"`
	ErrorCode *string         `json:"errorCode"`
	Estimates *videoEstimates `json:"estimates"`
	Download  *videoDownload  `json:"download"`

	// OutputSize is documented as a string and accepted as a number.
	OutputSize json.RawMessage `json:"outputSize"`
}

type videoDownload struct {
	URL       *string         `json:"url"`
	ExpiresAt json.RawMessage `json:"expiresAt"`
}

// Video returns a video enhancement model. Readable IDs are mapped onto
// Topaz API model names; other IDs are sent unchanged.
func (p *Provider) Video(modelID string) *VideoModel {
	return &VideoModel{
		modelID:  modelID,
		provider: p,
	}
}

// VideoModel returns Video(modelID).
func (p *Provider) VideoModel(modelID string) (model.VideoModel, error) {
	return p.Video(modelID), nil
}

// Provider returns the provider identifier, "topaz.video" by default.
func (m *VideoModel) Provider() string {
	return m.provider.name + ".video"
}

// ModelID returns the model identifier.
func (m *VideoModel) ModelID() string {
	return m.modelID
}

// MaxVideosPerCall returns 1.
func (m *VideoModel) MaxVideosPerCall() int {
	return 1
}

// Start creates an express request and uploads an inline input video. A
// request whose upload fails is canceled, which refunds reserved credits.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := time.Now()

	videoOptions, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	headers, err := m.provider.authHeaders(options.Headers)
	if err != nil {
		return nil, err
	}

	plan, warnings, err := m.buildExpressPlan(options.VideoOptions, videoOptions)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, m.provider.apiRequest(videoExpressPath, headers), plan.request)
	if err != nil {
		return nil, err
	}

	var created expressResponse
	if err := apicall.DecodeJSON(response.Body, &created); err != nil {
		return nil, err
	}

	if created.RequestID == nil {
		return nil, fmt.Errorf("%w: Topaz did not return a requestId for the video request", model.ErrInvalidResponse)
	}

	requestID := *created.RequestID

	if plan.upload != nil {
		if err := m.uploadVideo(ctx, requestID, plan, created.UploadURLs); err != nil {
			m.cancelQuietly(ctx, requestID, headers)

			return nil, err
		}
	}

	operation, err := json.Marshal(VideoOperation{
		RequestID:       requestID,
		OutputContainer: plan.request.Output.Container,
	})
	if err != nil {
		return nil, fmt.Errorf("encode video operation: %w", err)
	}

	metadata := VideoMetadata{
		RequestID: requestID,
	}

	if created.Estimates != nil {
		metadata.EstimatedCredits = created.Estimates.Cost
	}

	return &model.VideoStartResult{
		Operation: operation,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: metadata,
		},
		Warnings: warnings,
		Response: m.responseInfo(timestamp, response),
	}, nil
}

// Status checks an express request. Failed and canceled requests report an
// error status; a completed request without a download URL returns an
// error wrapping model.ErrInvalidResponse.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	timestamp := time.Now()

	var operation VideoOperation
	if err := json.Unmarshal(options.Operation, &operation); err != nil {
		return nil, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	headers, err := m.provider.authHeaders(options.Headers)
	if err != nil {
		return nil, err
	}

	statusPath := videoPathPrefix + url.PathEscape(operation.RequestID) + videoStatusPath

	response, err := apicall.Get(ctx, m.provider.apiRequest(statusPath, headers))
	if err != nil {
		return nil, err
	}

	var status videoStatusResponse
	if err := apicall.DecodeJSON(response.Body, &status); err != nil {
		return nil, err
	}

	result := &model.VideoStatusResult{
		Status:   model.VideoPending,
		Response: m.responseInfo(timestamp, response),
	}

	state := ""
	if status.Status != nil {
		state = *status.Status
	}

	switch state {
	case videoComplete:
		return completedVideo(result, operation, status)

	case videoFailed, videoCanceled:
		result.Status = model.VideoError
		result.Error = failureMessage(operation.RequestID, state, status)
		result.ProviderMetadata = model.ProviderMetadata{
			Namespace: VideoMetadata{
				RequestID: operation.RequestID,
				ErrorCode: status.ErrorCode,
			},
		}

		return result, nil

	default:
		return result, nil
	}
}

// completedVideo reports the enhanced video and its billing. Topaz
// invoices the lower bound of the cost estimate, which it recomputes once
// it has the input.
func completedVideo(result *model.VideoStatusResult, operation VideoOperation, status videoStatusResponse) (*model.VideoStatusResult, error) {
	if status.Download == nil || status.Download.URL == nil {
		return nil, fmt.Errorf(
			"%w: Topaz reported request %s complete but returned no download URL",
			model.ErrInvalidResponse,
			operation.RequestID,
		)
	}

	mediaType, ok := containerMediaTypes[operation.OutputContainer]
	if !ok {
		mediaType = containerMediaTypes[defaultOutputContainer]
	}

	metadata := VideoMetadata{
		RequestID:  operation.RequestID,
		OutputSize: presentJSON(status.OutputSize),
		ExpiresAt:  presentJSON(status.Download.ExpiresAt),
	}

	if status.Estimates != nil && len(status.Estimates.Cost) > 0 {
		metadata.Credits = new(slices.Min(status.Estimates.Cost))
		metadata.EstimatedCredits = status.Estimates.Cost
	}

	result.Status = model.VideoCompleted
	result.Videos = []model.VideoData{
		model.VideoDataURL{
			URL:       *status.Download.URL,
			MediaType: mediaType,
		},
	}
	result.ProviderMetadata = model.ProviderMetadata{
		Namespace: metadata,
	}

	return result, nil
}

// failureMessage describes a failed or canceled request with its code and
// message.
func failureMessage(requestID string, state string, status videoStatusResponse) string {
	message := fmt.Sprintf("Topaz video request %s %s", requestID, state)

	if status.ErrorCode != nil {
		message += " (" + *status.ErrorCode + ")"
	}

	if status.Message != nil {
		return message + ": " + *status.Message
	}

	return message + "."
}

// uploadVideo uploads the inline input to the first upload URL. The URL
// comes from the response, so it is validated unless it is on the API's
// origin, and it never receives the API key: presigned URLs carry their own
// credentials.
func (m *VideoModel) uploadVideo(ctx context.Context, requestID string, plan expressPlan, uploadURLs []string) error {
	if len(uploadURLs) == 0 {
		return fmt.Errorf("%w: Topaz returned no upload URL for request %s", model.ErrInvalidResponse, requestID)
	}

	uploadURL := uploadURLs[0]

	if !origin.Same(uploadURL, m.provider.baseURL) {
		if err := download.ValidateURL(uploadURL); err != nil {
			return err
		}
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPut, uploadURL, bytes.NewReader(plan.upload))
	if err != nil {
		return fmt.Errorf("%w: build upload request: %w", model.ErrInvalidArgument, err)
	}

	request.Header.Set("Content-Type", plan.uploadMediaType)

	response, err := m.provider.client().Do(request)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}

		return &model.APICallError{
			Message: "uploading the input video failed",
			URL:     uploadURL,
			Cause:   err,
		}
	}
	defer closeQuietly(response.Body)

	if response.StatusCode >= 200 && response.StatusCode <= 299 {
		return nil
	}

	body, _ := io.ReadAll(io.LimitReader(response.Body, maxUploadErrorBytes))

	return &model.APICallError{
		Message:         fmt.Sprintf("Uploading the input video failed with status %d.", response.StatusCode),
		URL:             uploadURL,
		StatusCode:      response.StatusCode,
		ResponseHeaders: response.Header,
		ResponseBody:    string(body),
		IsRetryable:     model.RetryableStatus(response.StatusCode),
	}
}

// cancelQuietly cancels a request before processing starts. It is detached
// from the caller's cancellation, which may be why the start failed, and
// its failure is ignored in favor of the original error.
func (m *VideoModel) cancelQuietly(ctx context.Context, requestID string, headers http.Header) {
	cancelCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cancelTimeout)
	defer cancel()

	request, err := http.NewRequestWithContext(cancelCtx, http.MethodDelete, m.provider.baseURL+videoPathPrefix+url.PathEscape(requestID), nil)
	if err != nil {
		return
	}

	request.Header = headers.Clone()

	response, err := m.provider.client().Do(request)
	if err != nil {
		return
	}

	closeQuietly(response.Body)
}

func (m *VideoModel) responseInfo(timestamp time.Time, response *apicall.Response) model.ResponseInfo {
	return model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}
}

// presentJSON returns value unless it is absent or null.
func presentJSON(value json.RawMessage) json.RawMessage {
	trimmed := bytes.TrimSpace(value)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}

	return trimmed
}

// closeQuietly closes a response body whose outcome is already decided.
func closeQuietly(body io.Closer) {
	_ = body.Close()
}
