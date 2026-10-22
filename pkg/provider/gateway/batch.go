package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"regexp"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// batchDefaultMaxLineBytes bounds one line of a results stream.
const batchDefaultMaxLineBytes = 64 << 20

var (
	_ model.Batch          = (*Batch)(nil)
	_ model.BatchCanceller = (*Batch)(nil)
	_ model.BatchProvider  = (*Provider)(nil)
)

// WithBatchResultMaxLineBytes bounds one line of a streamed batch result.
// The default is 64 MiB.
func WithBatchResultMaxLineBytes(maxLineBytes int) Option {
	return func(p *Provider) {
		p.batchMaxLineBytes = maxLineBytes
	}
}

// Batch returns the batch interface, an *Batch that runs durable batches of
// text requests through the gateway. Batches are experimental.
func (p *Provider) Batch() model.Batch {
	return &Batch{
		provider: p,
	}
}

// Batch runs text requests through the gateway's batch surface. The batch
// ID is the gateway job ID; provider batch IDs stay on the gateway. The
// batch contract is experimental and may change.
type Batch struct {
	provider *Provider
}

// Provider returns the provider identifier, "gateway.batch" by default.
func (b *Batch) Provider() string {
	return b.provider.name + ".batch"
}

// SupportedURLs reports that the gateway downloads file URLs of every media
// type.
func (b *Batch) SupportedURLs() map[string][]*regexp.Regexp {
	return supportedURLs
}

// batchStartRequest is the body of a batch start.
type batchStartRequest struct {
	CallbackURL     string                `json:"callbackUrl,omitempty"`
	Requests        []batchStartEntry     `json:"requests"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// batchStartEntry is one request of a batch start.
type batchStartEntry struct {
	ID      string      `json:"id"`
	Type    string      `json:"type"`
	ModelID string      `json:"modelId"`
	Options callOptions `json:"options"`
}

// batchOperationRequest identifies a batch for status, results and cancel
// calls.
type batchOperationRequest struct {
	BatchID string `json:"batchId"`
}

// batchStatusResponse is a batch status on the wire.
type batchStatusResponse struct {
	Status           model.BatchState    `json:"status"`
	RawStatus        *string             `json:"rawStatus"`
	RequestCounts    *batchRequestCounts `json:"requestCounts"`
	Error            *batchError         `json:"error"`
	CreatedAt        *string             `json:"createdAt"`
	ExpiresAt        *string             `json:"expiresAt"`
	ProviderMetadata wireMetadata        `json:"providerMetadata"`
}

// batchStartResponse is the status of a started batch with its ID and
// warnings.
type batchStartResponse struct {
	batchStatusResponse

	BatchID  string              `json:"batchId"`
	Warnings []batchWarningEntry `json:"warnings"`
}

type batchRequestCounts struct {
	Total     *int `json:"total"`
	Pending   *int `json:"pending"`
	Completed *int `json:"completed"`
	Failed    *int `json:"failed"`
}

type batchError struct {
	Message    string  `json:"message"`
	Type       *string `json:"type"`
	Code       *string `json:"code"`
	StatusCode *int    `json:"statusCode"`
}

type batchWarningEntry struct {
	RequestID *string     `json:"requestId"`
	Warning   wireWarning `json:"warning"`
}

// StartBatch starts a durable batch. All requests must use the same model.
// A gateway idempotency key makes retries replay the original batch.
func (b *Batch) StartBatch(ctx context.Context, options model.BatchStartOptions) (*model.BatchStartResult, error) {
	requests, modelID, err := gatewayBatchRequests(options.Requests)
	if err != nil {
		return nil, err
	}

	idempotencyKey, forwardedOptions, err := splitIdempotencyKey(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	body, err := batchStartBody(requests, options.WebhookURL, forwardedOptions)
	if err != nil {
		return nil, err
	}

	modelHeaders := http.Header{
		"Ai-Model-Id": {
			modelID,
		},
	}

	if idempotencyKey != "" {
		modelHeaders.Set("Idempotency-Key", idempotencyKey)
	}

	var decoded batchStartResponse
	if err := b.post(ctx, "start", options.Headers, modelHeaders, body, &decoded); err != nil {
		return nil, err
	}

	if decoded.BatchID == "" {
		return nil, fmt.Errorf("%w: batch start response has no batchId", model.ErrInvalidResponse)
	}

	status, err := decoded.status()
	if err != nil {
		return nil, err
	}

	return &model.BatchStartResult{
		BatchID:     decoded.BatchID,
		BatchStatus: status,
		Warnings:    batchWarnings(decoded.Warnings),
	}, nil
}

// gatewayBatchRequests accepts at least one text request, all for the same
// model.
func gatewayBatchRequests(requests []model.BatchRequest) ([]model.TextBatchRequest, string, error) {
	if len(requests) == 0 {
		return nil, "", fmt.Errorf("%w: the AI Gateway Batch API requires at least one request", model.ErrInvalidArgument)
	}

	textRequests := make([]model.TextBatchRequest, 0, len(requests))

	for _, request := range requests {
		textRequest, ok := request.(model.TextBatchRequest)
		if !ok {
			return nil, "", fmt.Errorf("%w: the AI Gateway Batch API does not support batch requests of type %T", model.ErrUnsupported, request)
		}

		if len(textRequests) > 0 && textRequest.ModelID != textRequests[0].ModelID {
			return nil, "", fmt.Errorf(
				"%w: the AI Gateway Batch API requires all requests in a batch to use the same model; found %q and %q",
				model.ErrInvalidArgument,
				textRequests[0].ModelID,
				textRequest.ModelID,
			)
		}

		textRequests = append(textRequests, textRequest)
	}

	return textRequests, textRequests[0].ModelID, nil
}

// splitIdempotencyKey removes the gateway idempotency key from the provider
// options. The key travels as a header, so that equivalent retries keep the
// same body.
func splitIdempotencyKey(providerOptions model.ProviderOptions) (string, model.ProviderOptions, error) {
	raw, found, err := model.ProviderValue[json.RawMessage](providerOptions, Namespace)
	if err != nil || !found || !jsonobject.IsObject(raw) {
		return "", providerOptions, err
	}

	gatewayOptions, err := jsonobject.Parse(raw)
	if err != nil {
		return "", nil, fmt.Errorf("%w: gateway provider options: %w", model.ErrInvalidArgument, err)
	}

	if !gatewayOptions.Has("idempotencyKey") {
		return "", providerOptions, nil
	}

	key, _ := gatewayOptions.GetString("idempotencyKey")
	remaining := gatewayOptions.Without("idempotencyKey")

	forwarded := make(model.ProviderOptions, len(providerOptions))
	maps.Copy(forwarded, providerOptions)

	if len(remaining) == 0 {
		delete(forwarded, Namespace)

		return key, forwarded, nil
	}

	encoded, err := remaining.Encode()
	if err != nil {
		return "", nil, fmt.Errorf("%w: gateway provider options: %w", model.ErrInvalidArgument, err)
	}

	forwarded[Namespace] = encoded

	return key, forwarded, nil
}

// batchStartBody encodes every request's call options like a language model
// call. Headers and raw chunks do not apply to batch requests.
func batchStartBody(requests []model.TextBatchRequest, webhookURL string, providerOptions model.ProviderOptions) (batchStartRequest, error) {
	entries := make([]batchStartEntry, 0, len(requests))

	for _, request := range requests {
		callOptions := request.Options
		callOptions.Headers = nil
		callOptions.IncludeRawChunks = false

		encoded, err := encodeCallOptions(callOptions)
		if err != nil {
			return batchStartRequest{}, err
		}

		entries = append(entries, batchStartEntry{
			ID:      request.ID,
			Type:    "text",
			ModelID: request.ModelID,
			Options: encoded,
		})
	}

	return batchStartRequest{
		CallbackURL:     webhookURL,
		Requests:        entries,
		ProviderOptions: providerOptions,
	}, nil
}

func batchWarnings(entries []batchWarningEntry) []model.BatchWarning {
	if len(entries) == 0 {
		return nil
	}

	warnings := make([]model.BatchWarning, 0, len(entries))

	for _, entry := range entries {
		warnings = append(warnings, model.BatchWarning{
			RequestID: stringValue(entry.RequestID),
			Warning:   decodeWarnings([]wireWarning{entry.Warning})[0],
		})
	}

	return warnings
}

// BatchStatus returns the latest status of a gateway batch job.
func (b *Batch) BatchStatus(ctx context.Context, options model.BatchOperationOptions) (*model.BatchStatus, error) {
	var decoded batchStatusResponse
	if err := b.post(ctx, "status", options.Headers, nil, batchOperationRequest{
		BatchID: options.BatchID,
	}, &decoded); err != nil {
		return nil, err
	}

	status, err := decoded.status()
	if err != nil {
		return nil, err
	}

	return &status, nil
}

// CancelBatch requests the cancellation of a gateway batch job. Status and
// partial results remain separate reads.
func (b *Batch) CancelBatch(ctx context.Context, options model.BatchOperationOptions) (*model.BatchCancelResult, error) {
	var decoded batchStatusResponse
	if err := b.post(ctx, "cancel", options.Headers, nil, batchOperationRequest{
		BatchID: options.BatchID,
	}, &decoded); err != nil {
		return nil, err
	}

	if err := decoded.validate(); err != nil {
		return nil, err
	}

	return &model.BatchCancelResult{
		ProviderMetadata: decoded.ProviderMetadata.providerMetadata(),
	}, nil
}

// post sends a batch call and decodes its JSON response into target.
func (b *Batch) post(ctx context.Context, operation string, callHeaders http.Header, modelHeaders http.Header, body any, target any) error {
	request, method, err := b.provider.modelRequest("/batch/"+operation, callHeaders, modelHeaders)
	if err != nil {
		return err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return batchCallError(err, method)
	}

	return apicall.DecodeJSON(response.Body, target)
}

// batchCallError converts failures into gateway errors, keeping
// cancellation: an aborted start may still have been accepted, so it must
// not become a retryable gateway error.
func batchCallError(err error, method authMethod) error {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}

	return asGatewayError(err, method)
}

func (r batchStatusResponse) validate() error {
	switch r.Status {
	case model.BatchPending, model.BatchCompleted, model.BatchFailed:
		return nil
	default:
		return fmt.Errorf("%w: unknown batch status %q", model.ErrInvalidResponse, r.Status)
	}
}

// status converts the wire status.
func (r batchStatusResponse) status() (model.BatchStatus, error) {
	if err := r.validate(); err != nil {
		return model.BatchStatus{}, err
	}

	createdAt, err := batchTime("createdAt", r.CreatedAt)
	if err != nil {
		return model.BatchStatus{}, err
	}

	expiresAt, err := batchTime("expiresAt", r.ExpiresAt)
	if err != nil {
		return model.BatchStatus{}, err
	}

	status := model.BatchStatus{
		State:            r.Status,
		RawStatus:        stringValue(r.RawStatus),
		CreatedAt:        createdAt,
		ExpiresAt:        expiresAt,
		ProviderMetadata: r.ProviderMetadata.providerMetadata(),
	}

	if counts := r.RequestCounts; counts != nil {
		status.RequestCounts = model.NormalizeBatchRequestCounts(counts.Total, counts.Pending, counts.Completed, counts.Failed)
	}

	if r.Error != nil {
		status.Error = r.Error.batchError()
	}

	return status, nil
}

func (e *batchError) batchError() *model.BatchError {
	if e == nil {
		return nil
	}

	converted := &model.BatchError{
		Message: e.Message,
		Type:    stringValue(e.Type),
		Code:    stringValue(e.Code),
	}

	if e.StatusCode != nil {
		converted.StatusCode = *e.StatusCode
	}

	return converted
}

// batchTime parses an optional RFC 3339 timestamp.
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
