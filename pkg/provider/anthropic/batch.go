package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// batchRequestIDPattern is the form Anthropic accepts for custom IDs.
var batchRequestIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

var (
	_ model.Batch          = (*Batch)(nil)
	_ model.BatchCanceller = (*Batch)(nil)
	_ model.BatchLister    = (*Batch)(nil)
	_ model.BatchProvider  = (*Provider)(nil)
)

// BatchOptions are the batch settings read from the "anthropic" provider
// options namespace and, for a renamed provider, its own namespace.
type BatchOptions struct {
	// AnthropicBeta lists beta features enabled for the whole batch. Message
	// Batches do not accept betas on individual requests.
	AnthropicBeta []string `json:"anthropicBeta,omitempty"`
}

// BatchMetadata is the "anthropic" provider metadata of a batch status.
type BatchMetadata struct {
	ArchivedAt        *string            `json:"archivedAt"`
	CancelInitiatedAt *string            `json:"cancelInitiatedAt"`
	EndedAt           *string            `json:"endedAt"`
	RequestCounts     BatchRequestCounts `json:"requestCounts"`
	ResultsURL        *string            `json:"resultsUrl"`
}

// BatchRequestCounts are Anthropic's own request counts of a batch.
type BatchRequestCounts struct {
	Processing int `json:"processing"`
	Succeeded  int `json:"succeeded"`
	Errored    int `json:"errored"`
	Canceled   int `json:"canceled"`
	Expired    int `json:"expired"`
}

// BatchItemMetadata is the "anthropic" provider metadata of a failed batch
// request.
type BatchItemMetadata struct {
	RequestID string `json:"requestId"`
}

// WithBatchResultMaxLineBytes bounds one line of a downloaded batch result
// file. The default is 64 MiB.
func WithBatchResultMaxLineBytes(maxLineBytes int) Option {
	return func(p *Provider) {
		p.batchMaxLineBytes = maxLineBytes
	}
}

// Batch returns the batch interface, an *Batch that processes Messages API
// requests with the Message Batches API. Batches are experimental.
func (p *Provider) Batch() model.Batch {
	return &Batch{
		provider: p,
	}
}

// Batch processes text requests with the Anthropic Message Batches API. The
// batch contract is experimental and may change.
type Batch struct {
	provider *Provider
}

// Provider returns the provider identifier, "anthropic.batch" by default.
func (b *Batch) Provider() string {
	return b.provider.baseName() + ".batch"
}

// SupportedURLs reports the URLs the Messages API downloads itself.
func (b *Batch) SupportedURLs() map[string][]*regexp.Regexp {
	return b.provider.Messages("").SupportedURLs()
}

// StartBatch prepares every request like a Messages API call and creates
// the batch.
func (b *Batch) StartBatch(ctx context.Context, options model.BatchStartOptions) (*model.BatchStartResult, error) {
	requests, err := textBatchRequests(options.Requests)
	if err != nil {
		return nil, err
	}

	explicitBetas, err := b.batchBetas(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	configuredHeaders, err := b.provider.configuredHeaders(ctx)
	if err != nil {
		return nil, err
	}

	userBetas := slices.Concat(explicitBetas, betasFromHeaders(configuredHeaders, options.Headers))

	var warnings []model.BatchWarning
	if options.WebhookURL != "" {
		warnings = append(warnings, model.BatchWarning{
			Warning: model.UnsupportedWarning(
				"webhookUrl",
				"The Anthropic Message Batches API does not support completion webhooks.",
			),
		})
	}

	prepared, err := b.prepareRequests(requests, userBetas)
	if err != nil {
		return nil, err
	}

	warnings = append(warnings, prepared.warnings...)

	batch, err := b.createBatch(ctx, prepared, configuredHeaders, options.Headers)
	if err != nil {
		return nil, err
	}

	status, err := batch.status()
	if err != nil {
		return nil, err
	}

	return &model.BatchStartResult{
		BatchID:     batch.ID,
		BatchStatus: status,
		Warnings:    warnings,
	}, nil
}

// textBatchRequests accepts only text requests with valid, unique IDs.
func textBatchRequests(requests []model.BatchRequest) ([]model.TextBatchRequest, error) {
	textRequests := make([]model.TextBatchRequest, 0, len(requests))
	seen := make(map[string]bool, len(requests))

	for _, request := range requests {
		textRequest, ok := request.(model.TextBatchRequest)
		if !ok {
			return nil, fmt.Errorf(
				"%w: the Anthropic Message Batches API does not support batch requests of type %T",
				model.ErrUnsupported,
				request,
			)
		}

		if !batchRequestIDPattern.MatchString(textRequest.ID) {
			return nil, fmt.Errorf(
				"%w: Anthropic batch request ID %q must match ^[A-Za-z0-9_-]{1,64}$",
				model.ErrInvalidArgument,
				textRequest.ID,
			)
		}

		if seen[textRequest.ID] {
			return nil, fmt.Errorf(
				"%w: Anthropic batch request IDs must be unique; duplicate ID %q",
				model.ErrInvalidArgument,
				textRequest.ID,
			)
		}

		seen[textRequest.ID] = true
		textRequests = append(textRequests, textRequest)
	}

	return textRequests, nil
}

// batchBetas reads the beta features from "anthropic" and, for a renamed
// provider, from its own namespace, which takes precedence.
func (b *Batch) batchBetas(providerOptions model.ProviderOptions) ([]string, error) {
	options, _, err := model.ProviderValue[BatchOptions](providerOptions, Namespace)
	if err != nil {
		return nil, err
	}

	namespace := b.provider.optionsNamespace()
	if namespace == Namespace {
		return options.AnthropicBeta, nil
	}

	custom, found, err := model.ProviderValue[BatchOptions](providerOptions, namespace)
	if err != nil {
		return nil, err
	}

	if found && custom.AnthropicBeta != nil {
		return custom.AnthropicBeta, nil
	}

	return options.AnthropicBeta, nil
}

// batchRequestEntry is one request of a batch creation.
type batchRequestEntry struct {
	CustomID string `json:"custom_id"`
	Params   any    `json:"params"`
}

// batchCreateRequest is the body of a batch creation.
type batchCreateRequest struct {
	Requests []batchRequestEntry `json:"requests"`
}

// preparedBatch is the request entries of a batch, with the beta features
// they need and the warnings of their preparation.
type preparedBatch struct {
	entries  []batchRequestEntry
	betas    *betaSet
	warnings []model.BatchWarning
}

func (b *Batch) prepareRequests(requests []model.TextBatchRequest, userBetas []string) (*preparedBatch, error) {
	prepared := &preparedBatch{
		entries: make([]batchRequestEntry, 0, len(requests)),
		betas:   newBetaSet(),
	}

	prepared.betas.add(userBetas...)

	for _, request := range requests {
		entry, call, err := b.prepareRequest(request, userBetas)
		if err != nil {
			return nil, err
		}

		prepared.entries = append(prepared.entries, entry)
		prepared.betas.add(call.betas.names...)

		for _, warning := range call.warnings {
			prepared.warnings = append(prepared.warnings, model.BatchWarning{
				RequestID: entry.CustomID,
				Warning:   warning,
			})
		}
	}

	return prepared, nil
}

// prepareRequest converts one batch request into the parameters of a
// non-streaming Messages API call, rejecting the features whose results a
// batch cannot convert.
func (b *Batch) prepareRequest(request model.TextBatchRequest, userBetas []string) (batchRequestEntry, *preparedRequest, error) {
	requestBetas, err := b.batchBetas(request.Options.ProviderOptions)
	if err != nil {
		return batchRequestEntry{}, nil, err
	}

	if len(requestBetas) > 0 {
		return batchRequestEntry{}, nil, fmt.Errorf(
			"%w: Anthropic Message Batches do not support per-request betas (request %q); set providerOptions.anthropic.anthropicBeta on StartBatch instead",
			model.ErrUnsupported,
			request.ID,
		)
	}

	messages := b.provider.Messages(request.ModelID)

	prepared, err := messages.prepareRequest(request.Options, false, userBetas)
	if err != nil {
		return batchRequestEntry{}, nil, err
	}

	if err := validateBatchRequest(request, prepared); err != nil {
		return batchRequestEntry{}, nil, err
	}

	params, err := b.provider.host.transformedBody(prepared.body, prepared.betas.names)
	if err != nil {
		return batchRequestEntry{}, nil, err
	}

	entry := batchRequestEntry{
		CustomID: request.ID,
		Params:   params,
	}

	return entry, prepared, nil
}

// validateBatchRequest rejects requests whose results cannot be converted
// without the original call, and settings batches do not accept.
func validateBatchRequest(request model.TextBatchRequest, prepared *preparedRequest) error {
	if prepared.usesJSONResponseTool {
		return fmt.Errorf(
			"%w: Anthropic Message Batches cannot decode the JSON-tool structured-output fallback (request %q) because batch results are retrieved independently of the start call; use a model that supports native output_format structured outputs",
			model.ErrUnsupported,
			request.ID,
		)
	}

	for _, tool := range request.Options.Tools {
		providerTool, ok := tool.(model.ProviderTool)
		if !ok || prepared.toolNames.ToProvider(providerTool.Name) == providerTool.Name {
			continue
		}

		return fmt.Errorf(
			"%w: Anthropic Message Batches cannot restore the custom provider-tool name %q when results are retrieved independently of the start call (request %q); use the provider's canonical tool name",
			model.ErrUnsupported,
			providerTool.Name,
			request.ID,
		)
	}

	if prepared.body.Speed != "" {
		return fmt.Errorf(
			"%w: Anthropic Message Batches do not support speed (request %q)",
			model.ErrUnsupported,
			request.ID,
		)
	}

	if prepared.body.Fallbacks != nil {
		for _, fallback := range prepared.body.Fallbacks.Chain {
			if fallback.Speed != "" {
				return fmt.Errorf(
					"%w: Anthropic Message Batches do not support fallback speed (request %q)",
					model.ErrUnsupported,
					request.ID,
				)
			}
		}
	}

	return nil
}

func (b *Batch) createBatch(ctx context.Context, prepared *preparedBatch, configuredHeaders http.Header, callHeaders http.Header) (*batchResponse, error) {
	request, err := b.apiRequest("", configuredHeaders, callHeaders)
	if err != nil {
		return nil, err
	}

	if len(prepared.betas.names) > 0 {
		request.Headers.Set("anthropic-beta", prepared.betas.header())
	}

	response, err := apicall.PostJSON(ctx, request, batchCreateRequest{
		Requests: prepared.entries,
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
	request, err := b.operationRequest(ctx, "/"+url.PathEscape(options.BatchID)+"/cancel", options.Headers)
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
		query.Set("after_id", options.Cursor)
	}

	path := ""
	if len(query) > 0 {
		path = "?" + query.Encode()
	}

	request, err := b.operationRequest(ctx, path, options.Headers)
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
		if err := batch.validate(); err != nil {
			return nil, err
		}

		status, err := batch.status()
		if err != nil {
			return nil, err
		}

		result.Batches[index] = model.BatchListItem{
			BatchID:     batch.ID,
			BatchStatus: status,
		}
	}

	if page.HasMore && page.LastID != nil {
		result.NextCursor = *page.LastID
	}

	return result, nil
}

func (b *Batch) retrieveBatch(ctx context.Context, options model.BatchOperationOptions) (*batchResponse, error) {
	request, err := b.operationRequest(ctx, "/"+url.PathEscape(options.BatchID), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, err
	}

	return decodeBatchResponse(response.Body)
}

// operationRequest describes a call to a batch endpoint with the
// provider's headers.
func (b *Batch) operationRequest(ctx context.Context, path string, callHeaders http.Header) (apicall.Request, error) {
	configuredHeaders, err := b.provider.configuredHeaders(ctx)
	if err != nil {
		return apicall.Request{}, err
	}

	return b.apiRequest(path, configuredHeaders, callHeaders)
}

func (b *Batch) apiRequest(path string, configuredHeaders http.Header, callHeaders http.Header) (apicall.Request, error) {
	baseURL, err := b.provider.resolveBaseURL()
	if err != nil {
		return apicall.Request{}, err
	}

	return apicall.Request{
		Client:     b.provider.httpClient,
		URL:        baseURL + "/messages/batches" + path,
		Headers:    apicall.MergeHeaders(configuredHeaders, callHeaders),
		ParseError: parseErrorMessage,
	}, nil
}

// batchResponse is a message batch object.
type batchResponse struct {
	ID                string              `json:"id"`
	Type              string              `json:"type"`
	ProcessingStatus  string              `json:"processing_status"`
	RequestCounts     *BatchRequestCounts `json:"request_counts"`
	CreatedAt         string              `json:"created_at"`
	ExpiresAt         string              `json:"expires_at"`
	ArchivedAt        *string             `json:"archived_at"`
	CancelInitiatedAt *string             `json:"cancel_initiated_at"`
	EndedAt           *string             `json:"ended_at"`
	ResultsURL        *string             `json:"results_url"`
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

// validate checks the members every batch object has.
func (r batchResponse) validate() error {
	if r.Type != "message_batch" {
		return fmt.Errorf("%w: batch type must be \"message_batch\", got %q", model.ErrInvalidResponse, r.Type)
	}

	if r.ID == "" || r.ProcessingStatus == "" || r.RequestCounts == nil {
		return fmt.Errorf("%w: batch response has no id, processing_status or request_counts", model.ErrInvalidResponse)
	}

	return nil
}

// status converts the batch into a normalized status.
func (r batchResponse) status() (model.BatchStatus, error) {
	createdAt, err := batchTime("created_at", r.CreatedAt)
	if err != nil {
		return model.BatchStatus{}, err
	}

	expiresAt, err := batchTime("expires_at", r.ExpiresAt)
	if err != nil {
		return model.BatchStatus{}, err
	}

	return model.BatchStatus{
		State:         batchState(r.ProcessingStatus),
		RawStatus:     r.ProcessingStatus,
		RequestCounts: r.requestCounts(),
		CreatedAt:     createdAt,
		ExpiresAt:     expiresAt,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: BatchMetadata{
				ArchivedAt:        r.ArchivedAt,
				CancelInitiatedAt: r.CancelInitiatedAt,
				EndedAt:           r.EndedAt,
				RequestCounts:     *r.RequestCounts,
				ResultsURL:        r.ResultsURL,
			},
		},
	}, nil
}

// batchTime parses an RFC 3339 timestamp of a batch object.
func batchTime(name string, value string) (time.Time, error) {
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return time.Time{}, fmt.Errorf("%w: batch %s: %w", model.ErrInvalidResponse, name, err)
	}

	return parsed, nil
}

// batchState maps the processing status. Only ended batches have results;
// every other status, known or not, is pending.
func batchState(processingStatus string) model.BatchState {
	if processingStatus == "ended" {
		return model.BatchCompleted
	}

	return model.BatchPending
}

func (r batchResponse) requestCounts() *model.BatchRequestCounts {
	counts := r.RequestCounts

	total := counts.Processing + counts.Succeeded + counts.Errored + counts.Canceled + counts.Expired
	failed := counts.Errored + counts.Canceled + counts.Expired

	return model.NormalizeBatchRequestCounts(&total, &counts.Processing, &counts.Succeeded, &failed)
}
