package ai

import (
	"context"
	"fmt"
	"maps"
	"net/http"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/internal/merge"
	"github.com/aarock1234/ai/pkg/model"
)

// EmbedOptions configures Embed.
type EmbedOptions struct {
	Model model.EmbeddingModel

	// Value is the text to embed.
	Value string

	// Dimensions optionally requests a vector size. It must be positive and
	// requires provider support.
	Dimensions *int

	// MaxRetries is the number of retries per model call. Nil selects 2;
	// zero disables retries.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// OnStart is called before the model is called.
	OnStart func(ctx context.Context, event EmbedStartEvent)

	// OnEnd is called after the embedding succeeds.
	OnEnd func(ctx context.Context, event EmbedEndEvent)

	// Telemetry sends the lifecycle events of the call to telemetry
	// integrations.
	Telemetry TelemetryOptions

	// LogWarnings receives the warnings of the call. Nil discards them; they
	// are always returned on the result.
	LogWarnings LogWarningsFunc
}

// EmbedResult is the outcome of Embed.
type EmbedResult struct {
	Value     string
	Embedding []float64

	// Usage is nil when the provider does not report it.
	Usage *model.EmbeddingUsage

	Warnings         []model.Warning
	ProviderMetadata model.ProviderMetadata
	Response         model.ResponseInfo
}

// EmbedManyOptions configures EmbedMany.
type EmbedManyOptions struct {
	Model model.EmbeddingModel

	// Values are the texts to embed.
	Values []string

	// Dimensions optionally requests a vector size. It must be positive and
	// requires provider support.
	Dimensions *int

	// MaxParallelCalls limits concurrent model calls when the values are
	// split into batches and the model supports parallel calls. Zero means no
	// limit.
	MaxParallelCalls int

	// MaxRetries is the number of retries per model call. Nil selects 2;
	// zero disables retries.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// OnStart is called before the first model call.
	OnStart func(ctx context.Context, event EmbedStartEvent)

	// OnEnd is called after every batch succeeds.
	OnEnd func(ctx context.Context, event EmbedEndEvent)

	// Telemetry sends the lifecycle events of the call to telemetry
	// integrations.
	Telemetry TelemetryOptions

	// LogWarnings receives the warnings of every model call together. Nil
	// discards them; they are always returned on the result.
	LogWarnings LogWarningsFunc
}

// EmbedManyResult is the outcome of EmbedMany.
type EmbedManyResult struct {
	Values []string

	// Embeddings has one vector per value, in value order.
	Embeddings [][]float64

	// Usage sums the usage of every model call. It is nil when any call did
	// not report usage.
	Usage *model.EmbeddingUsage

	Warnings []model.Warning

	// ProviderMetadata merges the metadata of every model call per provider.
	ProviderMetadata model.ProviderMetadata

	// Responses has one entry per model call.
	Responses []model.ResponseInfo
}

// EmbedStartEvent describes an Embed or EmbedMany call before the model is
// called.
type EmbedStartEvent struct {
	CallID string

	// OperationID is "ai.embed" or "ai.embedMany".
	OperationID string

	Provider        string
	ModelID         string
	Values          []string
	Dimensions      *int
	MaxRetries      int
	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// EmbedEndEvent describes a successful Embed or EmbedMany call.
type EmbedEndEvent struct {
	CallID      string
	OperationID string
	Provider    string
	ModelID     string
	Values      []string
	Embeddings  [][]float64

	// Usage is nil when the provider did not report it.
	Usage *model.EmbeddingUsage

	Warnings         []model.Warning
	ProviderMetadata model.ProviderMetadata
	Responses        []model.ResponseInfo
}

// embeddingCall is one model call for a batch of values.
type embeddingCall struct {
	model      model.EmbeddingModel
	retry      retryPolicy
	values     []string
	dimensions *int
	headers    http.Header
	options    model.ProviderOptions

	// telemetry receives the model call events of each attempt, which
	// belong to the operation callID and are named operationID.
	telemetry   telemetryDispatcher
	callID      string
	operationID string
}

// Embed embeds one value with an embedding model.
func Embed(ctx context.Context, options EmbedOptions) (*EmbedResult, error) {
	if err := validateEmbeddingOptions(options.Model, options.Dimensions); err != nil {
		return nil, err
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	telemetry := newTelemetryDispatcher(options.Telemetry)

	start := EmbedStartEvent{
		CallID:      id.New("call"),
		OperationID: "ai.embed",
		Provider:    options.Model.Provider(),
		ModelID:     options.Model.ModelID(),
		Values: []string{
			options.Value,
		},
		Dimensions:      options.Dimensions,
		MaxRetries:      retry.maxRetries,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	}

	if options.OnStart != nil {
		options.OnStart(ctx, start)
	}

	telemetry.onEmbedStart(ctx, start)

	response, err := callEmbeddingModel(ctx, embeddingCall{
		model:       options.Model,
		retry:       retry,
		values:      start.Values,
		dimensions:  options.Dimensions,
		headers:     options.Headers,
		options:     options.ProviderOptions,
		telemetry:   telemetry,
		callID:      start.CallID,
		operationID: "ai.embed.doEmbed",
	})
	if err != nil {
		return nil, telemetry.onOperationError(ctx, start.CallID, err)
	}

	result := &EmbedResult{
		Value:            options.Value,
		Embedding:        response.Embeddings[0],
		Usage:            response.Usage,
		Warnings:         response.Warnings,
		ProviderMetadata: response.ProviderMetadata,
		Response:         response.Response,
	}

	options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: result.Warnings,
		Provider: start.Provider,
		ModelID:  start.ModelID,
	})

	end := EmbedEndEvent{
		CallID:           start.CallID,
		OperationID:      start.OperationID,
		Provider:         start.Provider,
		ModelID:          start.ModelID,
		Values:           start.Values,
		Embeddings:       response.Embeddings,
		Usage:            result.Usage,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Responses: []model.ResponseInfo{
			result.Response,
		},
	}

	if options.OnEnd != nil {
		options.OnEnd(ctx, end)
	}

	telemetry.onEmbedEnd(ctx, end)

	return result, nil
}

// EmbedMany embeds several values. Values are split into batches when the
// model limits the number of embeddings per call; batches run concurrently
// when the model supports parallel calls.
func EmbedMany(ctx context.Context, options EmbedManyOptions) (*EmbedManyResult, error) {
	if err := validateEmbeddingOptions(options.Model, options.Dimensions); err != nil {
		return nil, err
	}

	if options.MaxParallelCalls < 0 {
		return nil, &InvalidArgumentError{
			Parameter: "maxParallelCalls",
			Value:     options.MaxParallelCalls,
			Message:   "maxParallelCalls must be >= 0",
		}
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	telemetry := newTelemetryDispatcher(options.Telemetry)

	start := EmbedStartEvent{
		CallID:          id.New("call"),
		OperationID:     "ai.embedMany",
		Provider:        options.Model.Provider(),
		ModelID:         options.Model.ModelID(),
		Values:          options.Values,
		Dimensions:      options.Dimensions,
		MaxRetries:      retry.maxRetries,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	}

	if options.OnStart != nil {
		options.OnStart(ctx, start)
	}

	telemetry.onEmbedStart(ctx, start)

	responses, err := embedBatches(ctx, options, embeddingCall{
		model:       options.Model,
		retry:       retry,
		dimensions:  options.Dimensions,
		headers:     options.Headers,
		options:     options.ProviderOptions,
		telemetry:   telemetry,
		callID:      start.CallID,
		operationID: "ai.embedMany.doEmbed",
	})
	if err != nil {
		return nil, telemetry.onOperationError(ctx, start.CallID, err)
	}

	result := combineEmbeddingResponses(options.Values, responses)

	options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: result.Warnings,
		Provider: start.Provider,
		ModelID:  start.ModelID,
	})

	end := EmbedEndEvent{
		CallID:           start.CallID,
		OperationID:      start.OperationID,
		Provider:         start.Provider,
		ModelID:          start.ModelID,
		Values:           result.Values,
		Embeddings:       result.Embeddings,
		Usage:            result.Usage,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Responses:        result.Responses,
	}

	if options.OnEnd != nil {
		options.OnEnd(ctx, end)
	}

	telemetry.onEmbedEnd(ctx, end)

	return result, nil
}

func validateEmbeddingOptions(embeddingModel model.EmbeddingModel, dimensions *int) error {
	if embeddingModel == nil {
		return &InvalidArgumentError{
			Parameter: "model",
			Message:   "model is required",
		}
	}

	if dimensions != nil && *dimensions <= 0 {
		return &InvalidArgumentError{
			Parameter: "dimensions",
			Value:     *dimensions,
			Message:   "dimensions must be a positive integer",
		}
	}

	return nil
}

// embedBatches embeds the values in batches of at most the model's limit and
// returns the responses in value order. Each batch is a copy of call with
// the batch's values.
func embedBatches(ctx context.Context, options EmbedManyOptions, call embeddingCall) ([]*model.EmbedResult, error) {
	batches := splitIntoBatches(options.Values, options.Model.MaxEmbeddingsPerCall())

	parallelCalls := 1
	if options.Model.SupportsParallelCalls() {
		parallelCalls = options.MaxParallelCalls
	}

	return mapConcurrently(ctx, batches, parallelCalls, func(ctx context.Context, batch []string) (*model.EmbedResult, error) {
		batchCall := call
		batchCall.values = batch

		return callEmbeddingModel(ctx, batchCall)
	})
}

// splitIntoBatches splits values into batches of at most size values. A size
// of zero or less means one batch, which calls the model even without values;
// with a limit, no values produce no batches.
func splitIntoBatches(values []string, size int) [][]string {
	if size > 0 && len(values) == 0 {
		return nil
	}

	if size <= 0 || len(values) <= size {
		return [][]string{
			values,
		}
	}

	batches := make([][]string, 0, (len(values)+size-1)/size)
	for start := 0; start < len(values); start += size {
		end := min(start+size, len(values))
		batches = append(batches, values[start:end])
	}

	return batches
}

// callEmbeddingModel embeds one batch with retries and verifies that the
// model returned one embedding per value.
func callEmbeddingModel(ctx context.Context, call embeddingCall) (*model.EmbedResult, error) {
	response, err := retryCall(ctx, call.retry, call.attempt)
	if err != nil {
		return nil, err
	}

	if response == nil || len(response.Embeddings) != len(call.values) {
		received := 0
		if response != nil {
			received = len(response.Embeddings)
		}

		return nil, fmt.Errorf(
			"%w: expected %d embeddings, but received %d",
			model.ErrInvalidResponse,
			len(call.values),
			received,
		)
	}

	return response, nil
}

// attempt makes one embedding model call, reporting its start and, when
// the model returned a result, its end to telemetry.
func (call embeddingCall) attempt(ctx context.Context) (*model.EmbedResult, error) {
	start := EmbeddingModelCallStartEvent{
		CallID:      call.callID,
		EmbedCallID: id.New("call"),
		OperationID: call.operationID,
		Provider:    call.model.Provider(),
		ModelID:     call.model.ModelID(),
		Values:      call.values,
		Dimensions:  call.dimensions,
	}

	call.telemetry.onEmbeddingModelCallStart(ctx, start)

	response, err := call.model.Embed(ctx, model.EmbedOptions{
		Values:          call.values,
		Dimensions:      call.dimensions,
		Headers:         call.headers,
		ProviderOptions: call.options,
	})
	if err != nil || response == nil {
		return response, err
	}

	call.telemetry.onEmbeddingModelCallEnd(ctx, EmbeddingModelCallEndEvent{
		CallID:      start.CallID,
		EmbedCallID: start.EmbedCallID,
		OperationID: start.OperationID,
		Provider:    start.Provider,
		ModelID:     start.ModelID,
		Values:      start.Values,
		Embeddings:  response.Embeddings,
		Usage:       response.Usage,
	})

	return response, nil
}

func combineEmbeddingResponses(values []string, responses []*model.EmbedResult) *EmbedManyResult {
	result := &EmbedManyResult{
		Values:     values,
		Embeddings: make([][]float64, 0, len(values)),
		Usage: &model.EmbeddingUsage{
			Tokens: 0,
		},
		Responses: make([]model.ResponseInfo, 0, len(responses)),
	}

	for _, response := range responses {
		result.Embeddings = append(result.Embeddings, response.Embeddings...)
		result.Warnings = append(result.Warnings, response.Warnings...)
		result.Responses = append(result.Responses, response.Response)
		result.Usage = addEmbeddingUsage(result.Usage, response.Usage)
		result.ProviderMetadata = mergeProviderMetadataShallow(result.ProviderMetadata, response.ProviderMetadata)
	}

	return result
}

// addEmbeddingUsage sums usage; an unreported count makes the total unknown.
func addEmbeddingUsage(total, usage *model.EmbeddingUsage) *model.EmbeddingUsage {
	if total == nil || usage == nil {
		return nil
	}

	return &model.EmbeddingUsage{
		Tokens: total.Tokens + usage.Tokens,
	}
}

// mergeProviderMetadataShallow merges metadata per provider: the keys of a
// later call replace those of an earlier call within each provider object.
func mergeProviderMetadataShallow(base, next model.ProviderMetadata) model.ProviderMetadata {
	if next == nil {
		return base
	}

	merged := make(model.ProviderMetadata, len(base)+len(next))
	maps.Copy(merged, base)

	for provider, value := range next {
		combined, ok := merge.Shallow(merged[provider], value)
		if !ok {
			merged[provider] = value

			continue
		}

		merged[provider] = combined
	}

	return merged
}
