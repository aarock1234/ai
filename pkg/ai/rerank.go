package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/internal/merge"
	"github.com/aarock1234/ai/pkg/model"
)

// RerankOptions configures Rerank. Document is string for text documents;
// any other type must encode to a JSON object.
type RerankOptions[Document any] struct {
	Model model.RerankingModel

	// Documents are the documents to rank.
	Documents []Document

	// Query is the text to rank the documents against.
	Query string

	// TopN optionally limits the number of ranked documents returned.
	TopN *int

	// MaxRetries is the number of retries per model call. Nil selects 2;
	// zero disables retries.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// OnStart is called before the model is called.
	OnStart func(ctx context.Context, event RerankStartEvent[Document])

	// OnEnd is called after the ranking succeeds.
	OnEnd func(ctx context.Context, event RerankEndEvent[Document])

	// Telemetry sends the lifecycle events of the call to telemetry
	// integrations, with the documents as any.
	Telemetry TelemetryOptions

	// LogWarnings receives the warnings of the call. Nil discards them; they
	// are always returned on the result.
	LogWarnings LogWarningsFunc
}

// RerankResult is the outcome of Rerank.
type RerankResult[Document any] struct {
	OriginalDocuments []Document

	// RerankedDocuments are the documents from most to least relevant. There
	// may be fewer than the originals when TopN is set.
	RerankedDocuments []Document

	// Ranking pairs each reranked document with its original index and
	// relevance score, from most to least relevant.
	Ranking []RankedDocument[Document]

	Warnings         []model.Warning
	ProviderMetadata model.ProviderMetadata
	Response         model.ResponseInfo
}

// RankedDocument is one entry of a ranking.
type RankedDocument[Document any] struct {
	// OriginalIndex is the document's position in RerankOptions.Documents.
	OriginalIndex int

	Score    float64
	Document Document
}

// RerankStartEvent describes a Rerank call before the model is called.
type RerankStartEvent[Document any] struct {
	CallID string

	// OperationID is "ai.rerank".
	OperationID string

	Provider        string
	ModelID         string
	Documents       []Document
	Query           string
	TopN            *int
	MaxRetries      int
	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// RerankEndEvent describes a successful Rerank call.
type RerankEndEvent[Document any] struct {
	CallID           string
	OperationID      string
	Provider         string
	ModelID          string
	Documents        []Document
	Query            string
	Ranking          []RankedDocument[Document]
	Warnings         []model.Warning
	ProviderMetadata model.ProviderMetadata
	Response         model.ResponseInfo
}

// Rerank orders documents by relevance to a query. Without documents it
// returns an empty ranking without calling the model.
func Rerank[Document any](ctx context.Context, options RerankOptions[Document]) (*RerankResult[Document], error) {
	if options.Model == nil {
		return nil, &InvalidArgumentError{
			Parameter: "model",
			Message:   "model is required",
		}
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	telemetry := newTelemetryDispatcher(options.Telemetry)

	start := RerankStartEvent[Document]{
		CallID:          id.New("call"),
		OperationID:     "ai.rerank",
		Provider:        options.Model.Provider(),
		ModelID:         options.Model.ModelID(),
		Documents:       options.Documents,
		Query:           options.Query,
		TopN:            options.TopN,
		MaxRetries:      retry.maxRetries,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	}

	if options.OnStart != nil {
		options.OnStart(ctx, start)
	}

	if telemetry.enabled() {
		telemetry.onRerankStart(ctx, start.forTelemetry())
	}

	result, err := rankDocuments(ctx, options, rerankCall{
		retry:     retry,
		telemetry: telemetry,
		callID:    start.CallID,
	})
	if err != nil {
		return nil, telemetry.onOperationError(ctx, start.CallID, err)
	}

	options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: result.Warnings,
		Provider: start.Provider,
		ModelID:  start.ModelID,
	})

	end := RerankEndEvent[Document]{
		CallID:           start.CallID,
		OperationID:      start.OperationID,
		Provider:         start.Provider,
		ModelID:          start.ModelID,
		Documents:        options.Documents,
		Query:            options.Query,
		Ranking:          result.Ranking,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Response:         result.Response,
	}

	if options.OnEnd != nil {
		options.OnEnd(ctx, end)
	}

	if telemetry.enabled() {
		telemetry.onRerankEnd(ctx, end.forTelemetry())
	}

	return result, nil
}

// rerankCall carries the retry policy and telemetry of one Rerank call to
// its model calls.
type rerankCall struct {
	retry     retryPolicy
	telemetry telemetryDispatcher
	callID    string
}

func rankDocuments[Document any](
	ctx context.Context,
	options RerankOptions[Document],
	call rerankCall,
) (*RerankResult[Document], error) {
	if len(options.Documents) == 0 {
		return &RerankResult[Document]{
			OriginalDocuments: []Document{},
			RerankedDocuments: []Document{},
			Ranking:           []RankedDocument[Document]{},
			Response:          defaultRerankResponse(options.Model.ModelID()),
		}, nil
	}

	request, err := rerankRequest(options)
	if err != nil {
		return nil, err
	}

	response, err := retryCall(ctx, call.retry, func(ctx context.Context) (*model.RerankResult, error) {
		return rerankAttempt(ctx, options, call, request)
	})
	if err != nil {
		return nil, err
	}

	if response == nil {
		return nil, fmt.Errorf("%w: reranking model returned no result", model.ErrInvalidResponse)
	}

	if err := validateRankingIndices(response.Ranking, len(options.Documents)); err != nil {
		return nil, err
	}

	ranking := make([]RankedDocument[Document], len(response.Ranking))
	reranked := make([]Document, len(response.Ranking))

	for position, ranked := range response.Ranking {
		document := options.Documents[ranked.Index]

		ranking[position] = RankedDocument[Document]{
			OriginalIndex: ranked.Index,
			Score:         ranked.RelevanceScore,
			Document:      document,
		}
		reranked[position] = document
	}

	return &RerankResult[Document]{
		OriginalDocuments: options.Documents,
		RerankedDocuments: reranked,
		Ranking:           ranking,
		Warnings:          response.Warnings,
		ProviderMetadata:  response.ProviderMetadata,
		Response:          rerankResponse(response.Response, options.Model.ModelID()),
	}, nil
}

// rerankAttempt makes one reranking model call, reporting its start and,
// when the model returned a result, its end to telemetry.
func rerankAttempt[Document any](
	ctx context.Context,
	options RerankOptions[Document],
	call rerankCall,
	request model.RerankOptions,
) (*model.RerankResult, error) {
	start := RerankingModelCallStartEvent{
		CallID:        call.callID,
		OperationID:   "ai.rerank.doRerank",
		Provider:      options.Model.Provider(),
		ModelID:       options.Model.ModelID(),
		DocumentsType: rerankDocumentsType(request),
		Query:         options.Query,
		TopN:          options.TopN,
	}

	if call.telemetry.enabled() {
		start.Documents = anyDocuments(options.Documents)
	}

	call.telemetry.onRerankingModelCallStart(ctx, start)

	response, err := options.Model.Rerank(ctx, request)
	if err != nil || response == nil {
		return response, err
	}

	call.telemetry.onRerankingModelCallEnd(ctx, RerankingModelCallEndEvent{
		CallID:        start.CallID,
		OperationID:   start.OperationID,
		Provider:      start.Provider,
		ModelID:       start.ModelID,
		DocumentsType: start.DocumentsType,
		Ranking:       response.Ranking,
	})

	return response, nil
}

// rerankDocumentsType returns "text" for text documents and "object" for
// JSON object documents.
func rerankDocumentsType(request model.RerankOptions) string {
	if request.ObjectDocuments != nil {
		return rerankDocumentsObject
	}

	return rerankDocumentsText
}

// rerankRequest sends string documents as text and other documents as JSON
// objects.
func rerankRequest[Document any](options RerankOptions[Document]) (model.RerankOptions, error) {
	request := model.RerankOptions{
		Query:           options.Query,
		TopN:            options.TopN,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	}

	if texts, ok := any(options.Documents).([]string); ok {
		request.TextDocuments = texts

		return request, nil
	}

	objects := make([]json.RawMessage, len(options.Documents))
	for index, document := range options.Documents {
		object, ok := merge.Object(document)
		if !ok {
			return model.RerankOptions{}, &InvalidArgumentError{
				Parameter: "documents",
				Value:     index,
				Message:   "documents must be strings or encode to JSON objects",
			}
		}

		encoded, err := json.Marshal(object)
		if err != nil {
			return model.RerankOptions{}, fmt.Errorf("encode document %d: %w", index, err)
		}

		objects[index] = encoded
	}

	request.ObjectDocuments = objects

	return request, nil
}

func validateRankingIndices(ranking []model.RankedDocument, documentCount int) error {
	for _, ranked := range ranking {
		if ranked.Index < 0 || ranked.Index >= documentCount {
			return fmt.Errorf(
				"%w: invalid ranking index %d: expected an integer between 0 and %d",
				model.ErrInvalidResponse,
				ranked.Index,
				documentCount-1,
			)
		}
	}

	return nil
}

// rerankResponse fills the timestamp and model ID when the provider did not
// report them.
func rerankResponse(response model.ResponseInfo, modelID string) model.ResponseInfo {
	if response.Timestamp.IsZero() {
		response.Timestamp = time.Now()
	}

	if response.ModelID == "" {
		response.ModelID = modelID
	}

	return response
}

func defaultRerankResponse(modelID string) model.ResponseInfo {
	return rerankResponse(model.ResponseInfo{}, modelID)
}
