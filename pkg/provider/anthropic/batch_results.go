package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/jsonlines"
	"github.com/aarock1234/ai/internal/origin"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

// knownBatchBlockTypes are the content block types a batch result must
// decode. Blocks of other types are newer than this package and skipped.
var knownBatchBlockTypes = map[string]bool{
	blockAdvisorToolResult:                 true,
	blockBashCodeExecutionToolResult:       true,
	blockCodeExecutionToolResult:           true,
	blockCompaction:                        true,
	blockContainerUpload:                   true,
	blockFallback:                          true,
	blockMCPToolResult:                     true,
	blockMCPToolUse:                        true,
	blockRedactedThinking:                  true,
	blockServerToolUse:                     true,
	blockText:                              true,
	blockTextEditorCodeExecutionToolResult: true,
	blockThinking:                          true,
	blockToolSearchToolResult:              true,
	blockToolUse:                           true,
	blockWebFetchToolResult:                true,
	blockWebSearchToolResult:               true,
}

// BatchResults streams the results of an ended batch from its results
// file, which is downloaded lazily.
func (b *Batch) BatchResults(ctx context.Context, options model.BatchOperationOptions) (*model.BatchResultStream, error) {
	batch, err := b.retrieveBatch(ctx, options)
	if err != nil {
		return nil, err
	}

	if batchState(batch.ProcessingStatus) == model.BatchPending {
		return nil, fmt.Errorf("%w: Anthropic batch %q is not complete", model.ErrInvalidArgument, options.BatchID)
	}

	if batch.ArchivedAt != nil {
		return nil, fmt.Errorf("%w: Anthropic batch %q results are no longer available", model.ErrInvalidArgument, options.BatchID)
	}

	if batch.ResultsURL == nil {
		return nil, fmt.Errorf("%w: Anthropic batch %q completed without batch output", model.ErrInvalidResponse, options.BatchID)
	}

	request, err := b.resultsRequest(ctx, *batch.ResultsURL, options.Headers)
	if err != nil {
		return nil, err
	}

	reader := &batchResultReader{
		batch:   b,
		ctx:     ctx,
		request: request,
		lines:   jsonlines.NewReader(b.provider.batchMaxLineBytes),
	}

	return model.NewBatchResultStream(reader.items(), reader.lines.Close), nil
}

// resultsRequest describes the download of the results file. Credentials
// are sent only when the file is served from the API's own origin; other
// URLs must pass the safe download checks and receive no headers.
func (b *Batch) resultsRequest(ctx context.Context, resultsURL string, callHeaders http.Header) (apicall.Request, error) {
	baseURL, err := b.provider.resolveBaseURL()
	if err != nil {
		return apicall.Request{}, err
	}

	if origin.Same(resultsURL, baseURL) {
		request, err := b.operationRequest(ctx, "", callHeaders)
		if err != nil {
			return apicall.Request{}, err
		}

		request.URL = resultsURL

		return request, nil
	}

	if err := download.ValidateURL(resultsURL); err != nil {
		return apicall.Request{}, fmt.Errorf("%w: batch results URL: %w", model.ErrInvalidResponse, err)
	}

	return apicall.Request{
		Client:     b.provider.httpClient,
		URL:        resultsURL,
		ParseError: parseErrorMessage,
	}, nil
}

// batchResultReader reads the JSON Lines results file of one batch. Its
// lines reader owns the body of the file while it is read.
type batchResultReader struct {
	batch   *Batch
	ctx     context.Context
	request apicall.Request
	lines   *jsonlines.Reader
}

// items yields the item result of every line of the file.
func (r *batchResultReader) items() iter.Seq2[model.BatchItemResult, error] {
	return func(yield func(model.BatchItemResult, error) bool) {
		if err := r.open(); err != nil {
			yield(nil, err)

			return
		}

		for line, err := range r.lines.Lines() {
			if err != nil {
				yield(nil, err)

				return
			}

			item, err := r.batch.convertResultLine(line)
			if !yield(item, err) || err != nil {
				return
			}
		}
	}
}

func (r *batchResultReader) open() error {
	stream, err := apicall.GetStream(r.ctx, r.request)
	if err != nil {
		return err
	}

	return r.lines.Open(stream.Body)
}

// batchResultLine is one line of the results file.
type batchResultLine struct {
	CustomID string          `json:"custom_id"`
	Result   json.RawMessage `json:"result"`
}

// batchResult is the result of one request, discriminated by its type.
type batchResult struct {
	Type    string           `json:"type"`
	Message json.RawMessage  `json:"message"`
	Error   *batchErrorEntry `json:"error"`
}

// batchErrorEntry is the error of an errored request.
type batchErrorEntry struct {
	Type      string            `json:"type"`
	Error     *batchErrorDetail `json:"error"`
	RequestID *string           `json:"request_id"`
}

type batchErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// convertResultLine converts one line into a text item result. Lines
// without a custom ID end the stream; results that cannot be decoded
// become failed items.
func (b *Batch) convertResultLine(line []byte) (model.BatchItemResult, error) {
	var decoded batchResultLine
	if err := json.Unmarshal(line, &decoded); err != nil {
		return nil, fmt.Errorf("%w: decode batch result line: %w", model.ErrInvalidResponse, err)
	}

	if decoded.CustomID == "" {
		return nil, fmt.Errorf("%w: batch result line has no custom_id", model.ErrInvalidResponse)
	}

	var result batchResult
	if err := json.Unmarshal(decoded.Result, &result); err != nil {
		return invalidBatchItem(decoded.CustomID), nil
	}

	switch result.Type {
	case "canceled":
		return model.TextBatchItemResult{
			ID:     decoded.CustomID,
			Status: model.BatchItemCancelled,
		}, nil

	case "expired":
		return model.TextBatchItemResult{
			ID:     decoded.CustomID,
			Status: model.BatchItemExpired,
		}, nil

	case "errored":
		return erroredBatchItem(decoded.CustomID, result.Error), nil

	case "succeeded":
		generated, ok := b.convertMessage(result.Message)
		if !ok {
			return invalidBatchItem(decoded.CustomID), nil
		}

		return model.TextBatchItemResult{
			ID:     decoded.CustomID,
			Status: model.BatchItemSucceeded,
			Result: generated,
		}, nil

	default:
		return invalidBatchItem(decoded.CustomID), nil
	}
}

func erroredBatchItem(customID string, entry *batchErrorEntry) model.BatchItemResult {
	if entry == nil || entry.Type != "error" || entry.Error == nil {
		return invalidBatchItem(customID)
	}

	item := model.TextBatchItemResult{
		ID:     customID,
		Status: model.BatchItemFailed,
		Error: &model.BatchError{
			Message: entry.Error.Message,
			Type:    entry.Error.Type,
		},
	}

	if entry.RequestID != nil {
		item.ProviderMetadata = model.ProviderMetadata{
			Namespace: BatchItemMetadata{
				RequestID: *entry.RequestID,
			},
		}
	}

	return item
}

func invalidBatchItem(customID string) model.BatchItemResult {
	return model.TextBatchItemResult{
		ID:     customID,
		Status: model.BatchItemFailed,
		Error: &model.BatchError{
			Message: "Anthropic returned an invalid Message batch result.",
			Code:    "invalid_response",
		},
	}
}

// convertMessage converts a succeeded message, reporting false when it
// cannot be decoded.
func (b *Batch) convertMessage(raw json.RawMessage) (*model.GenerateResult, bool) {
	decoded, nativeUsage, ok := decodeBatchMessage(raw)
	if !ok {
		return nil, false
	}

	metadata := MessagesMetadata{
		Usage:                decoded.Usage,
		StopSequence:         decoded.StopSequence,
		StopDetails:          stopDetailsOf(decoded.StopDetails),
		InputTransformations: decoded.InputTransformations,
		SafeguardResults:     decoded.SafeguardResults,
		Iterations:           convertIterations(nativeUsage.Iterations),
		Container:            containerOf(decoded.Container),
		ContextManagement:    appliedContextManagementOf(decoded.ContextManagement),
	}

	return &model.GenerateResult{
		Content:      batchContent(decoded.Content, b.provider.generateID),
		FinishReason: finishReason(decoded.StopReason, false),
		Usage:        convertUsage(nativeUsage, decoded.Usage),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: metadata,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID:      valueOrZero(decoded.ID),
				ModelID: valueOrZero(decoded.Model),
			},
		},
	}, true
}

// decodeBatchMessage decodes a message of a batch result. Content blocks of
// types this package does not know are skipped, so that newer API output
// does not fail the whole result; known blocks must decode.
func decodeBatchMessage(raw json.RawMessage) (*messagesResponse, usage, bool) {
	var response messagesResponse
	if err := json.Unmarshal(raw, &response); err != nil || response.Type != "message" {
		return nil, usage{}, false
	}

	for _, rawBlock := range response.RawContent {
		block, err := decodeResponseBlock(rawBlock)
		if err == nil {
			response.Content = append(response.Content, block)

			continue
		}

		var header blockHeader
		if json.Unmarshal(rawBlock, &header) != nil || header.Type == "" || knownBatchBlockTypes[header.Type] {
			return nil, usage{}, false
		}
	}

	var decodedUsage usage
	if err := json.Unmarshal(response.Usage, &decodedUsage); err != nil {
		return nil, usage{}, false
	}

	return &response, decodedUsage, true
}

// batchContent converts the content of a batch result. Batch results are
// retrieved without the original request, so tool names are canonical,
// citations have no documents, and code execution calls are dynamic.
func batchContent(blocks []responseBlock, generateID func() string) []model.Content {
	converter := &contentConverter{
		toolNames:                toolnames.New(nil, providerToolNames),
		generateID:               generateID,
		mcpCalls:                 make(map[string]model.ToolCall),
		serverToolCalls:          make(map[string]string),
		markCodeExecutionDynamic: true,
	}

	hasWebSearchResults := containsWebSearchResults(blocks)

	var content []model.Content

	for _, block := range blocks {
		switch typed := block.(type) {
		case *responseText:
			content = append(content, converter.textContent(typed, hasWebSearchResults)...)

		case *responseServerToolUse:
			content = append(content, batchServerToolCall(converter, typed))

		case *responseMCPToolResult:
			result := converter.mcpToolResult(typed)
			if result.ToolName == "" {
				result.ToolName = "mcp"
			}

			content = append(content, result)

		default:
			blockContent, _ := converter.responseContent([]responseBlock{
				block,
			})

			content = append(content, blockContent...)
		}
	}

	return content
}

// batchServerToolCall converts a provider-executed call, keeping calls of
// tools this package does not know under their own name.
func batchServerToolCall(converter *contentConverter, block *responseServerToolUse) model.ToolCall {
	if call, ok := converter.serverToolCall(block); ok {
		return call
	}

	return model.ToolCall{
		ToolCallID:       block.ID,
		ToolName:         block.Name,
		Input:            compactInput(block.Input),
		ProviderExecuted: true,
		ProviderMetadata: callerMetadata(block.Caller),
	}
}
