package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// BatchResults streams the result of every request of a terminal gateway
// batch job as newline-delimited JSON. The gateway rejects batches that are
// still running.
func (b *Batch) BatchResults(ctx context.Context, options model.BatchOperationOptions) (*model.BatchResultStream, error) {
	request, method, err := b.provider.modelRequest("/batch/results", options.Headers, nil)
	if err != nil {
		return nil, err
	}

	stream, err := apicall.PostJSONStream(ctx, request, batchOperationRequest{
		BatchID: options.BatchID,
	})
	if err != nil {
		return nil, batchCallError(err, method)
	}

	items := batchResultItems(stream.Body, b.maxLineBytes())

	return model.NewBatchResultStream(items, stream.Body.Close), nil
}

// maxLineBytes returns the configured line limit, or 64 MiB.
func (b *Batch) maxLineBytes() int {
	if b.provider.batchMaxLineBytes > 0 {
		return b.provider.batchMaxLineBytes
	}

	return batchDefaultMaxLineBytes
}

// batchResultItems yields the item result of every line.
func batchResultItems(body io.Reader, maxLineBytes int) iter.Seq2[model.BatchItemResult, error] {
	return func(yield func(model.BatchItemResult, error) bool) {
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64<<10), maxLineBytes)

		for scanner.Scan() {
			line := bytes.TrimSpace(scanner.Bytes())
			if len(line) == 0 {
				continue
			}

			item, err := decodeBatchItem(line)
			if !yield(item, err) || err != nil {
				return
			}
		}

		if err := scanner.Err(); err != nil {
			yield(nil, batchScanError(err, maxLineBytes))
		}
	}
}

func batchScanError(err error, maxLineBytes int) error {
	if errors.Is(err, bufio.ErrTooLong) {
		return fmt.Errorf("%w: JSON Lines response exceeded maximum line size of %d bytes", model.ErrInvalidResponse, maxLineBytes)
	}

	return fmt.Errorf("read batch results: %w", err)
}

// batchItemLine is one item result on the wire. The gateway sanitizes the
// results, so they are passed through after their shape is checked.
type batchItemLine struct {
	Type             string                `json:"type"`
	ID               string                `json:"id"`
	Status           model.BatchItemStatus `json:"status"`
	Result           *batchGenerateResult  `json:"result"`
	Error            *batchError           `json:"error"`
	ProviderMetadata wireMetadata          `json:"providerMetadata"`
}

// batchGenerateResult is a normalized generate result on the wire.
type batchGenerateResult struct {
	generateResponse

	Response *batchResponseMetadata `json:"response"`
}

type batchResponseMetadata struct {
	ID        *string `json:"id"`
	Timestamp *string `json:"timestamp"`
	ModelID   *string `json:"modelId"`
}

// decodeBatchItem decodes one line. Lines without a text type, an ID or a
// known status end the stream.
func decodeBatchItem(line []byte) (model.BatchItemResult, error) {
	var decoded batchItemLine
	if err := json.Unmarshal(line, &decoded); err != nil {
		return nil, fmt.Errorf("%w: decode batch result line: %w", model.ErrInvalidResponse, err)
	}

	if decoded.Type != "text" || decoded.ID == "" {
		return nil, fmt.Errorf("%w: batch result line must be a text item with an id", model.ErrInvalidResponse)
	}

	switch decoded.Status {
	case model.BatchItemSucceeded, model.BatchItemFailed, model.BatchItemCancelled, model.BatchItemExpired:
	default:
		return nil, fmt.Errorf("%w: unknown batch item status %q", model.ErrInvalidResponse, decoded.Status)
	}

	item := model.TextBatchItemResult{
		ID:               decoded.ID,
		Status:           decoded.Status,
		Error:            decoded.Error.batchError(),
		ProviderMetadata: decoded.ProviderMetadata.providerMetadata(),
	}

	if decoded.Status == model.BatchItemSucceeded && decoded.Result != nil {
		result, err := decoded.Result.generateResult()
		if err != nil {
			return nil, err
		}

		item.Result = result
	}

	return item, nil
}

// generateResult converts the result, parsing its ISO timestamp.
func (r batchGenerateResult) generateResult() (*model.GenerateResult, error) {
	content, err := decodeContents(r.Content)
	if err != nil {
		return nil, err
	}

	result := &model.GenerateResult{
		Content:          content,
		FinishReason:     r.FinishReason.finishReason(),
		Usage:            r.Usage.usage(),
		ProviderMetadata: r.ProviderMetadata.providerMetadata(),
		Warnings:         decodeWarnings(r.Warnings),
	}

	if r.Response == nil {
		return result, nil
	}

	result.Response.ID = stringValue(r.Response.ID)
	result.Response.ModelID = stringValue(r.Response.ModelID)

	if r.Response.Timestamp != nil {
		timestamp, err := time.Parse(time.RFC3339Nano, *r.Response.Timestamp)
		if err != nil {
			return nil, fmt.Errorf("%w: batch result timestamp: %w", model.ErrInvalidResponse, err)
		}

		result.Response.Timestamp = timestamp
	}

	return result, nil
}
