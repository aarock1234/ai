package ai

import (
	"context"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// BatchItemResult is the terminal result of one batch request:
// TextBatchItemResult or ImageBatchItemResult.
type BatchItemResult interface {
	isBatchItemResult()
}

// TextBatchItemResult is the result of a text request. Result is set when
// the request succeeded; Error is set when it failed and may be set when it
// was cancelled or expired.
type TextBatchItemResult struct {
	ID     string
	Status model.BatchItemStatus

	Result *TextBatchGenerationResult
	Error  *model.BatchError

	ProviderMetadata model.ProviderMetadata
}

// TextBatchGenerationResult is the normalized output of a successful text
// request.
type TextBatchGenerationResult struct {
	// Content is the ordered output, including sources, files and parsed
	// tool calls.
	Content []ContentPart

	// Text joins the text of the output.
	Text string

	FinishReason    model.UnifiedFinishReason
	RawFinishReason string

	Usage LanguageModelUsage

	Response         model.ResponseMetadata
	ProviderMetadata model.ProviderMetadata
}

// ImageBatchItemResult is the result of an image request. Result is set
// when the request succeeded; Error is set when it failed and may be set
// when it was cancelled or expired.
type ImageBatchItemResult struct {
	ID     string
	Status model.BatchItemStatus

	Result *ImageBatchGenerationResult
	Error  *model.BatchError

	ProviderMetadata model.ProviderMetadata
}

// ImageBatchGenerationResult is the normalized output of a successful image
// request.
type ImageBatchGenerationResult struct {
	Images   []GeneratedFile
	Warnings []model.Warning
	Response model.ResponseInfo

	ProviderMetadata model.ProviderMetadata

	// Usage is nil when the provider did not report it.
	Usage *model.ImageUsage
}

func (TextBatchItemResult) isBatchItemResult()  {}
func (ImageBatchItemResult) isBatchItemResult() {}

// convertBatchItemResult normalizes one item result.
func convertBatchItemResult(ctx context.Context, item model.BatchItemResult, tools []Tool) (BatchItemResult, error) {
	switch typed := item.(type) {
	case model.TextBatchItemResult:
		return convertTextBatchItem(ctx, typed, tools)

	case model.ImageBatchItemResult:
		return convertImageBatchItem(typed)

	default:
		return nil, fmt.Errorf("%w: batch item result %T", model.ErrInvalidResponse, item)
	}
}

func convertTextBatchItem(ctx context.Context, item model.TextBatchItemResult, tools []Tool) (BatchItemResult, error) {
	converted := TextBatchItemResult{
		ID:               item.ID,
		Status:           item.Status,
		Error:            item.Error,
		ProviderMetadata: item.ProviderMetadata,
	}

	if item.Status != model.BatchItemSucceeded {
		return converted, nil
	}

	if item.Result == nil {
		return nil, fmt.Errorf("%w: succeeded batch item %q has no result", model.ErrInvalidResponse, item.ID)
	}

	result, err := convertTextBatchResult(ctx, item.Result, tools)
	if err != nil {
		return nil, err
	}

	converted.Result = result

	return converted, nil
}

// convertTextBatchResult parses the tool calls of a generation result and
// converts its content like GenerateText does, without executing tools.
func convertTextBatchResult(ctx context.Context, result *model.GenerateResult, tools []Tool) (*TextBatchGenerationResult, error) {
	var toolCalls []ToolCall
	var text strings.Builder

	for _, content := range result.Content {
		switch typed := content.(type) {
		case model.ToolCall:
			parsed, err := parseToolCall(ctx, toolCallParseRequest{
				call:  typed,
				tools: tools,
			})
			if err != nil {
				return nil, err
			}

			toolCalls = append(toolCalls, parsed)

		case model.Text:
			text.WriteString(typed.Text)
		}
	}

	content, err := convertModelContent(ctx, modelContentConversion{
		content:   result.Content,
		toolCalls: toolCalls,
		tools:     tools,
		files:     newGeneratedFileResolver(),
	})
	if err != nil {
		return nil, err
	}

	return &TextBatchGenerationResult{
		Content:          content,
		Text:             text.String(),
		FinishReason:     result.FinishReason.Unified,
		RawFinishReason:  result.FinishReason.Raw,
		Usage:            newLanguageModelUsage(result.Usage),
		Response:         result.Response.ResponseMetadata,
		ProviderMetadata: result.ProviderMetadata,
	}, nil
}

func convertImageBatchItem(item model.ImageBatchItemResult) (BatchItemResult, error) {
	converted := ImageBatchItemResult{
		ID:               item.ID,
		Status:           item.Status,
		Error:            item.Error,
		ProviderMetadata: item.ProviderMetadata,
	}

	if item.Status != model.BatchItemSucceeded {
		return converted, nil
	}

	if item.Result == nil {
		return nil, fmt.Errorf("%w: succeeded batch item %q has no result", model.ErrInvalidResponse, item.ID)
	}

	converted.Result = &ImageBatchGenerationResult{
		Images:   generatedImages(item.Result),
		Warnings: item.Result.Warnings,
		Response: model.ResponseInfo{
			ResponseMetadata: item.Result.Response.ResponseMetadata,
			Headers:          item.Result.Response.Headers,
		},
		ProviderMetadata: item.Result.ProviderMetadata,
		Usage:            item.Result.Usage,
	}

	return converted, nil
}
