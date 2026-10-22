package ai

import (
	"context"
	"fmt"
	"reflect"

	"github.com/aarock1234/ai/pkg/model"
)

// batchPreparation converts the requests of one batch. Tool declarations
// are shared across requests, so it remembers them by name.
type batchPreparation struct {
	batch        model.Batch
	declarations map[string]model.Tool
}

// prepareBatchRequests converts application requests into model requests,
// in order.
func prepareBatchRequests(ctx context.Context, batch model.Batch, requests []BatchRequest) ([]model.BatchRequest, error) {
	preparation := batchPreparation{
		batch:        batch,
		declarations: make(map[string]model.Tool),
	}

	prepared := make([]model.BatchRequest, 0, len(requests))

	for _, request := range requests {
		converted, err := preparation.request(ctx, request)
		if err != nil {
			return nil, err
		}

		prepared = append(prepared, converted)

		if ctx.Err() != nil {
			return nil, context.Cause(ctx)
		}
	}

	return prepared, nil
}

func (p batchPreparation) request(ctx context.Context, request BatchRequest) (model.BatchRequest, error) {
	switch typed := request.(type) {
	case TextBatchRequest:
		return p.textRequest(ctx, typed)

	case ImageBatchRequest:
		return imageBatchRequest(typed)

	default:
		return nil, invalidBatchRequests(fmt.Sprintf("unsupported batch request type %T", request))
	}
}

// textRequest prepares a text request like one GenerateText step.
func (p batchPreparation) textRequest(ctx context.Context, request TextBatchRequest) (model.BatchRequest, error) {
	if err := validateTextBatchSettings(request); err != nil {
		return nil, err
	}

	if err := validateTools(request.Tools); err != nil {
		return nil, err
	}

	prompt, err := standardizePrompt(request.Prompt)
	if err != nil {
		return nil, err
	}

	callOptions, err := prepareCallSettings(request.CallSettings)
	if err != nil {
		return nil, err
	}

	tools := prepareTools(request.Tools, request.ToolOrder)
	if err := p.checkCompatibleTools(request.ID, tools); err != nil {
		return nil, err
	}

	messages, err := convertToModelPrompt(ctx, convertPromptOptions{
		prompt:        prompt,
		supportedURLs: p.batch.SupportedURLs(),
	})
	if err != nil {
		return nil, err
	}

	callOptions.Prompt = messages
	callOptions.Tools = tools
	callOptions.ToolChoice = prepareToolChoice(request.ToolChoice)
	callOptions.ProviderOptions = request.ProviderOptions

	return model.TextBatchRequest{
		ID:      request.ID,
		ModelID: request.Model,
		Options: callOptions,
	}, nil
}

// validateTextBatchSettings rejects the settings that apply to whole batch
// operations rather than to one request.
func validateTextBatchSettings(request TextBatchRequest) error {
	if request.MaxRetries != nil {
		return &InvalidArgumentError{
			Parameter: "requests",
			Value:     request.ID,
			Message:   "maxRetries does not apply to batch requests",
		}
	}

	if len(request.Headers) > 0 {
		return &InvalidArgumentError{
			Parameter: "requests",
			Value:     request.ID,
			Message:   "headers do not apply to batch requests; set them on the batch operation",
		}
	}

	return nil
}

// checkCompatibleTools requires every tool name to have the same
// declaration in every request, because some providers declare tools once
// per batch.
func (p batchPreparation) checkCompatibleTools(requestID string, tools []model.Tool) error {
	for _, tool := range tools {
		previous, declared := p.declarations[tool.ToolName()]
		if declared && !reflect.DeepEqual(previous, tool) {
			return &InvalidArgumentError{
				Parameter: "requests",
				Value:     requestID,
				Message:   fmt.Sprintf("tool %q must have the same definition in every batch request", tool.ToolName()),
			}
		}

		p.declarations[tool.ToolName()] = tool
	}

	return nil
}

// imageBatchRequest prepares an image request like one GenerateImage call.
func imageBatchRequest(request ImageBatchRequest) (model.BatchRequest, error) {
	files, err := imageModelFiles(request.Images)
	if err != nil {
		return nil, err
	}

	mask, err := imageModelMask(request.Mask)
	if err != nil {
		return nil, err
	}

	return model.ImageBatchRequest{
		ID:      request.ID,
		ModelID: request.Model,
		Options: model.ImageOptions{
			Prompt:          request.Prompt,
			N:               max(request.N, 1),
			Size:            request.Size,
			AspectRatio:     request.AspectRatio,
			Seed:            request.Seed,
			Files:           files,
			Mask:            mask,
			ProviderOptions: request.ProviderOptions,
		},
	}, nil
}
