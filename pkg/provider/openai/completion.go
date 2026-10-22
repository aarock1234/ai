package openai

import (
	"context"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// CompletionModel is a legacy Completions language model, such as
// gpt-3.5-turbo-instruct. The prompt is rendered as a text transcript.
type CompletionModel struct {
	modelID string
	config  config
}

func newCompletionModel(modelID string, c config) *CompletionModel {
	return &CompletionModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as "openai.completion".
func (m *CompletionModel) Provider() string {
	return m.config.modelProvider("completion")
}

// ModelID returns the model identifier.
func (m *CompletionModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that the model downloads no URLs itself.
func (m *CompletionModel) SupportedURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{}
}

// completionCall is a prepared Completions request and its warnings.
type completionCall struct {
	request  completionRequest
	warnings []model.Warning
}

// prepareCall converts call options into a request body. Tools and JSON
// output have no Completions form and produce warnings.
func (m *CompletionModel) prepareCall(options model.CallOptions) (completionCall, error) {
	openaiOptions, err := readCompletionOptions(options.ProviderOptions, m.config.name)
	if err != nil {
		return completionCall{}, err
	}

	warnings := completionWarnings(options)

	prompt, stopSequences, err := completionPrompt(options.Prompt)
	if err != nil {
		return completionCall{}, err
	}

	stop := slices.Concat(stopSequences, options.StopSequences)

	return completionCall{
		request: completionRequest{
			Model: m.config.requestModelID,

			Echo:      openaiOptions.Echo,
			LogitBias: openaiOptions.LogitBias,
			Logprobs:  openaiOptions.wireLogprobs(),
			Suffix:    openaiOptions.Suffix,
			User:      openaiOptions.User,

			MaxTokens:        options.MaxOutputTokens,
			Temperature:      options.Temperature,
			TopP:             options.TopP,
			FrequencyPenalty: options.FrequencyPenalty,
			PresencePenalty:  options.PresencePenalty,
			Seed:             options.Seed,

			Prompt: prompt,
			Stop:   stop,
		},
		warnings: warnings,
	}, nil
}

func completionWarnings(options model.CallOptions) []model.Warning {
	var warnings []model.Warning

	if options.TopK != nil {
		warnings = append(warnings, model.UnsupportedWarning("topK", ""))
	}

	if len(options.Tools) > 0 {
		warnings = append(warnings, model.UnsupportedWarning("tools", ""))
	}

	if options.ToolChoice != nil {
		warnings = append(warnings, model.UnsupportedWarning("toolChoice", ""))
	}

	if options.ResponseFormat != nil && options.ResponseFormat.Type != model.ResponseFormatText {
		warnings = append(warnings, model.UnsupportedWarning("responseFormat", "JSON response format is not supported."))
	}

	return warnings
}

// Generate performs one Completions call.
func (m *CompletionModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	call, err := m.prepareCall(options)
	if err != nil {
		return nil, err
	}

	request, err := m.config.request(ctx, "/completions", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, call.request)
	if err != nil {
		return nil, err
	}

	var decoded completionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if err := decoded.validate(); err != nil {
		return nil, err
	}

	choice := decoded.Choices[0]

	return &model.GenerateResult{
		Content: []model.Content{
			model.Text{
				Text: *choice.Text,
			},
		},
		FinishReason: model.FinishReason{
			Unified: chatFinishReason(*choice.FinishReason),
			Raw:     *choice.FinishReason,
		},
		Usage: completionConvertUsage(decoded.Usage),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: CompletionMetadata{
				Logprobs: choice.Logprobs,
			},
		},
		Warnings: call.warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: chatResponseMetadata(decoded.ID, decoded.Model, decoded.Created),
			Headers:          response.Headers,
			Body:             response.Body,
		},
	}, nil
}

// completionConvertUsage maps token usage. Completions reports no cache or
// reasoning breakdown.
func completionConvertUsage(usage *completionUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	promptTokens := chatCount(usage.PromptTokens)
	completionTokens := chatCount(usage.CompletionTokens)

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:   usage.PromptTokens,
			NoCache: &promptTokens,
		},
		OutputTokens: model.OutputTokens{
			Total: usage.CompletionTokens,
			Text:  &completionTokens,
		},
		Raw: usage.raw,
	}
}
