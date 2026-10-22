package openaicompatible

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// completionPath is the legacy completions endpoint.
const completionPath = "/completions"

// CompletionConfig configures a CompletionModel.
type CompletionConfig struct {
	// Provider is the model's provider identifier, such as
	// "name.completion".
	Provider string

	// BuildURL returns the full URL of an API path such as
	// "/chat/completions", or an error when the settings needed to build it
	// are missing.
	BuildURL URLBuilder

	// Headers returns the provider headers of one request.
	Headers func() (http.Header, error)

	// HTTPClient sends requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// ParseError reads failed response bodies. Nil uses ParseError.
	ParseError ErrorParser

	// IncludeUsage requests usage in streams with stream_options.
	IncludeUsage bool

	// SupportedURLs are the URLs the model downloads itself.
	SupportedURLs map[string][]*regexp.Regexp
}

// CompletionModel is an OpenAI-compatible legacy completions language model.
// It renders chat prompts as a plain-text transcript.
type CompletionModel struct {
	modelID string
	config  CompletionConfig
}

// NewCompletionModel returns a completion model that uses config.
func NewCompletionModel(modelID string, config CompletionConfig) *CompletionModel {
	return &CompletionModel{
		modelID: modelID,
		config:  config,
	}
}

// Provider returns the configured provider identifier.
func (m *CompletionModel) Provider() string {
	return m.config.Provider
}

// ModelID returns the model identifier.
func (m *CompletionModel) ModelID() string {
	return m.modelID
}

// SupportedURLs returns the configured URL patterns.
func (m *CompletionModel) SupportedURLs() map[string][]*regexp.Regexp {
	return m.config.SupportedURLs
}

// completionArgs is a prepared request.
type completionArgs struct {
	body     map[string]json.RawMessage
	warnings []model.Warning
}

// Generate performs a non-streaming completion.
func (m *CompletionModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	args, err := m.prepareArgs(options, false)
	if err != nil {
		return nil, err
	}

	request, err := m.apiRequest(options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, args.body)
	if err != nil {
		return nil, err
	}

	var decoded completionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if len(decoded.Choices) == 0 {
		return nil, fmt.Errorf("%w: response did not contain any choices", model.ErrInvalidResponse)
	}

	choice := decoded.Choices[0]

	var content []model.Content
	if text := valueOrEmpty(choice.Text); text != "" {
		content = append(content, model.Text{
			Text: text,
		})
	}

	return &model.GenerateResult{
		Content:      content,
		FinishReason: MapFinishReason(valueOrEmpty(choice.FinishReason)),
		Usage:        convertCompletionUsage(decoded.Usage),
		Warnings:     args.warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: responseMetadata(decoded.ID, decoded.Model, decoded.Created),
			Headers:          response.Headers,
			Body:             response.Body,
		},
	}, nil
}

// Stream performs a streaming completion.
func (m *CompletionModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	args, err := m.prepareArgs(options, true)
	if err != nil {
		return nil, err
	}

	request, err := m.apiRequest(options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, args.body)
	if err != nil {
		return nil, err
	}

	processor := &completionStream{
		includeRawChunks: options.IncludeRawChunks,
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
	}

	start := model.StreamStart{
		Warnings: args.warnings,
	}

	result := model.NewStreamResult(chatsse.Parts(ctx, response.Body, start, processor), response.Body.Close)
	result.Request = model.RequestInfo{
		Body: response.RequestBody,
	}
	result.Response = model.StreamResponseInfo{
		Headers: response.Headers,
	}

	return result, nil
}

// prepareArgs builds the request body and warnings of a call. Provider
// options are sent unchanged, and the prompt and stop sequences replace them.
func (m *CompletionModel) prepareArgs(options model.CallOptions, stream bool) (completionArgs, error) {
	name := optionsName(m.config.Provider)
	warnings := deprecatedNameWarnings(name, options.ProviderOptions)

	completionOptions, err := mergedOptions[CompletionOptions](options.ProviderOptions, name, toCamelCase(name))
	if err != nil {
		return completionArgs{}, err
	}

	warnings = append(warnings, unsupportedCompletionSettings(options)...)

	prompt, err := convertCompletionPrompt(options.Prompt)
	if err != nil {
		return completionArgs{}, err
	}

	body, err := jsonobject.Members(completionRequest{
		Model:            m.modelID,
		Echo:             completionOptions.Echo,
		LogitBias:        completionOptions.LogitBias,
		Suffix:           completionOptions.Suffix,
		User:             completionOptions.User,
		MaxTokens:        options.MaxOutputTokens,
		Temperature:      options.Temperature,
		TopP:             options.TopP,
		FrequencyPenalty: options.FrequencyPenalty,
		PresencePenalty:  options.PresencePenalty,
		Seed:             options.Seed,
	})
	if err != nil {
		return completionArgs{}, fmt.Errorf("encode request body: %w", err)
	}

	passthrough, err := mergedMembers(options.ProviderOptions, name, toCamelCase(name))
	if err != nil {
		return completionArgs{}, err
	}

	maps.Copy(body, passthrough)

	if err := setCompletionPrompt(body, prompt, options.StopSequences); err != nil {
		return completionArgs{}, err
	}

	if stream {
		if err := setStreamMembers(body, m.config.IncludeUsage); err != nil {
			return completionArgs{}, err
		}
	}

	return completionArgs{
		body:     body,
		warnings: warnings,
	}, nil
}

// unsupportedCompletionSettings warns about settings that completions ignore.
func unsupportedCompletionSettings(options model.CallOptions) []model.Warning {
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

// setCompletionPrompt sets the transcript and the stop sequences, which take
// precedence over passthrough options.
func setCompletionPrompt(body map[string]json.RawMessage, prompt completionPrompt, userStopSequences []string) error {
	promptText, err := json.Marshal(prompt.Text)
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}

	body["prompt"] = promptText
	delete(body, "stop")

	stop := slices.Concat(prompt.StopSequences, userStopSequences)
	if len(stop) == 0 {
		return nil
	}

	encodedStop, err := json.Marshal(stop)
	if err != nil {
		return fmt.Errorf("encode stop sequences: %w", err)
	}

	body["stop"] = encodedStop

	return nil
}

func (m *CompletionModel) apiRequest(callHeaders http.Header) (apicall.Request, error) {
	return newAPIRequest(apiRequestSettings{
		path:        completionPath,
		buildURL:    m.config.BuildURL,
		headers:     m.config.Headers,
		callHeaders: callHeaders,
		client:      m.config.HTTPClient,
		parseError:  m.config.ParseError,
	})
}

var _ model.LanguageModel = (*CompletionModel)(nil)
