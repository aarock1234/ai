package openresponses

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

// supportedURLs are the image URLs that endpoints download themselves.
var supportedURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
}

// effortMap maps reasoning levels to Open Responses efforts.
var effortMap = map[model.ReasoningEffort]string{
	model.ReasoningMinimal: "low",
	model.ReasoningLow:     "low",
	model.ReasoningMedium:  "medium",
	model.ReasoningHigh:    "high",
	model.ReasoningXHigh:   "xhigh",
	model.ReasoningMax:     "max",
}

// ResponsesModel is an Open Responses language model.
type ResponsesModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "name.responses".
func (m *ResponsesModel) Provider() string {
	return m.provider.name + ".responses"
}

// ModelID returns the model identifier.
func (m *ResponsesModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that endpoints download image URLs themselves.
func (m *ResponsesModel) SupportedURLs() map[string][]*regexp.Regexp {
	return supportedURLs
}

// Generate performs a non-streaming Responses API call.
func (m *ResponsesModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	body, warnings, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	request := m.provider.apiRequest(options.Headers)

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded responsesResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if err := m.responseFailure(decoded, request.URL, response); err != nil {
		return nil, err
	}

	content, hasToolCalls := m.generatedContent(decoded.Output)

	var timestamp time.Time
	if decoded.CreatedAt != nil {
		timestamp = time.Unix(int64(*decoded.CreatedAt), 0)
	}

	return &model.GenerateResult{
		Content:      content,
		FinishReason: finishReason(decoded.IncompleteDetails, hasToolCalls),
		Usage:        convertUsage(decoded.Usage),
		Warnings:     warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID:        decoded.ID,
				Timestamp: timestamp,
				ModelID:   decoded.Model,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// responseFailure reports a response error, or a response without output.
func (m *ResponsesModel) responseFailure(decoded responsesResponse, url string, response *apicall.Response) error {
	if decoded.Error != nil {
		metadata := m.provider.classify(*decoded.Error)
		status := cmp.Or(metadata.StatusCode, http.StatusBadRequest)

		return &model.APICallError{
			Message:         decoded.Error.Message,
			URL:             url,
			StatusCode:      status,
			ResponseHeaders: response.Headers,
			ResponseBody:    string(response.Body),
			IsRetryable:     optional.Or(metadata.IsRetryable, model.RetryableStatus(status)),
			Cause:           decoded.Error,
		}
	}

	if decoded.Output != nil {
		return nil
	}

	detail := decoded.Status
	if decoded.IncompleteDetails != nil && decoded.IncompleteDetails.Reason != nil {
		detail = *decoded.IncompleteDetails.Reason
	}

	message := "Responses API returned no output"
	if detail != "" {
		message = fmt.Sprintf("%s (%s)", message, detail)
	}

	return &model.APICallError{
		Message:         message,
		URL:             url,
		StatusCode:      http.StatusInternalServerError,
		ResponseHeaders: response.Headers,
		ResponseBody:    string(response.Body),
	}
}

// Stream performs a streaming Responses API call.
func (m *ResponsesModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	body, warnings, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	body.Stream = true

	response, err := apicall.PostJSONStream(ctx, m.provider.apiRequest(options.Headers), body)
	if err != nil {
		return nil, err
	}

	processor := newResponsesStream(responsesStreamSettings{
		namespace:        m.provider.name,
		includeRawChunks: options.IncludeRawChunks,
		classify:         m.provider.classify,
	})

	start := model.StreamStart{
		Warnings: warnings,
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

// prepareRequest builds the request body and warnings of a call.
func (m *ResponsesModel) prepareRequest(options model.CallOptions) (responsesRequest, []model.Warning, error) {
	var warnings []model.Warning

	if options.StopSequences != nil {
		warnings = append(warnings, model.UnsupportedWarning("stopSequences", ""))
	}

	if options.TopK != nil {
		warnings = append(warnings, model.UnsupportedWarning("topK", ""))
	}

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning("seed", ""))
	}

	settings := inputSettings{
		namespace:           m.provider.name,
		strictResponseInput: m.provider.strictResponseInput,
		customToolID:        m.provider.customToolID,
		providerTools:       providerToolsByName(options.Tools),
	}

	input, err := convertInput(options.Prompt, settings)
	if err != nil {
		return responsesRequest{}, nil, err
	}

	warnings = append(warnings, input.warnings...)

	tools, toolWarnings, err := m.prepareTools(options.Tools)
	if err != nil {
		return responsesRequest{}, nil, err
	}

	warnings = append(warnings, toolWarnings...)

	format, formatWarnings := m.textFormat(options.ResponseFormat)
	warnings = append(warnings, formatWarnings...)

	responsesOptions, err := readResponsesOptions(options.ProviderOptions, m.provider.name)
	if err != nil {
		return responsesRequest{}, nil, err
	}

	reasoningConfig, reasoningWarnings := reasoningParamOf(responsesOptions, options.Reasoning)
	warnings = append(warnings, reasoningWarnings...)

	body := responsesRequest{
		Model:            m.modelID,
		Input:            input.items,
		Instructions:     input.instructions,
		MaxOutputTokens:  options.MaxOutputTokens,
		Temperature:      options.Temperature,
		TopP:             options.TopP,
		PresencePenalty:  options.PresencePenalty,
		FrequencyPenalty: options.FrequencyPenalty,
		Reasoning:        reasoningConfig,
		Tools:            tools,
		ToolChoice:       toolChoiceOf(options.ToolChoice, settings),
	}

	if format != nil {
		body.Text = &textParam{
			Format: *format,
		}
	}

	return body, warnings, nil
}

func providerToolsByName(tools []model.Tool) map[string]model.ProviderTool {
	byName := make(map[string]model.ProviderTool)
	for _, tool := range tools {
		if providerTool, ok := tool.(model.ProviderTool); ok {
			byName[providerTool.Name] = providerTool
		}
	}

	return byName
}

// prepareTools converts function tools and custom tools. Other provider
// tools are not supported.
func (m *ResponsesModel) prepareTools(tools []model.Tool) ([]requestTool, []model.Warning, error) {
	var converted []requestTool
	var warnings []model.Warning

	for _, tool := range tools {
		switch tool := tool.(type) {
		case model.FunctionTool:
			converted = append(converted, requestTool{
				Type:        "function",
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
				Strict:      tool.Strict,
			})

		case model.ProviderTool:
			if tool.ID != m.provider.customToolID {
				warnings = append(warnings, model.UnsupportedWarning("provider-defined tool "+tool.ID, ""))

				continue
			}

			customTool, err := customRequestTool(tool)
			if err != nil {
				return nil, nil, err
			}

			converted = append(converted, customTool)
		}
	}

	return converted, warnings, nil
}

// customRequestTool converts a custom tool. Formats other than text or a
// complete grammar are omitted.
func customRequestTool(tool model.ProviderTool) (requestTool, error) {
	var args CustomToolArgs
	if len(tool.Args) > 0 {
		if err := json.Unmarshal(tool.Args, &args); err != nil {
			return requestTool{}, fmt.Errorf("%w: arguments of tool %s: %w", model.ErrInvalidArgument, tool.ID, err)
		}
	}

	return requestTool{
		Type:        "custom",
		Name:        tool.Name,
		Description: args.Description,
		Format:      customToolFormat(args.Format),
	}, nil
}

func customToolFormat(format *CustomToolFormat) *CustomToolFormat {
	if format == nil {
		return nil
	}

	switch format.Type {
	case CustomToolFormatGrammar:
		if format.Syntax != GrammarRegex && format.Syntax != GrammarLark {
			return nil
		}

		return &CustomToolFormat{
			Type:       CustomToolFormatGrammar,
			Syntax:     format.Syntax,
			Definition: format.Definition,
		}

	case CustomToolFormatText:
		return &CustomToolFormat{
			Type: CustomToolFormatText,
		}

	default:
		return nil
	}
}

// toolChoiceOf names a function or custom tool, or passes the mode. A
// choice of another provider tool is omitted.
func toolChoiceOf(choice *model.ToolChoice, settings inputSettings) *toolChoiceParam {
	if choice == nil {
		return nil
	}

	if choice.Type != model.ToolChoiceTool {
		return &toolChoiceParam{
			mode: string(choice.Type),
		}
	}

	if settings.isCustomTool(choice.ToolName) {
		return &toolChoiceParam{
			toolType: "custom",
			name:     choice.ToolName,
		}
	}

	if _, isProviderTool := settings.providerTools[choice.ToolName]; isProviderTool {
		return nil
	}

	return &toolChoiceParam{
		toolType: "function",
		name:     choice.ToolName,
	}
}

// textFormat requests a strict JSON schema, or a JSON object without a
// schema, unless structured outputs are disabled.
func (m *ResponsesModel) textFormat(format *model.ResponseFormat) (*textFormat, []model.Warning) {
	if format == nil || format.Type != model.ResponseFormatJSON {
		return nil, nil
	}

	if !m.provider.structuredOutputs {
		return nil, []model.Warning{
			model.UnsupportedWarning("responseFormat", ""),
		}
	}

	if format.Schema == nil {
		return &textFormat{
			Type: "json_object",
		}, nil
	}

	name := format.Name
	if name == "" {
		name = "response"
	}

	return &textFormat{
		Type:        "json_schema",
		Name:        name,
		Description: format.Description,
		Schema:      format.Schema,
		Strict:      new(true),
	}, nil
}

// reasoningParamOf prefers the provider-native effort and otherwise maps
// the call's reasoning level.
func reasoningParamOf(options ResponsesOptions, level model.ReasoningEffort) (*reasoningParam, []model.Warning) {
	effort := options.ReasoningEffort

	var warnings []model.Warning

	if effort == nil && reasoning.IsCustom(level) {
		effort, warnings = levelEffort(level)
	}

	if effort == nil && options.ReasoningSummary == nil {
		return nil, warnings
	}

	return &reasoningParam{
		Effort:  effort,
		Summary: options.ReasoningSummary,
	}, warnings
}

// levelEffort maps a reasoning level to an Open Responses effort. "none" is
// sent as is; an unmappable level has no effort.
func levelEffort(level model.ReasoningEffort) (*string, []model.Warning) {
	if level == model.ReasoningNone {
		return new("none"), nil
	}

	mapped, warnings, ok := reasoning.Effort(level, effortMap)
	if !ok {
		return nil, warnings
	}

	return &mapped, warnings
}

// generatedContent converts the output items and reports whether any is a
// tool call.
func (m *ResponsesModel) generatedContent(items []outputItem) ([]model.Content, bool) {
	var content []model.Content
	hasToolCalls := false
	namespace := m.provider.name

	for _, item := range items {
		switch {
		case item.Reasoning != nil:
			content = append(content, reasoningContent(*item.Reasoning, namespace)...)

		case item.Message != nil:
			for _, part := range item.Message.Content {
				content = append(content, model.Text{
					Text:             part.Text,
					ProviderMetadata: textMetadata(namespace, item.Message.ID, outputAnnotations(part.Annotations)),
				})
			}

		case item.FunctionCall != nil:
			hasToolCalls = true
			content = append(content, model.ToolCall{
				ToolCallID:       item.FunctionCall.CallID,
				ToolName:         item.FunctionCall.Name,
				Input:            item.FunctionCall.Arguments,
				ProviderMetadata: itemMetadata(namespace, item.FunctionCall.ID),
			})

		case item.CustomToolCall != nil:
			hasToolCalls = true
			content = append(content, model.ToolCall{
				ToolCallID:       item.CustomToolCall.CallID,
				ToolName:         item.CustomToolCall.Name,
				Input:            string(jsonobject.EncodeString(item.CustomToolCall.Input)),
				ProviderMetadata: itemMetadata(namespace, item.CustomToolCall.ID),
			})
		}
	}

	return content, hasToolCalls
}

// reasoningContent returns one reasoning part per content part, or the
// joined summary when the item has no content.
func reasoningContent(item reasoningOutput, namespace string) []model.Content {
	if len(item.Content) == 0 {
		var summary string
		for _, part := range item.Summary {
			summary += part.Text
		}

		return []model.Content{
			model.Reasoning{
				Text:             summary,
				ProviderMetadata: reasoningMetadata(namespace, item, item.Content),
			},
		}
	}

	content := make([]model.Content, 0, len(item.Content))
	for _, part := range item.Content {
		content = append(content, model.Reasoning{
			Text: part.Text,
			ProviderMetadata: reasoningMetadata(namespace, item, []textContent{
				part,
			}),
		})
	}

	return content
}

func reasoningMetadata(namespace string, item reasoningOutput, content []textContent) model.ProviderMetadata {
	metadata := ReasoningMetadata{
		ItemID:                    item.ID,
		ReasoningSummary:          make([]SummaryText, 0, len(item.Summary)),
		ReasoningEncryptedContent: optional.Or(item.EncryptedContent, ""),
	}

	for _, part := range item.Summary {
		metadata.ReasoningSummary = append(metadata.ReasoningSummary, SummaryText{
			Type: summaryTextType,
			Text: part.Text,
		})
	}

	if content != nil {
		reasoningTexts := make([]ReasoningText, 0, len(content))
		for _, part := range content {
			reasoningTexts = append(reasoningTexts, ReasoningText{
				Type: reasoningTextType,
				Text: part.Text,
			})
		}

		metadata.ReasoningContent = &reasoningTexts
	}

	return model.ProviderMetadata{
		namespace: metadata,
	}
}

func textMetadata(namespace string, itemID string, annotations []URLCitation) model.ProviderMetadata {
	return model.ProviderMetadata{
		namespace: TextMetadata{
			ItemID:      itemID,
			Annotations: annotations,
		},
	}
}

func itemMetadata(namespace string, itemID string) model.ProviderMetadata {
	return model.ProviderMetadata{
		namespace: ItemMetadata{
			ItemID: itemID,
		},
	}
}

// outputAnnotations returns the annotations when all of them are complete
// URL citations.
func outputAnnotations(raw json.RawMessage) []URLCitation {
	citations, ok := validCitations(raw)
	if !ok || len(citations) == 0 {
		return nil
	}

	return citations
}

// finishReason maps the incomplete reason.
func finishReason(details *incompleteDetails, hasToolCalls bool) model.FinishReason {
	var reason *string
	if details != nil {
		reason = details.Reason
	}

	return model.FinishReason{
		Unified: unifiedFinishReason(reason, hasToolCalls),
		Raw:     optional.Or(reason, ""),
	}
}

func unifiedFinishReason(reason *string, hasToolCalls bool) model.UnifiedFinishReason {
	switch {
	case reason == nil && hasToolCalls:
		return model.FinishToolCalls
	case reason == nil:
		return model.FinishStop
	case *reason == "max_output_tokens":
		return model.FinishLength
	case *reason == "content_filter":
		return model.FinishContentFilter
	case hasToolCalls:
		return model.FinishToolCalls
	default:
		return model.FinishOther
	}
}

// convertUsage converts the usage. The uncached input and text output
// counts are always reported.
func convertUsage(usage *responsesUsage) model.Usage {
	var converted model.Usage

	var cacheRead, cacheWrite, reasoningTokens *int
	if usage != nil {
		converted.InputTokens.Total = usage.InputTokens
		converted.OutputTokens.Total = usage.OutputTokens
		converted.Raw = usage.raw

		if usage.InputTokensDetails != nil {
			cacheRead = usage.InputTokensDetails.CachedTokens
			cacheWrite = usage.InputTokensDetails.CacheWriteTokens
		}

		if usage.OutputTokensDetails != nil {
			reasoningTokens = usage.OutputTokensDetails.ReasoningTokens
		}
	}

	converted.InputTokens.CacheRead = cacheRead
	converted.InputTokens.CacheWrite = cacheWrite
	converted.InputTokens.NoCache = new(
		optional.Or(converted.InputTokens.Total, 0) - optional.Or(cacheRead, 0) - optional.Or(cacheWrite, 0),
	)
	converted.OutputTokens.Reasoning = reasoningTokens
	converted.OutputTokens.Text = new(optional.Or(converted.OutputTokens.Total, 0) - optional.Or(reasoningTokens, 0))

	return converted
}

var _ model.LanguageModel = (*ResponsesModel)(nil)
