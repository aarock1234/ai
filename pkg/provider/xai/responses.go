package xai

import (
	"context"
	"fmt"
	"regexp"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

// responsesPath is the Responses API endpoint.
const responsesPath = "/responses"

// xHighReasoningModel is the model that accepts the xhigh effort.
const xHighReasoningModel = "grok-4.6"

// modelsWithoutReasoningEffort matches the grok-4.20 reasoning and
// non-reasoning models, including dated variants, which reject every
// reasoning effort.
var modelsWithoutReasoningEffort = regexp.MustCompile(`^grok-4\.20(-\d{4})?-(non-)?reasoning$`)

// supportedURLs are the URLs that xAI reads itself. Non-image documents are
// sent as file URLs instead of being downloaded.
var supportedURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
	"application/pdf": {
		regexp.MustCompile(`^https?://.*$`),
	},
	"text/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
}

// ResponsesModel is an xAI Responses API language model.
type ResponsesModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "xai.responses" by default.
func (m *ResponsesModel) Provider() string {
	return m.provider.name + ".responses"
}

// ModelID returns the model identifier.
func (m *ResponsesModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that xAI reads image, PDF and text URLs itself.
func (m *ResponsesModel) SupportedURLs() map[string][]*regexp.Regexp {
	return supportedURLs
}

// Generate performs a non-streaming Responses API call.
func (m *ResponsesModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	prepared, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(responsesPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, prepared.body)
	if err != nil {
		return nil, err
	}

	var decoded responsesResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	builder := contentBuilder{
		names:      prepared.names,
		generateID: m.provider.generateID,
	}

	for _, item := range decoded.Output {
		if err := builder.addItem(item); err != nil {
			return nil, err
		}
	}

	return &model.GenerateResult{
		Content:          builder.content,
		FinishReason:     statusFinishReason(decoded.Status, builder.hasFunctionCall),
		Usage:            convertUsage(decoded.Usage),
		ProviderMetadata: responseMetadataOf(decoded).providerMetadata(),
		Warnings:         prepared.warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: responseMetadata(decoded),
			Headers:          response.Headers,
			Body:             response.Body,
		},
	}, nil
}

// Stream performs a streaming Responses API call.
func (m *ResponsesModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	prepared, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	prepared.body.Stream = true

	request, err := m.provider.apiRequest(responsesPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, prepared.body)
	if err != nil {
		return nil, err
	}

	processor := newResponsesStream(responsesStreamSettings{
		names:            prepared.names,
		includeRawChunks: options.IncludeRawChunks,
		generateID:       m.provider.generateID,
	})

	start := model.StreamStart{
		Warnings: prepared.warnings,
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

// preparedRequest is a request body with its warnings and the names of the
// declared server-side tools.
type preparedRequest struct {
	body     responsesRequest
	warnings []model.Warning
	names    toolNames
}

// prepareRequest builds the request body and warnings of a call.
func (m *ResponsesModel) prepareRequest(options model.CallOptions) (preparedRequest, error) {
	responsesOptions, err := readResponsesOptions(options.ProviderOptions)
	if err != nil {
		return preparedRequest{}, err
	}

	var warnings []model.Warning

	if options.FrequencyPenalty != nil {
		warnings = append(warnings, model.UnsupportedWarning("frequencyPenalty", ""))
	}

	if options.PresencePenalty != nil {
		warnings = append(warnings, model.UnsupportedWarning("presencePenalty", ""))
	}

	if options.StopSequences != nil {
		warnings = append(warnings, model.UnsupportedWarning("stopSequences", ""))
	}

	input, inputWarnings, err := convertInput(options.Prompt)
	if err != nil {
		return preparedRequest{}, err
	}

	warnings = append(warnings, inputWarnings...)

	tools, err := prepareTools(options.Tools, options.ToolChoice)
	if err != nil {
		return preparedRequest{}, err
	}

	warnings = append(warnings, tools.warnings...)

	effort, reasoningWarnings := m.reasoningEffort(responsesOptions, options.Reasoning)
	warnings = append(warnings, reasoningWarnings...)

	body := responsesRequest{
		Model:              m.modelID,
		Input:              input,
		TopLogprobs:        responsesOptions.TopLogprobs,
		MaxOutputTokens:    options.MaxOutputTokens,
		Temperature:        options.Temperature,
		TopP:               options.TopP,
		TopK:               options.TopK,
		MinP:               responsesOptions.MinP,
		Seed:               options.Seed,
		Text:               textConfigOf(options.ResponseFormat),
		Include:            includeValuesOf(responsesOptions),
		PreviousResponseID: responsesOptions.PreviousResponseID,
		MaxTurns:           responsesOptions.MaxTurns,
		ParallelToolCalls:  responsesOptions.ParallelToolCalls,
		PromptCacheKey:     responsesOptions.PromptCacheKey,
		SafetyIdentifier:   responsesOptions.SafetyIdentifier,
		ServiceTier:        responsesOptions.ServiceTier,
		User:               responsesOptions.User,
		Tools:              tools.tools,
		ToolChoice:         tools.toolChoice,
	}

	if optional.Or(responsesOptions.Logprobs, false) || responsesOptions.TopLogprobs != nil {
		body.Logprobs = new(true)
	}

	if effort != "" {
		body.Reasoning = &reasoningConfig{
			Effort: effort,
		}
	}

	if responsesOptions.Store != nil && !*responsesOptions.Store {
		body.Store = new(false)
	}

	return preparedRequest{
		body:     body,
		warnings: warnings,
		names:    serverToolNames(options.Tools),
	}, nil
}

// reasoningEffort prefers the provider option and otherwise maps the call's
// reasoning level, which some grok-4.20 models reject.
func (m *ResponsesModel) reasoningEffort(options ResponsesOptions, level model.ReasoningEffort) (ReasoningEffort, []model.Warning) {
	if options.ReasoningEffort != "" || !reasoning.IsCustom(level) {
		return options.ReasoningEffort, nil
	}

	if modelsWithoutReasoningEffort.MatchString(m.modelID) {
		return "", []model.Warning{
			model.UnsupportedWarning("reasoning", fmt.Sprintf("reasoning %q is not supported by this model.", level)),
		}
	}

	if level == model.ReasoningNone {
		return ReasoningEffortNone, nil
	}

	highest := ReasoningEffortHigh
	if m.modelID == xHighReasoningModel {
		highest = ReasoningEffortXHigh
	}

	effortMap := map[model.ReasoningEffort]ReasoningEffort{
		model.ReasoningMinimal: ReasoningEffortLow,
		model.ReasoningLow:     ReasoningEffortLow,
		model.ReasoningMedium:  ReasoningEffortMedium,
		model.ReasoningHigh:    ReasoningEffortHigh,
		model.ReasoningXHigh:   highest,
		model.ReasoningMax:     highest,
	}

	effort, warnings, ok := reasoning.Effort(level, effortMap)
	if !ok {
		return "", warnings
	}

	return effort, warnings
}

// includeValuesOf adds encrypted reasoning to the requested include values
// when responses are not stored, so that reasoning can be replayed.
func includeValuesOf(options ResponsesOptions) []IncludeValue {
	include := options.Include
	if options.Store != nil && !*options.Store {
		include = append(append([]IncludeValue{}, include...), IncludeEncryptedReasoning)
	}

	return include
}

// textConfigOf requests a strict JSON schema, or a JSON object without a
// schema.
func textConfigOf(format *model.ResponseFormat) *textConfig {
	if format == nil || format.Type != model.ResponseFormatJSON {
		return nil
	}

	if format.Schema == nil {
		return &textConfig{
			Format: textFormat{
				Type: "json_object",
			},
		}
	}

	name := format.Name
	if name == "" {
		name = "response"
	}

	return &textConfig{
		Format: textFormat{
			Type:        "json_schema",
			Strict:      new(true),
			Name:        name,
			Description: format.Description,
			Schema:      format.Schema,
		},
	}
}

// statusFinishReason maps the response status. A function call always
// reports tool calls.
func statusFinishReason(status *string, hasFunctionCall bool) model.FinishReason {
	raw := optional.Or(status, "")

	unified := unifiedFinishReason(raw)
	if hasFunctionCall {
		unified = model.FinishToolCalls
	}

	return model.FinishReason{
		Unified: unified,
		Raw:     raw,
	}
}

func unifiedFinishReason(reason string) model.UnifiedFinishReason {
	switch reason {
	case "stop", "completed":
		return model.FinishStop
	case "length", "max_output_tokens":
		return model.FinishLength
	case "tool_calls", "function_call":
		return model.FinishToolCalls
	case "content_filter":
		return model.FinishContentFilter
	default:
		return model.FinishOther
	}
}

// convertUsage converts the usage. Missing usage reports zero counts.
// Cached tokens are included in the input tokens unless they exceed them.
func convertUsage(usage *responsesUsage) model.Usage {
	if usage == nil {
		return model.Usage{
			InputTokens: model.InputTokens{
				Total:      new(0),
				NoCache:    new(0),
				CacheRead:  new(0),
				CacheWrite: new(0),
			},
			OutputTokens: model.OutputTokens{
				Total:     new(0),
				Text:      new(0),
				Reasoning: new(0),
			},
		}
	}

	cacheReadTokens := 0
	if usage.InputTokensDetails != nil {
		cacheReadTokens = optional.Or(usage.InputTokensDetails.CachedTokens, 0)
	}

	reasoningTokens := 0
	if usage.OutputTokensDetails != nil {
		reasoningTokens = optional.Or(usage.OutputTokensDetails.ReasoningTokens, 0)
	}

	total := usage.InputTokens
	noCache := usage.InputTokens - cacheReadTokens
	if cacheReadTokens > usage.InputTokens {
		total = usage.InputTokens + cacheReadTokens
		noCache = usage.InputTokens
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:     new(total),
			NoCache:   new(noCache),
			CacheRead: new(cacheReadTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(usage.OutputTokens),
			Text:      new(usage.OutputTokens - reasoningTokens),
			Reasoning: new(reasoningTokens),
		},
		Raw: usage.raw,
	}
}

// responseMetadataOf collects the cost and routing members of a response.
func responseMetadataOf(response responsesResponse) ResponsesMetadata {
	metadata := ResponsesMetadata{
		ServiceTier:      optional.Or(response.ServiceTier, ""),
		PromptCacheKey:   optional.Or(response.PromptCacheKey, ""),
		SafetyIdentifier: optional.Or(response.SafetyIdentifier, ""),
	}

	if response.Usage != nil {
		metadata.CostInUSDTicks = response.Usage.CostInUSDTicks
	}

	return metadata
}

// responseMetadata converts the response identification fields.
func responseMetadata(response responsesResponse) model.ResponseMetadata {
	metadata := model.ResponseMetadata{
		ID:      optional.Or(response.ID, ""),
		ModelID: optional.Or(response.Model, ""),
	}

	if response.CreatedAt != nil {
		metadata.Timestamp = time.Unix(int64(*response.CreatedAt), 0)
	}

	return metadata
}

var _ model.LanguageModel = (*ResponsesModel)(nil)
