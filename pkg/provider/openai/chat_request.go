package openai

import (
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// chatCall is a prepared Chat Completions request and its warnings.
type chatCall struct {
	request  chatRequest
	warnings []model.Warning
}

// prepareCall converts call options into a request body. Settings the model
// does not accept are removed with warnings.
func (m *ChatModel) prepareCall(options model.CallOptions) (chatCall, error) {
	openaiOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return chatCall{}, err
	}

	capabilities := capabilitiesFor(m.modelID)
	isReasoningModel := capabilities.isReasoningModel
	if openaiOptions.ForceReasoning != nil {
		isReasoningModel = *openaiOptions.ForceReasoning
	}

	var warnings []model.Warning

	effort, effortWarnings := m.reasoningEffort(openaiOptions, options.Reasoning, capabilities)
	warnings = append(warnings, effortWarnings...)

	if options.TopK != nil {
		warnings = append(warnings, model.UnsupportedWarning("topK", ""))
	}

	if chatHasReasoningSummary(options.ProviderOptions) {
		warnings = append(warnings, model.UnsupportedWarning(
			"reasoningSummary",
			"reasoningSummary is only supported by the Responses API, not the Chat Completions API",
		))
	}

	systemMode := chatSystemMode(openaiOptions, isReasoningModel, capabilities)

	messages, messageWarnings, err := chatPrompt(options.Prompt, systemMode)
	if err != nil {
		return chatCall{}, err
	}

	warnings = append(warnings, messageWarnings...)

	responseFormat, formatWarnings, err := chatResponseFormatFor(options.ResponseFormat, openaiOptions)
	if err != nil {
		return chatCall{}, err
	}

	warnings = append(warnings, formatWarnings...)

	request := chatRequest{
		Model: m.config.requestModelID,

		LogitBias:         openaiOptions.LogitBias,
		User:              openaiOptions.User,
		ParallelToolCalls: openaiOptions.ParallelToolCalls,

		MaxTokens:        options.MaxOutputTokens,
		Temperature:      options.Temperature,
		TopP:             options.TopP,
		FrequencyPenalty: options.FrequencyPenalty,
		PresencePenalty:  options.PresencePenalty,
		ResponseFormat:   responseFormat,
		Stop:             options.StopSequences,
		Seed:             options.Seed,
		Verbosity:        openaiOptions.TextVerbosity,

		MaxCompletionTokens:  openaiOptions.MaxCompletionTokens,
		Store:                openaiOptions.Store,
		Metadata:             openaiOptions.Metadata,
		Prediction:           openaiOptions.Prediction,
		ReasoningEffort:      effort,
		ServiceTier:          openaiOptions.ServiceTier,
		PromptCacheKey:       openaiOptions.PromptCacheKey,
		PromptCacheOptions:   openaiOptions.PromptCacheOptions,
		PromptCacheRetention: openaiOptions.PromptCacheRetention,
		SafetyIdentifier:     openaiOptions.SafetyIdentifier,

		Messages: messages,
	}

	request.setLogprobs(openaiOptions.Logprobs)

	if capabilities.supportedReasoningEfforts != nil && request.PromptCacheRetention != "" {
		request.PromptCacheRetention = ""
		warnings = append(warnings, model.UnsupportedWarning(
			"promptCacheRetention",
			"promptCacheRetention is not supported by GPT-6 and later models; use promptCacheOptions instead",
		))
	}

	if isReasoningModel {
		allowsSampling := effort == model.ReasoningNone && capabilities.supportsNonReasoningParameters
		warnings = append(warnings, request.removeReasoningUnsupported(allowsSampling)...)
	} else if chatIsSearchPreviewModel(m.modelID) {
		warnings = append(warnings, request.removeSearchPreviewUnsupported()...)
	}

	warnings = append(warnings, request.removeUnsupportedServiceTier(capabilities)...)

	tools, err := chatPrepareTools(options.Tools, options.ToolChoice)
	if err != nil {
		return chatCall{}, err
	}

	request.Tools = tools.tools
	request.ToolChoice = tools.toolChoice

	return chatCall{
		request:  request,
		warnings: append(warnings, tools.warnings...),
	}, nil
}

// reasoningEffort resolves the effort from the provider options, then the
// call's reasoning setting, and drops efforts the model does not support.
func (m *ChatModel) reasoningEffort(
	openaiOptions ChatOptions,
	reasoning model.ReasoningEffort,
	capabilities modelCapabilities,
) (model.ReasoningEffort, []model.Warning) {
	effort := openaiOptions.ReasoningEffort
	if effort == "" && reasoning != model.ReasoningProviderDefault {
		effort = reasoning
	}

	if effort == "" || capabilities.supportsEffort(effort) {
		return effort, nil
	}

	efforts := make([]string, 0, len(capabilities.supportedReasoningEfforts))
	for _, supported := range capabilities.supportedReasoningEfforts {
		efforts = append(efforts, string(supported))
	}

	return "", []model.Warning{
		model.UnsupportedWarning(
			"reasoningEffort",
			m.modelID+" only supports the following reasoning efforts: "+strings.Join(efforts, ", "),
		),
	}
}

func chatSystemMode(openaiOptions ChatOptions, isReasoningModel bool, capabilities modelCapabilities) ChatSystemMessageMode {
	if openaiOptions.SystemMessageMode != "" {
		return openaiOptions.SystemMessageMode
	}

	if isReasoningModel {
		return ChatSystemMessageDeveloper
	}

	return capabilities.systemMessageMode
}

// chatResponseFormatFor requests a JSON schema when one is given, and plain
// JSON mode otherwise.
func chatResponseFormatFor(
	format *model.ResponseFormat,
	openaiOptions ChatOptions,
) (*chatResponseFormat, []model.Warning, error) {
	if format == nil || format.Type != model.ResponseFormatJSON {
		return nil, nil, nil
	}

	if len(format.Schema) == 0 {
		return &chatResponseFormat{
			Type: chatResponseFormatJSONObject,
		}, nil, nil
	}

	schema, warnings, err := normalizeJSONSchema(format.Schema)
	if err != nil {
		return nil, nil, err
	}

	strict := true
	if openaiOptions.StrictJSONSchema != nil {
		strict = *openaiOptions.StrictJSONSchema
	}

	name := format.Name
	if name == "" {
		name = "response"
	}

	return &chatResponseFormat{
		Type: chatResponseFormatJSONSchema,
		JSONSchema: &chatJSONSchemaFormat{
			Schema:      schema,
			Strict:      strict,
			Name:        name,
			Description: format.Description,
		},
	}, warnings, nil
}

// setLogprobs requests log probabilities: true asks for the generated tokens
// only, and a number also asks for that many alternatives per token.
func (r *chatRequest) setLogprobs(setting *LogprobsSetting) {
	if !setting.requested() {
		return
	}

	enabled := true
	top := setting.Top

	r.Logprobs = &enabled
	r.TopLogprobs = &top
}

// removeReasoningUnsupported drops the settings reasoning models reject and
// moves max_tokens to max_completion_tokens. Sampling settings are kept when
// allowsSampling is set.
func (r *chatRequest) removeReasoningUnsupported(allowsSampling bool) []model.Warning {
	var warnings []model.Warning

	if !allowsSampling {
		warnings = append(warnings, r.removeReasoningSampling()...)
	}

	if r.FrequencyPenalty != nil {
		r.FrequencyPenalty = nil
		warnings = append(warnings, chatReasoningWarning("frequencyPenalty"))
	}

	if r.PresencePenalty != nil {
		r.PresencePenalty = nil
		warnings = append(warnings, chatReasoningWarning("presencePenalty"))
	}

	if r.LogitBias != nil {
		r.LogitBias = nil
		warnings = append(warnings, chatOtherWarning("logitBias is not supported for reasoning models"))
	}

	if r.TopLogprobs != nil {
		r.TopLogprobs = nil
		warnings = append(warnings, chatOtherWarning("topLogprobs is not supported for reasoning models"))
	}

	if r.MaxTokens != nil {
		if r.MaxCompletionTokens == nil {
			r.MaxCompletionTokens = r.MaxTokens
		}

		r.MaxTokens = nil
	}

	return warnings
}

func (r *chatRequest) removeReasoningSampling() []model.Warning {
	var warnings []model.Warning

	if r.Temperature != nil {
		r.Temperature = nil
		warnings = append(warnings, chatReasoningWarning("temperature"))
	}

	if r.TopP != nil {
		r.TopP = nil
		warnings = append(warnings, chatReasoningWarning("topP"))
	}

	if r.Logprobs != nil {
		r.Logprobs = nil
		warnings = append(warnings, chatOtherWarning("logprobs is not supported for reasoning models"))
	}

	return warnings
}

func (r *chatRequest) removeSearchPreviewUnsupported() []model.Warning {
	if r.Temperature == nil {
		return nil
	}

	r.Temperature = nil

	return []model.Warning{
		model.UnsupportedWarning(
			"temperature",
			"temperature is not supported for the search preview models and has been removed.",
		),
	}
}

// removeUnsupportedServiceTier drops flex and priority processing for models
// that do not offer it.
func (r *chatRequest) removeUnsupportedServiceTier(capabilities modelCapabilities) []model.Warning {
	switch r.ServiceTier {
	case ChatServiceTierFlex:
		if capabilities.supportsFlexProcessing {
			return nil
		}

		r.ServiceTier = ""

		return []model.Warning{
			model.UnsupportedWarning(
				"serviceTier",
				"flex processing is only available for o3, o4-mini, and gpt-5 models",
			),
		}

	case ChatServiceTierPriority, ChatServiceTierFast:
		if capabilities.supportsPriorityProcessing {
			return nil
		}

		r.ServiceTier = ""

		return []model.Warning{
			model.UnsupportedWarning(
				"serviceTier",
				"priority processing is only available for supported models (gpt-4, gpt-5, gpt-5-mini, o3, o4-mini) and requires Enterprise access. gpt-5-nano is not supported",
			),
		}

	default:
		return nil
	}
}

func chatIsSearchPreviewModel(modelID string) bool {
	return strings.HasPrefix(modelID, "gpt-4o-search-preview") || strings.HasPrefix(modelID, "gpt-4o-mini-search-preview")
}

func chatOtherWarning(message string) model.Warning {
	return model.Warning{
		Type:    model.WarningOther,
		Message: message,
	}
}

// chatReasoningWarning reports a setting removed for a reasoning model.
func chatReasoningWarning(feature string) model.Warning {
	return model.UnsupportedWarning(feature, feature+" is not supported for reasoning models")
}
