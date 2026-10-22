package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

const jsonResponseToolDescription = "Respond with a JSON object."

// configurableSafetyCategories receive the blocking threshold option.
var configurableSafetyCategories = []HarmCategory{
	HarmCategoryHateSpeech,
	HarmCategoryDangerousContent,
	HarmCategoryHarassment,
	HarmCategorySexuallyExplicit,
}

// preparedCall is a converted language model call.
type preparedCall struct {
	body      generateContentRequest
	warnings  []model.Warning
	toolNames toolnames.Mapping

	// headers are request headers derived from the options, such as the
	// Vertex AI request type headers.
	headers http.Header

	// generateID names sources and tool calls that have no wire ID.
	generateID func() string

	// jsonResponseToolName names the function tool that carries structured
	// output, when the model cannot combine a JSON response with tools.
	jsonResponseToolName string

	// metadataNamespaces key the output metadata.
	metadataNamespaces []string

	// streamFunctionCallArguments requests incremental function call
	// arguments when the call streams, which only Vertex AI supports.
	streamFunctionCallArguments bool
}

// requestHeaders returns the derived headers followed by the call headers.
func (c preparedCall) requestHeaders(callHeaders http.Header) http.Header {
	return apicall.MergeHeaders(c.headers, callHeaders)
}

// enableStreamedFunctionCallArguments asks a streaming call for incremental
// function call arguments when the options request them.
func (c *preparedCall) enableStreamedFunctionCallArguments() {
	if !c.streamFunctionCallArguments {
		return
	}

	var config toolConfig
	if c.body.ToolConfig != nil {
		config = *c.body.ToolConfig
	}

	var functionCalling functionCallingConfig
	if config.FunctionCallingConfig != nil {
		functionCalling = *config.FunctionCallingConfig
	}

	functionCalling.StreamFunctionCallArguments = true
	config.FunctionCallingConfig = &functionCalling
	c.body.ToolConfig = &config
}

// prepareCall converts call options. Gemini models before Gemini 3 reject a
// JSON response schema together with function tools, and Gemini 3 accepts
// the combination only with automatic tool choice, so such calls receive the
// schema as a forced JSON response tool instead.
func (m *ChatModel) prepareCall(ctx context.Context, options model.CallOptions) (preparedCall, error) {
	jsonTool, ok := m.jsonResponseTool(options)
	if !ok {
		return m.prepareRequest(ctx, options)
	}

	options.ResponseFormat = nil
	options.Tools = append(slices.Clone(options.Tools), jsonTool)
	options.ToolChoice = jsonResponseToolChoice(options.ToolChoice, jsonTool.Name)

	call, err := m.prepareRequest(ctx, options)
	if err != nil {
		return preparedCall{}, err
	}

	call.jsonResponseToolName = jsonTool.Name

	return call, nil
}

func (m *ChatModel) jsonResponseTool(options model.CallOptions) (model.FunctionTool, bool) {
	format := options.ResponseFormat
	if !isJSONSchemaFormat(format) {
		return model.FunctionTool{}, false
	}

	if !hasFunctionTool(options.Tools) || !isGeminiModel(m.modelID) {
		return model.FunctionTool{}, false
	}

	if capabilitiesOf(m.modelID).usesGemini3Features && !isForcedToolChoice(options.ToolChoice) {
		return model.FunctionTool{}, false
	}

	description := format.Description
	if description == "" {
		description = jsonResponseToolDescription
	}

	return model.FunctionTool{
		Name:        uniqueToolName(options.Tools, "json"),
		Description: description,
		InputSchema: format.Schema,
	}, true
}

func isJSONSchemaFormat(format *model.ResponseFormat) bool {
	return format != nil && format.Type == model.ResponseFormatJSON && len(format.Schema) > 0
}

func isForcedToolChoice(choice *model.ToolChoice) bool {
	if choice == nil {
		return false
	}

	return choice.Type == model.ToolChoiceRequired || choice.Type == model.ToolChoiceTool
}

func hasFunctionTool(tools []model.Tool) bool {
	for _, tool := range tools {
		if _, ok := tool.(model.FunctionTool); ok {
			return true
		}
	}

	return false
}

// uniqueToolName returns base, or base with the first numeric suffix that no
// tool uses.
func uniqueToolName(tools []model.Tool, base string) string {
	isTaken := func(name string) bool {
		return slices.ContainsFunc(tools, func(tool model.Tool) bool {
			return tool.ToolName() == name
		})
	}

	name := base
	for suffix := 1; isTaken(name); suffix++ {
		name = fmt.Sprintf("%s_%d", base, suffix)
	}

	return name
}

// jsonResponseToolChoice forces the JSON response tool unless the call
// already forces a specific tool.
func jsonResponseToolChoice(choice *model.ToolChoice, toolName string) *model.ToolChoice {
	if choice != nil && choice.Type == model.ToolChoiceTool {
		return choice
	}

	if choice != nil && choice.Type == model.ToolChoiceNone {
		return &model.ToolChoice{
			Type:     model.ToolChoiceTool,
			ToolName: toolName,
		}
	}

	return &model.ToolChoice{
		Type: model.ToolChoiceRequired,
	}
}

// prepareRequest builds the request body and collects warnings.
func (m *ChatModel) prepareRequest(ctx context.Context, options model.CallOptions) (preparedCall, error) {
	isVertex := m.provider.isVertex()

	googleOptions, err := chatOptions(options.ProviderOptions, m.provider.optionNamespaces())
	if err != nil {
		return preparedCall{}, err
	}

	warnings := m.optionWarnings(googleOptions, options.Tools)

	imageConfig := googleOptions.ImageConfig
	if !isVertex {
		var imageWarning *model.Warning

		imageConfig, imageWarning = geminiAPIImageConfig(googleOptions.ImageConfig, m.provider.name)
		if imageWarning != nil {
			warnings = append(warnings, *imageWarning)
		}
	}

	// Gemini 2.5 models reject penalties on the Gemini API only.
	frequencyPenalty := options.FrequencyPenalty
	presencePenalty := options.PresencePenalty
	if !isVertex && isGemini25Model(m.modelID) {
		frequencyPenalty, presencePenalty = nil, nil
		warnings = append(warnings, penaltyWarnings(options)...)
	}

	capabilities := capabilitiesOf(m.modelID)
	toolNames := toolnames.New(options.Tools, map[string]string{
		codeExecutionToolID: codeExecutionToolName,
	})

	functionResponseURLs := m.provider.functionResponseURLs(capabilities.usesGemini3Features)

	messages, err := m.downloadToolResultFiles(ctx, options.Prompt, functionResponseURLs)
	if err != nil {
		return preparedCall{}, err
	}

	prompt, err := convertPrompt(messages, promptSettings{
		isGemma:              isGemmaModel(m.modelID),
		isGemini3:            capabilities.usesGemini3Features,
		toolNames:            toolNames,
		metadataNamespaces:   m.provider.partMetadataNamespaces(),
		isVertex:             isVertex,
		functionResponseURLs: functionResponseURLs,
	})
	if err != nil {
		return preparedCall{}, err
	}

	warnings = append(warnings, prompt.warnings...)

	tools, err := prepareTools(options.Tools, options.ToolChoice, m.modelID, isVertex)
	if err != nil {
		return preparedCall{}, err
	}

	thinking, thinkingWarnings := thinkingConfig(options.Reasoning, googleOptions.ThinkingConfig, m.modelID)
	warnings = append(warnings, thinkingWarnings...)
	warnings = append(warnings, tools.warnings...)

	responseFormat, err := responseFormatConfig(options.ResponseFormat, googleOptions.StructuredOutputs)
	if err != nil {
		return preparedCall{}, err
	}

	body := generateContentRequest{
		GenerationConfig: generationConfig{
			MaxOutputTokens:    options.MaxOutputTokens,
			Temperature:        options.Temperature,
			TopK:               options.TopK,
			TopP:               options.TopP,
			FrequencyPenalty:   frequencyPenalty,
			PresencePenalty:    presencePenalty,
			StopSequences:      options.StopSequences,
			Seed:               options.Seed,
			ResponseMimeType:   responseFormat.mimeType,
			ResponseJSONSchema: responseFormat.schema,
			AudioTimestamp:     optional.Or(googleOptions.AudioTimestamp, false),
			ResponseModalities: googleOptions.ResponseModalities,
			ThinkingConfig:     thinking,
			MediaResolution:    googleOptions.MediaResolution,
			ImageConfig:        imageConfig,
		},
		Contents:          prompt.contents,
		SystemInstruction: prompt.systemInstruction,
		SafetySettings:    safetySettings(googleOptions),
		Tools:             tools.tools,
		ToolConfig:        requestToolConfig(tools.config, googleOptions.RetrievalConfig),
		CachedContent:     googleOptions.CachedContent,
		Labels:            googleOptions.Labels,
		ContinuationToken: googleOptions.ContinuationToken,
	}

	if !isVertex {
		body.ServiceTier = googleOptions.ServiceTier
	}

	return preparedCall{
		body:                        body,
		warnings:                    warnings,
		toolNames:                   toolNames,
		headers:                     m.requestTypeHeaders(googleOptions),
		generateID:                  m.provider.generateID,
		metadataNamespaces:          m.provider.metadataNamespaces(),
		streamFunctionCallArguments: isVertex && optional.Or(googleOptions.StreamFunctionCallArguments, false),
	}, nil
}

// downloadToolResultFiles downloads the tool result file URLs that a Vertex
// AI model cannot reference. Gemini API prompts are returned unchanged.
func (m *ChatModel) downloadToolResultFiles(ctx context.Context, prompt []model.Message, supportedURLs map[string][]*regexp.Regexp) ([]model.Message, error) {
	if !m.provider.isVertex() {
		return prompt, nil
	}

	downloads := toolResultDownloads{
		maxBytes:      m.provider.vertex.ToolResultDownloadMaxBytes,
		supportedURLs: supportedURLs,
	}

	return downloads.apply(ctx, prompt)
}

// requestTypeHeaders returns the Vertex AI pay-as-you-go headers selected by
// the options.
func (m *ChatModel) requestTypeHeaders(options ChatOptions) http.Header {
	if !m.provider.isVertex() {
		return nil
	}

	headers := http.Header{}

	if options.SharedRequestType != "" {
		headers.Set("X-Vertex-AI-LLM-Shared-Request-Type", string(options.SharedRequestType))
	}

	if options.RequestType != "" {
		headers.Set("X-Vertex-AI-LLM-Request-Type", string(options.RequestType))
	}

	return headers
}

// optionWarnings reports options that the provider's API ignores.
func (m *ChatModel) optionWarnings(options ChatOptions, tools []model.Tool) []model.Warning {
	if m.provider.isVertex() {
		return vertexOptionWarnings(options)
	}

	var warnings []model.Warning

	usesVertexRAGStore := slices.ContainsFunc(tools, func(tool model.Tool) bool {
		providerTool, ok := tool.(model.ProviderTool)

		return ok && providerTool.ID == vertexRAGStoreToolID
	})

	if usesVertexRAGStore {
		warnings = append(warnings, otherWarning(fmt.Sprintf(
			"The 'vertex_rag_store' tool is only supported with the Google Vertex provider "+
				"and might not be supported or could behave unexpectedly with the current Google provider (%s).",
			m.provider.name,
		)))
	}

	if optional.Or(options.StreamFunctionCallArguments, false) {
		warnings = append(warnings, otherWarning(fmt.Sprintf(
			"'streamFunctionCallArguments' is only supported on the Vertex AI API "+
				"and will be ignored with the current Google provider (%s). "+
				"See https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#streaming-fc",
			m.provider.name,
		)))
	}

	if options.SharedRequestType != "" || options.RequestType != "" {
		warnings = append(warnings, otherWarning(fmt.Sprintf(
			"'sharedRequestType' and 'requestType' are Vertex AI options and are ignored with the current Google provider (%s).",
			m.provider.name,
		)))
	}

	return warnings
}

// vertexOptionWarnings reports Gemini API options that Vertex AI ignores.
func vertexOptionWarnings(options ChatOptions) []model.Warning {
	if options.ServiceTier == "" {
		return nil
	}

	return []model.Warning{
		otherWarning(
			"'serviceTier' is a Gemini API option and is not supported on Vertex AI. " +
				"Use 'sharedRequestType' (and optionally 'requestType') instead. " +
				"See https://docs.cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo",
		),
	}
}

// geminiAPIImageConfig drops the image options that the Gemini API rejects.
func geminiAPIImageConfig(config *ImageConfig, providerName string) (*ImageConfig, *model.Warning) {
	if config == nil {
		return nil, nil
	}

	var dropped []string
	if config.PersonGeneration != "" {
		dropped = append(dropped, "'imageConfig.personGeneration'")
	}

	if config.ProminentPeople != "" {
		dropped = append(dropped, "'imageConfig.prominentPeople'")
	}

	if config.ImageOutputOptions != nil {
		dropped = append(dropped, "'imageConfig.imageOutputOptions'")
	}

	if len(dropped) == 0 {
		return config, nil
	}

	verb := "is a Vertex AI option and is"
	if len(dropped) > 1 {
		verb = "are Vertex AI options and are"
	}

	warning := otherWarning(fmt.Sprintf(
		"%s %s ignored with the current Google provider (%s).",
		strings.Join(dropped, ", "),
		verb,
		providerName,
	))

	return &ImageConfig{
		AspectRatio: config.AspectRatio,
		ImageSize:   config.ImageSize,
	}, &warning
}

func penaltyWarnings(options model.CallOptions) []model.Warning {
	var warnings []model.Warning

	if options.FrequencyPenalty != nil {
		warnings = append(warnings, model.UnsupportedWarning("frequencyPenalty", ""))
	}

	if options.PresencePenalty != nil {
		warnings = append(warnings, model.UnsupportedWarning("presencePenalty", ""))
	}

	return warnings
}

type responseFormatSettings struct {
	mimeType string
	schema   json.RawMessage
}

// responseFormatConfig requests JSON output. The schema is sent unless
// structured outputs are disabled.
func responseFormatConfig(format *model.ResponseFormat, structuredOutputs *bool) (responseFormatSettings, error) {
	if format == nil || format.Type != model.ResponseFormatJSON {
		return responseFormatSettings{}, nil
	}

	settings := responseFormatSettings{
		mimeType: "application/json",
	}

	sendSchema := optional.Or(structuredOutputs, true)
	if len(format.Schema) == 0 || !sendSchema {
		return settings, nil
	}

	schema, err := sanitizeResponseSchema(format.Schema)
	if err != nil {
		return responseFormatSettings{}, err
	}

	settings.schema = schema

	return settings, nil
}

// safetySettings returns the explicit settings, or the threshold applied to
// every configurable category.
func safetySettings(options ChatOptions) []SafetySetting {
	if options.SafetySettings != nil || options.Threshold == "" {
		return options.SafetySettings
	}

	settings := make([]SafetySetting, 0, len(configurableSafetyCategories))
	for _, category := range configurableSafetyCategories {
		settings = append(settings, SafetySetting{
			Category:  category,
			Threshold: options.Threshold,
		})
	}

	return settings
}

// requestToolConfig adds the retrieval config to the prepared tool config.
func requestToolConfig(prepared *toolConfig, retrievalConfig *RetrievalConfig) *toolConfig {
	if retrievalConfig == nil {
		return prepared
	}

	var config toolConfig
	if prepared != nil {
		config = *prepared
	}

	config.RetrievalConfig = retrievalConfig

	return &config
}

func otherWarning(message string) model.Warning {
	return model.Warning{
		Type:    model.WarningOther,
		Message: message,
	}
}
