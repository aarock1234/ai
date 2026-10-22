package amazonbedrock

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

const (
	// jsonToolName is the synthetic tool that carries JSON output.
	jsonToolName = "json"

	// defaultThinkingMaxTokens is added to the thinking budget when the
	// call sets no max output tokens.
	defaultThinkingMaxTokens = 4096

	// stopSequenceFieldPath requests the stop sequence of Claude models.
	stopSequenceFieldPath = "/delta/stop_sequence"
)

// preparedRequest is a Converse request body with the warnings of its
// preparation and the settings needed to read its response.
type preparedRequest struct {
	body     json.RawMessage
	warnings []model.Warning

	// usesJSONInstruction reports that JSON output was requested through
	// the system prompt, so text is trimmed to the JSON object.
	usesJSONInstruction bool

	// usesJSONResponseTool reports that JSON output arrives as a call of
	// the "json" tool.
	usesJSONResponseTool bool
}

// requestState is the request under construction.
type requestState struct {
	modelID  string
	family   ModelFamily
	options  model.CallOptions
	read     readOptions
	warnings []model.Warning

	capabilities   anthropic.ModelCapabilities
	anthropicModel bool
	openAIModel    bool
	gptOSSModel    bool

	sampling samplingSettings

	// additionalFields are the additional model request fields.
	additionalFields jsonobject.Object
}

// samplingSettings are the sampling settings that survive model
// restrictions.
type samplingSettings struct {
	temperature *float64
	topP        *float64
	topK        *int
}

// prepareRequest converts call options into a Converse request body.
func (m *ChatModel) prepareRequest(options model.CallOptions) (*preparedRequest, error) {
	read, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	state := &requestState{
		modelID:          m.modelID,
		family:           m.family,
		options:          options,
		read:             read,
		capabilities:     anthropic.CapabilitiesOf(m.modelID),
		additionalFields: read.bedrock.AdditionalModelRequestFields,
	}

	state.identifyModel()
	state.warnUnsupportedSettings()
	state.resolveSampling()
	state.warnUnsupportedResponseFormat()

	state.read.bedrock.ReasoningConfig = resolveReasoningConfig(
		options.Reasoning,
		state.read.bedrock.ReasoningConfig,
		reasoningTarget{
			modelID:        m.modelID,
			anthropicModel: state.anthropicModel,
			openAIModel:    state.openAIModel,
		},
		&state.warnings,
	)

	output := state.resolveStructuredOutput()

	tools, err := state.prepareTools(output)
	if err != nil {
		return nil, err
	}

	inference, err := state.applyReasoning()
	if err != nil {
		return nil, err
	}

	if err := state.applyStructuredOutput(output); err != nil {
		return nil, err
	}

	state.restrictInference(&inference)

	prompt, err := state.prompt(tools, output)
	if err != nil {
		return nil, err
	}

	body, err := state.encode(prompt, inference, tools)
	if err != nil {
		return nil, err
	}

	return &preparedRequest{
		body:                 body,
		warnings:             state.warnings,
		usesJSONInstruction:  output.useJSONInstruction,
		usesJSONResponseTool: output.jsonResponseTool != nil,
	}, nil
}

func (s *requestState) warn(feature string, details string) {
	s.warnings = append(s.warnings, model.UnsupportedWarning(feature, details))
}

func (s *requestState) warnUnsupportedSettings() {
	if s.options.FrequencyPenalty != nil {
		s.warn("frequencyPenalty", "")
	}

	if s.options.PresencePenalty != nil {
		s.warn("presencePenalty", "")
	}

	if s.options.Seed != nil {
		s.warn("seed", "")
	}
}

// resolveSampling drops the sampling settings of models that reject them
// and clamps the temperature to Bedrock's range, except for OpenAI models
// other than gpt-oss.
func (s *requestState) resolveSampling() {
	s.sampling = samplingSettings{
		temperature: s.options.Temperature,
		topP:        s.options.TopP,
		topK:        s.options.TopK,
	}

	if s.capabilities.RejectsSamplingParameters {
		s.dropSampling()
	}

	clamps := !s.openAIModel || s.gptOSSModel

	temperature := s.sampling.temperature
	if !clamps || temperature == nil {
		return
	}

	if *temperature > 1 {
		s.warn("temperature", fmt.Sprintf("%s exceeds bedrock maximum of 1.0. clamped to 1.0", formatNumber(*temperature)))
		s.sampling.temperature = new(1.0)
	} else if *temperature < 0 {
		s.warn("temperature", fmt.Sprintf("%s is below bedrock minimum of 0. clamped to 0", formatNumber(*temperature)))
		s.sampling.temperature = new(0.0)
	}
}

func (s *requestState) warnUnsupportedResponseFormat() {
	format := s.options.ResponseFormat
	if format != nil && format.Type != model.ResponseFormatText && format.Type != model.ResponseFormatJSON {
		s.warn("responseFormat", "Only text and json response formats are supported.")
	}
}

func (s *requestState) dropSampling() {
	if s.sampling.temperature != nil {
		s.warn("temperature", fmt.Sprintf("temperature is not supported by %s and will be ignored", s.modelID))
		s.sampling.temperature = nil
	}

	if s.sampling.topK != nil {
		s.warn("topK", fmt.Sprintf("topK is not supported by %s and will be ignored", s.modelID))
		s.sampling.topK = nil
	}

	if s.sampling.topP != nil {
		s.warn("topP", fmt.Sprintf("topP is not supported by %s and will be ignored", s.modelID))
		s.sampling.topP = nil
	}
}

func (s *requestState) identifyModel() {
	openAI, isOpenAI := openAIModel(s.modelID)

	s.openAIModel = isOpenAI
	s.gptOSSModel = isOpenAI && strings.HasPrefix(openAI, "openai.gpt-oss-")

	var budget *int
	if config := s.read.bedrock.ReasoningConfig; config != nil {
		budget = config.BudgetTokens
	}

	s.anthropicModel = isAnthropicModel(s.modelID, s.family, budget)
}

// structuredOutput is the resolved strategy for JSON output with a schema.
type structuredOutput struct {
	// nativeSchema is sent as output_config.format.
	nativeSchema json.RawMessage

	// useJSONInstruction asks for JSON in the system prompt.
	useJSONInstruction bool

	// jsonResponseTool carries JSON output as a forced tool call.
	jsonResponseTool *model.FunctionTool
}

func (s *requestState) resolveStructuredOutput() structuredOutput {
	mode := StructuredOutputAuto
	if configured := s.read.bedrock.StructuredOutputMode; configured != nil {
		mode = *configured
	} else if configured := s.read.anthropic.StructuredOutputMode; configured != nil {
		mode = *configured
	}

	if mode == StructuredOutputJSONTool {
		s.removeOutputFormat()
	}

	format := s.options.ResponseFormat
	if format == nil || format.Type != model.ResponseFormatJSON || len(format.Schema) == 0 {
		return structuredOutput{}
	}

	config := s.read.bedrock.ReasoningConfig
	thinking := isReasoningType(config, ReasoningEnabled) || isReasoningType(config, ReasoningAdaptive)

	nativeSupported := bedrock.SupportsNativeStructuredOutput(s.modelID) &&
		(s.capabilities.SupportsStructuredOutput || thinking || s.family == ModelFamilyAnthropic)

	if s.anthropicModel && (mode == StructuredOutputFormat || (mode == StructuredOutputAuto && nativeSupported)) {
		return structuredOutput{
			nativeSchema: anthropic.SanitizeJSONSchema(format.Schema),
		}
	}

	if s.anthropicModel && s.prefersJSONInstruction(mode) {
		return structuredOutput{
			useJSONInstruction: true,
		}
	}

	return structuredOutput{
		jsonResponseTool: &model.FunctionTool{
			Name:        jsonToolName,
			Description: "Respond with a JSON object.",
			InputSchema: format.Schema,
		},
	}
}

// prefersJSONInstruction reports whether a Claude model should be asked for
// JSON in the prompt because the JSON tool would be rejected or unreliable.
func (s *requestState) prefersJSONInstruction(mode StructuredOutputMode) bool {
	if s.capabilities.RejectsForcedToolUse {
		return true
	}

	unsupportedClaude := strings.Contains(s.modelID, "claude-") &&
		(!s.capabilities.IsKnownModel || !s.capabilities.SupportsStructuredOutput) &&
		!bedrock.SupportsNativeStructuredOutput(s.modelID) &&
		mode != StructuredOutputJSONTool

	if unsupportedClaude {
		return true
	}

	return mode != StructuredOutputJSONTool && !bedrock.SupportsStrictTools(s.modelID) && len(s.options.Tools) > 0
}

// removeOutputFormat removes output_config.format from the additional
// fields, and output_config when nothing else remains in it.
func (s *requestState) removeOutputFormat() {
	outputConfig, found := s.additionalFields.Get("output_config")
	if !found {
		return
	}

	object, err := jsonobject.Parse(outputConfig)
	if err != nil {
		return
	}

	object = object.Without("format")
	if len(object) == 0 {
		s.additionalFields = s.additionalFields.Without("output_config")

		return
	}

	encoded, err := object.Encode()
	if err != nil {
		return
	}

	s.additionalFields = s.additionalFields.With("output_config", encoded)
}

func (s *requestState) prepareTools(output structuredOutput) (preparedTools, error) {
	tools := s.options.Tools
	choice := s.options.ToolChoice

	if output.jsonResponseTool != nil {
		tools = append(append([]model.Tool{}, tools...), *output.jsonResponseTool)
		choice = &model.ToolChoice{
			Type: model.ToolChoiceRequired,
		}
	}

	var budget *int
	if config := s.read.bedrock.ReasoningConfig; config != nil {
		budget = config.BudgetTokens
	}

	prepared, err := prepareTools(toolPreparation{
		tools:                  tools,
		toolChoice:             choice,
		modelID:                s.modelID,
		family:                 s.family,
		reasoningBudgetTokens:  budget,
		disableParallelToolUse: s.read.anthropic.DisableParallelToolUse,
		rejectsForcedToolUse:   s.capabilities.RejectsForcedToolUse,
	})
	if err != nil {
		return preparedTools{}, err
	}

	s.warnings = append(s.warnings, prepared.warnings...)

	if prepared.anthropicToolChoice != nil {
		s.additionalFields = s.additionalFields.With("tool_choice", prepared.anthropicToolChoice)
	}

	if len(prepared.betas) > 0 || s.read.bedrock.AnthropicBeta != nil {
		betas := append(append([]string{}, s.read.bedrock.AnthropicBeta...), prepared.betas...)
		s.additionalFields = s.additionalFields.With("anthropic_beta", mustEncode(betas))
	}

	return prepared, nil
}

// applyReasoning sends the reasoning configuration in the form the model
// family expects and returns the inference configuration.
func (s *requestState) applyReasoning() (inferenceConfig, error) {
	config := s.read.bedrock.ReasoningConfig
	if config == nil {
		config = &ReasoningConfig{}
	}

	inference := inferenceConfig{
		MaxTokens:     s.options.MaxOutputTokens,
		Temperature:   s.sampling.temperature,
		TopP:          s.sampling.topP,
		TopK:          s.sampling.topK,
		StopSequences: s.options.StopSequences,
	}

	var budget *int
	if isReasoningType(config, ReasoningEnabled) {
		budget = config.BudgetTokens
	}

	switch {
	case s.anthropicModel && s.thinkingEnabled():
		if err := s.applyClaudeThinking(config, budget, &inference); err != nil {
			return inferenceConfig{}, err
		}

	case !s.anthropicModel:
		if config.BudgetTokens != nil {
			s.warn("budgetTokens", "budgetTokens applies only to Anthropic models on Bedrock and will be ignored for this model.")
		}

		if isReasoningType(config, ReasoningAdaptive) {
			s.warn("adaptive thinking", "adaptive thinking type applies only to Anthropic models on Bedrock.")
		}
	}

	if config.MaxReasoningEffort != nil {
		if err := s.applyReasoningEffort(config, budget); err != nil {
			return inferenceConfig{}, err
		}
	}

	highEffort := config.MaxReasoningEffort != nil &&
		(*config.MaxReasoningEffort == ReasoningEffortHigh ||
			*config.MaxReasoningEffort == ReasoningEffortXHigh ||
			*config.MaxReasoningEffort == ReasoningEffortMax)

	if isNova2ReasoningModel(s.modelID) && isReasoningType(config, ReasoningEnabled) && highEffort && inference.MaxTokens != nil {
		inference.MaxTokens = nil
		s.warn(
			"maxOutputTokens",
			fmt.Sprintf("maxOutputTokens is not supported by %s when high reasoning is enabled and will be ignored", s.modelID),
		)
	}

	return inference, nil
}

func (s *requestState) thinkingEnabled() bool {
	config := s.read.bedrock.ReasoningConfig

	return isReasoningType(config, ReasoningEnabled) || isReasoningType(config, ReasoningAdaptive)
}

// applyClaudeThinking adds Claude's thinking field. A thinking budget is
// added to the max tokens.
func (s *requestState) applyClaudeThinking(config *ReasoningConfig, budget *int, inference *inferenceConfig) error {
	if budget != nil {
		maxTokens := *budget + defaultThinkingMaxTokens
		if inference.MaxTokens != nil {
			maxTokens = *inference.MaxTokens + *budget
		}

		inference.MaxTokens = &maxTokens

		thinking := jsonobject.Object{
			{
				Key:   "type",
				Value: jsonobject.EncodeString("enabled"),
			},
			{
				Key:   "budget_tokens",
				Value: mustEncode(*budget),
			},
		}

		return s.setObjectField("thinking", thinking)
	}

	if !isReasoningType(config, ReasoningAdaptive) {
		return nil
	}

	thinking := jsonobject.Object{
		{
			Key:   "type",
			Value: jsonobject.EncodeString("adaptive"),
		},
	}

	if config.Display != nil {
		thinking = thinking.With("display", mustEncode(*config.Display))
	}

	return s.setObjectField("thinking", thinking)
}

// applyReasoningEffort sends the maximum reasoning effort: as
// output_config.effort for Claude, as reasoning_effort or reasoning.effort
// for OpenAI models, and as reasoningConfig for others such as Nova 2.
func (s *requestState) applyReasoningEffort(config *ReasoningConfig, budget *int) error {
	effort := mustEncode(*config.MaxReasoningEffort)

	switch {
	case s.anthropicModel:
		return s.setNestedField("output_config", "effort", effort)

	case s.gptOSSModel:
		s.additionalFields = s.additionalFields.With("reasoning_effort", effort)

		return nil

	case s.openAIModel:
		return s.setNestedField("reasoning", "effort", effort)

	default:
		reasoningConfig := jsonobject.Object{}

		if config.Type != nil && *config.Type != ReasoningAdaptive {
			reasoningConfig = reasoningConfig.With("type", mustEncode(*config.Type))
		}

		if budget != nil {
			reasoningConfig = reasoningConfig.With("budgetTokens", mustEncode(*budget))
		}

		reasoningConfig = reasoningConfig.With("maxReasoningEffort", effort)

		return s.setObjectField("reasoningConfig", reasoningConfig)
	}
}

func (s *requestState) applyStructuredOutput(output structuredOutput) error {
	if output.nativeSchema == nil {
		return nil
	}

	format := jsonobject.Object{
		{
			Key:   "type",
			Value: jsonobject.EncodeString("json_schema"),
		},
		{
			Key:   "schema",
			Value: output.nativeSchema,
		},
	}

	encoded, err := format.Encode()
	if err != nil {
		return fmt.Errorf("%w: response format schema: %w", model.ErrInvalidArgument, err)
	}

	return s.setNestedField("output_config", "format", encoded)
}

// setObjectField sets an additional field to an encoded object.
func (s *requestState) setObjectField(key string, object jsonobject.Object) error {
	encoded, err := object.Encode()
	if err != nil {
		return fmt.Errorf("%w: encode %s: %w", model.ErrInvalidArgument, key, err)
	}

	s.additionalFields = s.additionalFields.With(key, encoded)

	return nil
}

// setNestedField sets member of the additional field object under key.
func (s *requestState) setNestedField(key string, member string, value json.RawMessage) error {
	fields, err := withMember(s.additionalFields, key, member, value)
	if err != nil {
		return fmt.Errorf("%w: encode %s: %w", model.ErrInvalidArgument, key, err)
	}

	s.additionalFields = fields

	return nil
}

// restrictInference removes the inference settings that Claude thinking
// and OpenAI models reject.
func (s *requestState) restrictInference(inference *inferenceConfig) {
	if s.anthropicModel && s.thinkingEnabled() {
		if inference.Temperature != nil {
			inference.Temperature = nil
			s.warn("temperature", "temperature is not supported when thinking is enabled")
		}

		if inference.TopP != nil {
			inference.TopP = nil
			s.warn("topP", "topP is not supported when thinking is enabled")
		}

		if inference.TopK != nil {
			inference.TopK = nil
			s.warn("topK", "topK is not supported when thinking is enabled")
		}
	}

	if !s.openAIModel {
		return
	}

	// OpenAI models reject stop sequences on the Converse API; models
	// other than gpt-oss also reject temperature and topP.
	if !s.gptOSSModel {
		if inference.Temperature != nil {
			inference.Temperature = nil
			s.warnOpenAI("temperature")
		}

		if inference.TopP != nil {
			inference.TopP = nil
			s.warnOpenAI("topP")
		}
	}

	if inference.StopSequences != nil {
		inference.StopSequences = nil
		s.warnOpenAI("stopSequences")
	}
}

func (s *requestState) warnOpenAI(feature string) {
	s.warn(feature, feature+" is not supported by this OpenAI model on the Converse API")
}

// prompt converts the prompt, removing tool content when the request
// declares no tools, which Bedrock rejects, and adding the JSON instruction.
func (s *requestState) prompt(tools preparedTools, output structuredOutput) (convertedPrompt, error) {
	prompt := s.options.Prompt

	if !tools.hasTools() && hasToolContent(prompt) {
		prompt = withoutToolContent(prompt)
		s.warn(
			"toolContent",
			"Tool calls and results removed from conversation because Bedrock does not support tool content without active tools.",
		)
	}

	if output.useJSONInstruction {
		injected, err := injectJSONInstruction(prompt, s.options.ResponseFormat.Schema)
		if err != nil {
			return convertedPrompt{}, err
		}

		prompt = injected
	}

	return convertPrompt(prompt, isMistralModel(s.modelID))
}

// encode builds the request body. Passed-through provider options follow
// the known fields and may override them; the tool configuration comes
// last.
func (s *requestState) encode(prompt convertedPrompt, inference inferenceConfig, tools preparedTools) (json.RawMessage, error) {
	request := converseRequest{
		System:                       prompt.system,
		Messages:                     prompt.messages,
		AdditionalModelRequestFields: s.additionalFields,
		RequestMetadata:              s.read.bedrock.RequestMetadata,
	}

	if s.anthropicModel {
		request.AdditionalModelResponseFieldPaths = []string{
			stopSequenceFieldPath,
		}
	}

	if !inference.isEmpty() {
		request.InferenceConfig = &inference
	}

	if tier := s.read.bedrock.ServiceTier; tier != nil {
		request.ServiceTier = &serviceTierConfig{
			Type: *tier,
		}
	}

	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	body, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	body = body.Merge(s.read.passthrough)

	if tools.config != nil {
		config, err := json.Marshal(tools.config)
		if err != nil {
			return nil, fmt.Errorf("%w: encode tool configuration: %w", model.ErrInvalidArgument, err)
		}

		body = body.With("toolConfig", config)
	}

	return body.Encode()
}

// hasToolContent reports whether any message holds tool calls or results.
func hasToolContent(prompt []model.Message) bool {
	for _, message := range prompt {
		switch message := message.(type) {
		case model.AssistantMessage:
			if slices.ContainsFunc(message.Content, isToolPart) {
				return true
			}

		case model.ToolMessage:
			for _, part := range message.Content {
				if _, ok := part.(model.ToolResultPart); ok {
					return true
				}
			}
		}
	}

	return false
}

// withoutToolContent removes tool calls and results, and the messages they
// leave empty. System messages are kept.
func withoutToolContent(prompt []model.Message) []model.Message {
	filtered := make([]model.Message, 0, len(prompt))

	for _, message := range prompt {
		switch message := message.(type) {
		case model.UserMessage:
			if len(message.Content) > 0 {
				filtered = append(filtered, message)
			}

		case model.AssistantMessage:
			var content []model.AssistantPart
			for _, part := range message.Content {
				if !isToolPart(part) {
					content = append(content, part)
				}
			}

			if len(content) > 0 {
				message.Content = content
				filtered = append(filtered, message)
			}

		case model.ToolMessage:
			var content []model.ToolPart
			for _, part := range message.Content {
				if _, ok := part.(model.ToolResultPart); !ok {
					content = append(content, part)
				}
			}

			if len(content) > 0 {
				message.Content = content
				filtered = append(filtered, message)
			}

		default:
			filtered = append(filtered, message)
		}
	}

	return filtered
}

func isToolPart(part model.AssistantPart) bool {
	switch part.(type) {
	case model.ToolCallPart, model.ToolResultPart:
		return true
	default:
		return false
	}
}

// withMember sets member of the object stored under key, creating the
// object or replacing a value that is not one.
func withMember(fields jsonobject.Object, key string, member string, value json.RawMessage) (jsonobject.Object, error) {
	object := jsonobject.Object{}

	if existing, found := fields.Get(key); found {
		if parsed, err := jsonobject.Parse(existing); err == nil {
			object = parsed
		}
	}

	encoded, err := object.With(member, value).Encode()
	if err != nil {
		return nil, err
	}

	return fields.With(key, encoded), nil
}

// mustEncode encodes values whose types always encode.
func mustEncode(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("encode %T: %v", value, err))
	}

	return encoded
}

// formatNumber formats a number the way JavaScript prints it.
func formatNumber(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
