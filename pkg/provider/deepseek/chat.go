package deepseek

import (
	"context"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

// chatPath is the chat completions endpoint.
const chatPath = "/chat/completions"

// reasonerModelID is the legacy reasoning model, which always thinks.
const reasonerModelID = "deepseek-reasoner"

// imageURLs are the image URLs that DeepSeek downloads itself.
var imageURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
}

// reasoningEfforts maps provider-independent reasoning levels to DeepSeek
// efforts.
var reasoningEfforts = map[model.ReasoningEffort]ReasoningEffort{
	model.ReasoningMinimal: ReasoningEffortLow,
	model.ReasoningLow:     ReasoningEffortLow,
	model.ReasoningMedium:  ReasoningEffortHigh,
	model.ReasoningHigh:    ReasoningEffortHigh,
	model.ReasoningXHigh:   ReasoningEffortMax,
	model.ReasoningMax:     ReasoningEffortMax,
}

// canonicalEfforts maps the efforts accepted for compatibility to DeepSeek's
// canonical efforts.
var canonicalEfforts = map[ReasoningEffort]ReasoningEffort{
	ReasoningEffortMedium: ReasoningEffortHigh,
	ReasoningEffortXHigh:  ReasoningEffortMax,
}

// ChatConfig configures a ChatModel. Provider.Chat fills it for the DeepSeek
// API; other providers that serve DeepSeek models create it themselves.
type ChatConfig struct {
	// Provider is the model's provider identifier, such as "deepseek.chat".
	Provider string

	// OptionsNamespace is the provider options and metadata namespace. Empty
	// uses the part of Provider before the first dot.
	OptionsNamespace string

	// BuildURL returns the full URL of an API path such as
	// "/chat/completions", or an error when the settings needed to build it
	// are missing.
	BuildURL func(path string) (string, error)

	// Headers returns the provider headers of one request.
	Headers func() (http.Header, error)

	// HTTPClient sends requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// GenerateID names tool calls that arrive without an ID. Nil uses random
	// IDs.
	GenerateID func() string

	// SupportsAssistantPrefixCompletion allows a final assistant message to
	// be a prefix.
	SupportsAssistantPrefixCompletion bool

	// SupportsStrictToolCalls allows strict function tools.
	SupportsStrictToolCalls bool

	// SupportsPenaltySampling sends frequency and presence penalties, which
	// DeepSeek's own API has deprecated.
	SupportsPenaltySampling bool

	// SupportsStructuredOutputs sends JSON schemas as response formats
	// instead of system instructions.
	SupportsStructuredOutputs bool

	// ThinkingUnsupported omits the thinking settings for endpoints that do
	// not accept them.
	ThinkingUnsupported bool
}

// ChatModel is a DeepSeek chat completions language model.
type ChatModel struct {
	modelID string
	config  ChatConfig
}

// NewChatModel returns a chat model that uses config.
func NewChatModel(modelID string, config ChatConfig) *ChatModel {
	return &ChatModel{
		modelID: modelID,
		config:  config,
	}
}

// Provider returns the configured provider identifier.
func (m *ChatModel) Provider() string {
	return m.config.Provider
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that DeepSeek downloads image URLs itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return imageURLs
}

// chatArgs is a prepared request.
type chatArgs struct {
	body     chatRequest
	warnings []model.Warning
}

// Generate performs a non-streaming chat completion.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	args, err := m.prepareArgs(options)
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

	var decoded chatResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if err := decoded.validate(); err != nil {
		return nil, err
	}

	if len(decoded.Choices) == 0 {
		return nil, fmt.Errorf("%w: response did not contain any choices", model.ErrInvalidResponse)
	}

	choice := decoded.Choices[0]

	return &model.GenerateResult{
		Content:      m.generatedContent(choice.Message),
		FinishReason: mapFinishReason(choice.FinishReason),
		Usage:        convertUsage(decoded.Usage),
		ProviderMetadata: model.ProviderMetadata{
			m.namespace(): generateMetadata(decoded, choice),
		},
		Warnings: args.warnings,
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

// Stream performs a streaming chat completion with usage.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	args, err := m.prepareArgs(options)
	if err != nil {
		return nil, err
	}

	args.body.Stream = true
	args.body.StreamOptions = &streamOptions{
		IncludeUsage: true,
	}

	request, err := m.apiRequest(options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, args.body)
	if err != nil {
		return nil, err
	}

	processor := newChatStream(chatStreamSettings{
		namespace:        m.namespace(),
		includeRawChunks: options.IncludeRawChunks,
		generateID:       m.generateID,
		origin: streamOrigin{
			url:     request.URL,
			headers: response.Headers,
		},
	})

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

// prepareArgs builds the request body and warnings of a call.
func (m *ChatModel) prepareArgs(options model.CallOptions) (chatArgs, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions, m.namespace())
	if err != nil {
		return chatArgs{}, err
	}

	messages, warnings, err := convertChatPrompt(options.Prompt, promptSettings{
		responseFormat:                    options.ResponseFormat,
		modelID:                           m.modelID,
		namespace:                         m.namespace(),
		supportsAssistantPrefixCompletion: m.config.SupportsAssistantPrefixCompletion,
		supportsStructuredOutputs:         m.config.SupportsStructuredOutputs,
	})
	if err != nil {
		return chatArgs{}, err
	}

	warnings = append(warnings, m.unsupportedSettings(options)...)

	tools, err := prepareTools(options.Tools, options.ToolChoice, m.config.SupportsStrictToolCalls)
	if err != nil {
		return chatArgs{}, err
	}

	thinkingSetting, thinkingWarnings := m.thinking(chatOptions, options.Reasoning)
	warnings = append(warnings, thinkingWarnings...)

	thinkingEnabled := m.isThinkingEnabled(thinkingSetting)
	warnings = append(warnings, thinkingSamplingWarnings(thinkingEnabled, options)...)

	effort, effortWarnings := reasoningEffort(chatOptions, options.Reasoning)
	warnings = append(warnings, effortWarnings...)

	body := chatRequest{
		Model:          m.modelID,
		MaxTokens:      options.MaxOutputTokens,
		ResponseFormat: m.responseFormat(options.ResponseFormat, chatOptions),
		Stop:           options.StopSequences,
		Messages:       messages,
		Tools:          tools.Tools,
		ToolChoice:     tools.ToolChoice,
		Thinking:       thinkingSetting,
		UserID:         chatOptions.UserID,
	}

	if (chatOptions.Logprobs != nil && *chatOptions.Logprobs) || chatOptions.TopLogprobs != nil {
		body.Logprobs = new(true)
		body.TopLogprobs = chatOptions.TopLogprobs
	}

	if !thinkingEnabled {
		body.Temperature = options.Temperature
		body.TopP = options.TopP
	}

	if m.config.SupportsPenaltySampling {
		body.FrequencyPenalty = options.FrequencyPenalty
		body.PresencePenalty = options.PresencePenalty
	}

	if thinkingSetting == nil || thinkingSetting.Type != ThinkingDisabled {
		body.ReasoningEffort = effort
	}

	return chatArgs{
		body:     body,
		warnings: append(warnings, tools.Warnings...),
	}, nil
}

// unsupportedSettings warns about settings DeepSeek ignores or deprecated.
func (m *ChatModel) unsupportedSettings(options model.CallOptions) []model.Warning {
	var warnings []model.Warning

	if options.TopK != nil {
		warnings = append(warnings, model.UnsupportedWarning("topK", ""))
	}

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning("seed", ""))
	}

	if m.config.SupportsPenaltySampling {
		return warnings
	}

	if options.FrequencyPenalty != nil {
		warnings = append(warnings, deprecatedPenaltyWarning("frequencyPenalty"))
	}

	if options.PresencePenalty != nil {
		warnings = append(warnings, deprecatedPenaltyWarning("presencePenalty"))
	}

	return warnings
}

func deprecatedPenaltyWarning(setting string) model.Warning {
	return model.Warning{
		Type:    model.WarningDeprecated,
		Setting: setting,
		Message: fmt.Sprintf("%s is deprecated by DeepSeek and has been omitted. Remove %s from the request.", setting, setting),
	}
}

// thinking returns the thinking setting: the provider option, or the
// provider-independent reasoning level when it is set.
func (m *ChatModel) thinking(options ChatOptions, level model.ReasoningEffort) (*thinking, []model.Warning) {
	if m.config.ThinkingUnsupported {
		return nil, adaptiveThinkingWarnings(options)
	}

	if options.Thinking != nil && options.Thinking.Type != "" {
		thinkingType := options.Thinking.Type
		if thinkingType == ThinkingAdaptive {
			thinkingType = ThinkingEnabled
		}

		return &thinking{
			Type: thinkingType,
		}, adaptiveThinkingWarnings(options)
	}

	if !reasoning.IsCustom(level) {
		return nil, nil
	}

	if level == model.ReasoningNone {
		return &thinking{
			Type: ThinkingDisabled,
		}, nil
	}

	return &thinking{
		Type: ThinkingEnabled,
	}, nil
}

func adaptiveThinkingWarnings(options ChatOptions) []model.Warning {
	if options.Thinking == nil || options.Thinking.Type != ThinkingAdaptive {
		return nil
	}

	return []model.Warning{
		model.CompatibilityWarning(
			"thinking.type",
			`thinking.type "adaptive" is not a canonical DeepSeek value. mapped to "enabled".`,
		),
	}
}

// isThinkingEnabled reports whether the model thinks: when thinking is
// requested, and by default for the reasoner and V4 models.
func (m *ChatModel) isThinkingEnabled(setting *thinking) bool {
	if m.config.ThinkingUnsupported {
		return false
	}

	if setting != nil {
		return setting.Type != ThinkingDisabled
	}

	return m.modelID == reasonerModelID || isV4Model(m.modelID)
}

// thinkingSamplingWarnings reports sampling settings that thinking ignores.
func thinkingSamplingWarnings(thinkingEnabled bool, options model.CallOptions) []model.Warning {
	if !thinkingEnabled {
		return nil
	}

	var warnings []model.Warning

	if options.Temperature != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"temperature",
			"temperature has no effect when DeepSeek thinking is enabled. Set providerOptions.deepseek.thinking.type to 'disabled' to use temperature.",
		))
	}

	if options.TopP != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"topP",
			"topP has no effect when DeepSeek thinking is enabled. Set providerOptions.deepseek.thinking.type to 'disabled' to use topP.",
		))
	}

	return warnings
}

// reasoningEffort prefers the provider option, mapped to a canonical effort,
// and maps the provider-independent reasoning level otherwise.
func reasoningEffort(options ChatOptions, level model.ReasoningEffort) (ReasoningEffort, []model.Warning) {
	if options.ReasoningEffort != "" {
		canonical, ok := canonicalEfforts[options.ReasoningEffort]
		if !ok {
			return options.ReasoningEffort, nil
		}

		details := fmt.Sprintf("reasoningEffort %q is not a canonical DeepSeek value. mapped to %q.", options.ReasoningEffort, canonical)

		return canonical, []model.Warning{
			model.CompatibilityWarning("reasoningEffort", details),
		}
	}

	if !reasoning.IsCustom(level) || level == model.ReasoningNone {
		return "", nil
	}

	effort, warnings, _ := reasoning.Effort(level, reasoningEfforts)

	return effort, warnings
}

// responseFormat requests a JSON schema when the endpoint supports
// structured outputs and a JSON object otherwise.
func (m *ChatModel) responseFormat(format *model.ResponseFormat, options ChatOptions) *responseFormat {
	if format == nil || format.Type != model.ResponseFormatJSON {
		return nil
	}

	if !m.config.SupportsStructuredOutputs || format.Schema == nil {
		return &responseFormat{
			Type: responseFormatJSONObject,
		}
	}

	name := format.Name
	if name == "" {
		name = "response"
	}

	return &responseFormat{
		Type: responseFormatJSONSchema,
		JSONSchema: &jsonSchemaFormat{
			Schema:      format.Schema,
			Strict:      optional.Or(options.StrictJSONSchema, true),
			Name:        name,
			Description: format.Description,
		},
	}
}

// generatedContent converts a response message: reasoning, then tool calls,
// then text.
func (m *ChatModel) generatedContent(message responseMessage) []model.Content {
	var content []model.Content

	if text := valueOrEmpty(message.ReasoningContent); text != "" {
		content = append(content, model.Reasoning{
			Text: text,
		})
	}

	for _, toolCall := range message.ToolCalls {
		toolCallID := valueOrEmpty(toolCall.ID)
		if toolCallID == "" {
			toolCallID = m.generateID()
		}

		content = append(content, model.ToolCall{
			ToolCallID: toolCallID,
			ToolName:   toolCall.Function.Name,
			Input:      toolCall.Function.Arguments,
		})
	}

	if text := valueOrEmpty(message.Content); text != "" {
		content = append(content, model.Text{
			Text: text,
		})
	}

	return content
}

// generateMetadata reports cache usage and the response details.
func generateMetadata(response chatResponse, choice chatChoice) ChatMetadata {
	hit, miss := cacheTokens(response.Usage)

	metadata := ChatMetadata{
		PromptCacheHitTokens:  hit,
		PromptCacheMissTokens: miss,
		ResponseObject:        valueOrEmpty(response.Object),
		ChoiceIndex:           choice.Index,
		MessageRole:           valueOrEmpty(choice.Message.Role),
		Logprobs:              choice.Logprobs,
		SystemFingerprint:     valueOrEmpty(response.SystemFingerprint),
	}

	for _, toolCall := range choice.Message.ToolCalls {
		if toolCall.Type != nil {
			metadata.ToolCallTypes = append(metadata.ToolCallTypes, *toolCall.Type)
		}
	}

	return metadata
}

// namespace returns the provider options and metadata namespace.
func (m *ChatModel) namespace() string {
	if m.config.OptionsNamespace != "" {
		return m.config.OptionsNamespace
	}

	name, _, _ := strings.Cut(m.config.Provider, ".")

	return strings.TrimSpace(name)
}

// apiRequest describes a call to the chat completions endpoint.
func (m *ChatModel) apiRequest(callHeaders http.Header) (apicall.Request, error) {
	requestURL, err := m.requestURL()
	if err != nil {
		return apicall.Request{}, err
	}

	var providerHeaders http.Header
	if m.config.Headers != nil {
		headers, err := m.config.Headers()
		if err != nil {
			return apicall.Request{}, err
		}

		providerHeaders = headers
	}

	return apicall.Request{
		Client:     m.config.HTTPClient,
		URL:        requestURL,
		Headers:    apicall.MergeHeaders(providerHeaders, callHeaders),
		ParseError: parseError,
	}, nil
}

// requestURL returns the chat completions URL.
func (m *ChatModel) requestURL() (string, error) {
	if m.config.BuildURL == nil {
		return "", fmt.Errorf("%w: chat config sets no BuildURL", model.ErrInvalidArgument)
	}

	return m.config.BuildURL(chatPath)
}

// generateID returns a configured or random tool call ID.
func (m *ChatModel) generateID() string {
	if m.config.GenerateID != nil {
		return m.config.GenerateID()
	}

	return id.New("")
}

var _ model.LanguageModel = (*ChatModel)(nil)
