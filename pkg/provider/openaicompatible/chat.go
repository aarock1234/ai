package openaicompatible

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"regexp"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

// chatPath is the chat completions endpoint.
const chatPath = "/chat/completions"

// RequestBodyTransform rewrites a chat request body before it is sent. The
// members are arbitrary JSON. It returns the body to send and warnings about
// the changes it made.
type RequestBodyTransform func(body map[string]json.RawMessage) (map[string]json.RawMessage, []model.Warning, error)

// ChatConfig configures a ChatModel. Providers built on this package create
// it themselves; Provider.Chat fills it from the provider options.
type ChatConfig struct {
	// Provider is the model's provider identifier, such as "name.chat". The
	// part before the first dot is the provider options namespace.
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

	// SupportsStructuredOutputs enables JSON schema response formats.
	SupportsStructuredOutputs bool

	// SupportsMultiPartToolContent sends multi-part tool results as content
	// parts instead of JSON text.
	SupportsMultiPartToolContent bool

	// SupportedURLs are the URLs the model downloads itself.
	SupportedURLs map[string][]*regexp.Regexp

	// TransformRequestBody optionally rewrites request bodies.
	TransformRequestBody RequestBodyTransform

	// MetadataExtractor optionally captures provider-specific metadata.
	MetadataExtractor MetadataExtractor

	// ConvertUsage optionally replaces ConvertChatUsage.
	ConvertUsage UsageConverter

	// GenerateID names tool calls that arrive without an ID. Nil uses random
	// IDs.
	GenerateID func() string
}

// ChatModel is an OpenAI-compatible chat completions language model.
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

// SupportsStructuredOutputs reports whether JSON schema response formats are
// sent to the provider.
func (m *ChatModel) SupportsStructuredOutputs() bool {
	return m.config.SupportsStructuredOutputs
}

// SupportedURLs returns the configured URL patterns.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return m.config.SupportedURLs
}

// chatArgs is a prepared request.
type chatArgs struct {
	body        map[string]json.RawMessage
	warnings    []model.Warning
	metadataKey string
}

// Generate performs a non-streaming chat completion.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
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

	var decoded chatResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if len(decoded.Choices) == 0 {
		return nil, fmt.Errorf("%w: response did not contain any choices", model.ErrInvalidResponse)
	}

	choice := decoded.Choices[0]

	metadata, err := m.generateMetadata(args.metadataKey, response.Body, decoded.Usage)
	if err != nil {
		return nil, err
	}

	return &model.GenerateResult{
		Content:          generatedContent(choice.Message, args.metadataKey, m.generateID),
		FinishReason:     MapFinishReason(valueOrEmpty(choice.FinishReason)),
		Usage:            m.convertUsage(decoded.Usage),
		ProviderMetadata: metadata,
		Warnings:         args.warnings,
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

// Stream performs a streaming chat completion.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
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

	processor := newChatStream(chatStreamSettings{
		metadataKey:      args.metadataKey,
		includeRawChunks: options.IncludeRawChunks,
		convertUsage:     m.convertUsage,
		extractor:        m.streamExtractor(),
		generateID:       m.generateID,
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
func (m *ChatModel) prepareArgs(options model.CallOptions, stream bool) (chatArgs, error) {
	name := optionsName(m.config.Provider)

	compatible, err := mergedOptions[ChatOptions](
		options.ProviderOptions,
		deprecatedNamespace,
		Namespace,
		name,
		toCamelCase(name),
	)
	if err != nil {
		return chatArgs{}, err
	}

	warnings := m.callWarnings(options, name)

	tools, err := prepareTools(options.Tools, options.ToolChoice)
	if err != nil {
		return chatArgs{}, err
	}

	key := metadataKey(name, options.ProviderOptions)

	messages, err := convertChatPrompt(options.Prompt, promptSettings{
		provider:                     name,
		metadataKey:                  key,
		supportsMultiPartToolContent: m.config.SupportsMultiPartToolContent,
	})
	if err != nil {
		return chatArgs{}, err
	}

	head := chatRequest{
		Model:            m.modelID,
		User:             compatible.User,
		MaxTokens:        options.MaxOutputTokens,
		Temperature:      options.Temperature,
		TopP:             options.TopP,
		FrequencyPenalty: options.FrequencyPenalty,
		PresencePenalty:  options.PresencePenalty,
		ResponseFormat:   m.responseFormat(options.ResponseFormat, compatible),
		Stop:             options.StopSequences,
		Seed:             options.Seed,
	}

	tail := chatRequestTail{
		ReasoningEffort: reasoningEffort(compatible, options.Reasoning),
		Verbosity:       compatible.TextVerbosity,
		Messages:        messages,
		Tools:           tools.Tools,
		ToolChoice:      tools.ToolChoice,
	}

	passthrough, err := passthroughMembers(options.ProviderOptions, name)
	if err != nil {
		return chatArgs{}, err
	}

	body, err := assembleChatBody(head, passthrough, tail)
	if err != nil {
		return chatArgs{}, err
	}

	if stream {
		if err := setStreamMembers(body, m.config.IncludeUsage); err != nil {
			return chatArgs{}, err
		}
	}

	warnings = append(warnings, tools.Warnings...)

	body, warnings, err = m.transformBody(body, warnings)
	if err != nil {
		return chatArgs{}, err
	}

	return chatArgs{
		body:        body,
		warnings:    warnings,
		metadataKey: key,
	}, nil
}

// callWarnings reports deprecated provider options namespaces and the call
// settings that the model does not support.
func (m *ChatModel) callWarnings(options model.CallOptions, name string) []model.Warning {
	warnings := deprecatedNamespaceWarnings(options.ProviderOptions)
	warnings = append(warnings, deprecatedNameWarnings(name, options.ProviderOptions)...)

	if options.TopK != nil {
		warnings = append(warnings, model.UnsupportedWarning("topK", ""))
	}

	if m.ignoresSchema(options.ResponseFormat) {
		warnings = append(warnings, model.UnsupportedWarning(
			"responseFormat",
			"JSON response format schema is only supported with structuredOutputs",
		))
	}

	return warnings
}

// ignoresSchema reports whether a JSON schema cannot be sent because the
// provider lacks structured outputs.
func (m *ChatModel) ignoresSchema(format *model.ResponseFormat) bool {
	return format != nil &&
		format.Type == model.ResponseFormatJSON &&
		format.Schema != nil &&
		!m.config.SupportsStructuredOutputs
}

// responseFormat requests a JSON schema when the provider supports structured
// outputs and a JSON object otherwise.
func (m *ChatModel) responseFormat(format *model.ResponseFormat, options ChatOptions) *chatResponseFormat {
	if format == nil || format.Type != model.ResponseFormatJSON {
		return nil
	}

	if !m.config.SupportsStructuredOutputs || format.Schema == nil {
		return &chatResponseFormat{
			Type: chatResponseFormatJSONObject,
		}
	}

	name := format.Name
	if name == "" {
		name = "response"
	}

	strict := true
	if options.StrictJSONSchema != nil {
		strict = *options.StrictJSONSchema
	}

	return &chatResponseFormat{
		Type: chatResponseFormatJSONSchema,
		JSONSchema: &chatJSONSchema{
			Schema:      format.Schema,
			Strict:      strict,
			Name:        name,
			Description: format.Description,
		},
	}
}

// reasoningEffort prefers the provider option and falls back to the
// provider-independent reasoning level.
func reasoningEffort(options ChatOptions, level model.ReasoningEffort) *string {
	if options.ReasoningEffort != nil {
		return options.ReasoningEffort
	}

	if !reasoning.IsCustom(level) {
		return nil
	}

	return new(string(level))
}

// passthroughMembers returns the provider namespace options that are not
// ChatOptions fields. They are sent in the request body unchanged.
func passthroughMembers(options model.ProviderOptions, name string) (map[string]json.RawMessage, error) {
	members, err := mergedMembers(options, name, toCamelCase(name))
	if err != nil {
		return nil, err
	}

	for _, key := range chatOptionKeys {
		delete(members, key)
	}

	return members, nil
}

// assembleChatBody layers the passthrough options over the request head. The
// tail fields always win, and an absent tail field removes a passthrough
// member of the same name.
func assembleChatBody(head chatRequest, passthrough map[string]json.RawMessage, tail chatRequestTail) (map[string]json.RawMessage, error) {
	body, err := jsonobject.Members(head)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	maps.Copy(body, passthrough)

	tailMembers, err := jsonobject.Members(tail)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	for _, key := range chatRequestTailKeys {
		delete(body, key)
	}

	maps.Copy(body, tailMembers)

	return body, nil
}

func (m *ChatModel) transformBody(body map[string]json.RawMessage, warnings []model.Warning) (map[string]json.RawMessage, []model.Warning, error) {
	if m.config.TransformRequestBody == nil {
		return body, warnings, nil
	}

	transformed, transformWarnings, err := m.config.TransformRequestBody(body)
	if err != nil {
		return nil, nil, fmt.Errorf("transform request body: %w", err)
	}

	return transformed, append(warnings, transformWarnings...), nil
}

// apiRequest describes a call to the chat completions endpoint.
func (m *ChatModel) apiRequest(callHeaders http.Header) (apicall.Request, error) {
	return newAPIRequest(apiRequestSettings{
		path:        chatPath,
		buildURL:    m.config.BuildURL,
		headers:     m.config.Headers,
		callHeaders: callHeaders,
		client:      m.config.HTTPClient,
		parseError:  m.config.ParseError,
	})
}

func (m *ChatModel) convertUsage(usage *ChatUsage) model.Usage {
	if m.config.ConvertUsage != nil {
		return m.config.ConvertUsage(usage)
	}

	return ConvertChatUsage(usage)
}

// generateMetadata combines the prediction token counts with extracted
// metadata.
func (m *ChatModel) generateMetadata(key string, body json.RawMessage, usage *ChatUsage) (model.ProviderMetadata, error) {
	var extracted model.ProviderMetadata

	if m.config.MetadataExtractor != nil {
		var err error

		extracted, err = m.config.MetadataExtractor.ExtractMetadata(body)
		if err != nil {
			return nil, fmt.Errorf("extract provider metadata: %w", err)
		}
	}

	return chatMetadata(key, usage, extracted), nil
}

// generateID returns a configured or random tool call ID.
func (m *ChatModel) generateID() string {
	if m.config.GenerateID != nil {
		return m.config.GenerateID()
	}

	return id.New("")
}

func (m *ChatModel) streamExtractor() StreamMetadataExtractor {
	if m.config.MetadataExtractor == nil {
		return nil
	}

	return m.config.MetadataExtractor.NewStreamExtractor()
}

// generatedContent converts a response message: reasoning first, then the
// content parts in order, then tool calls.
func generatedContent(message chatResponseMessage, metadataKey string, generateID func() string) []model.Content {
	var content []model.Content

	thinking := reasoningText(message.ReasoningContent, message.Reasoning)
	if thinking != "" {
		content = append(content, model.Reasoning{
			Text: thinking,
		})
	}

	for _, piece := range message.Content.pieces() {
		content = append(content, pieceContent(piece))
	}

	for _, toolCall := range message.ToolCalls {
		toolCallID := toolCall.ID
		if toolCallID == "" {
			toolCallID = generateID()
		}

		content = append(content, model.ToolCall{
			ToolCallID:       toolCallID,
			ToolName:         toolCall.Function.Name,
			Input:            toolCall.Function.Arguments,
			ProviderMetadata: toolCallMetadata(metadataKey, toolCall.ExtraContent.thoughtSignature()),
		})
	}

	return content
}

func pieceContent(piece contentPiece) model.Content {
	if piece.Reasoning {
		return model.Reasoning{
			Text: piece.Text,
		}
	}

	return model.Text{
		Text: piece.Text,
	}
}

func valueOrEmpty(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}

var _ model.LanguageModel = (*ChatModel)(nil)
