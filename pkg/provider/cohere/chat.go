package cohere

import (
	"context"
	"regexp"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// chatPath is the chat endpoint.
	chatPath = "/chat"

	// maxThinkingBudget bounds the thinking budget derived from a reasoning
	// level, and is the output size the budget is a fraction of.
	maxThinkingBudget = 32768

	// defaultCitationTitle names cited documents without a title.
	defaultCitationTitle = "Document"
)

// imageURLs are the image URLs that Cohere downloads itself.
var imageURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
}

// ChatModel is a Cohere chat language model.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "cohere.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name + ".chat"
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Cohere downloads image URLs itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return imageURLs
}

// CitationMetadata is the provider metadata of a cited document source.
type CitationMetadata struct {
	// Start and End locate the cited span in the generated text.
	Start int `json:"start"`
	End   int `json:"end"`

	// Text is the cited span.
	Text string `json:"text"`

	Sources []CitationSource `json:"sources"`

	// CitationType is Cohere's citation type, if reported.
	CitationType string `json:"citationType,omitempty"`
}

// chatArgs is a prepared request.
type chatArgs struct {
	body     chatRequest
	warnings []model.Warning
}

// Generate performs a non-streaming chat call.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	args, err := m.prepareArgs(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(chatPath, options.Headers)
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

	result := &model.GenerateResult{
		Content:      m.generatedContent(decoded.Message),
		FinishReason: mapFinishReason(decoded.FinishReason),
		Usage:        convertUsage(decoded.Usage),
		Warnings:     args.warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.GenerationID != nil {
		result.Response.ID = *decoded.GenerationID
	}

	return result, nil
}

// Stream performs a streaming chat call.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	args, err := m.prepareArgs(options)
	if err != nil {
		return nil, err
	}

	args.body.Stream = true

	request, err := m.provider.apiRequest(chatPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, args.body)
	if err != nil {
		return nil, err
	}

	processor := &chatStream{
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

func (m *ChatModel) prepareArgs(options model.CallOptions) (chatArgs, error) {
	chatOptions, err := readChatOptions(options.ProviderOptions)
	if err != nil {
		return chatArgs{}, err
	}

	prompt, err := convertChatPrompt(options.Prompt)
	if err != nil {
		return chatArgs{}, err
	}

	tools, err := prepareTools(options.Tools, options.ToolChoice)
	if err != nil {
		return chatArgs{}, err
	}

	thinking, thinkingWarnings := resolveThinking(chatOptions, options.Reasoning)

	body := chatRequest{
		Model:            m.modelID,
		FrequencyPenalty: options.FrequencyPenalty,
		PresencePenalty:  options.PresencePenalty,
		MaxTokens:        options.MaxOutputTokens,
		Temperature:      options.Temperature,
		P:                options.TopP,
		K:                options.TopK,
		Seed:             options.Seed,
		StopSequences:    options.StopSequences,
		ResponseFormat:   chatResponseFormat(options.ResponseFormat),
		Messages:         prompt.Messages,
		Tools:            tools.Tools,
		ToolChoice:       tools.ToolChoice,
		Documents:        prompt.Documents,
		Thinking:         thinking,
	}

	return chatArgs{
		body:     body,
		warnings: append(tools.Warnings, thinkingWarnings...),
	}, nil
}

// chatResponseFormat requests a JSON object, constrained by the schema when
// there is one.
func chatResponseFormat(format *model.ResponseFormat) *responseFormat {
	if format == nil || format.Type != model.ResponseFormatJSON {
		return nil
	}

	return &responseFormat{
		Type:       "json_object",
		JSONSchema: format.Schema,
	}
}

// resolveThinking prefers the provider option. Otherwise the reasoning level
// disables thinking or sets a budget proportional to the maximum budget.
func resolveThinking(options ChatOptions, level model.ReasoningEffort) (*thinking, []model.Warning) {
	if options.Thinking != nil {
		thinkingType := options.Thinking.Type
		if thinkingType == "" {
			thinkingType = ThinkingEnabled
		}

		return &thinking{
			Type:        thinkingType,
			TokenBudget: options.Thinking.TokenBudget,
		}, nil
	}

	if !reasoning.IsCustom(level) {
		return nil, nil
	}

	if level == model.ReasoningNone {
		return &thinking{
			Type: ThinkingDisabled,
		}, nil
	}

	budget, warnings, ok := reasoning.Budget(level, reasoning.BudgetLimits{
		MaxOutputTokens: maxThinkingBudget,
		MinBudget:       reasoning.DefaultMinBudget,
		MaxBudget:       maxThinkingBudget,
	})
	if !ok {
		return nil, warnings
	}

	return &thinking{
		Type:        ThinkingEnabled,
		TokenBudget: &budget,
	}, nil
}

// generatedContent converts text and thinking, then citations, then tool
// calls.
func (m *ChatModel) generatedContent(message responseMessage) []model.Content {
	var content []model.Content

	for _, item := range message.Content {
		switch {
		case item.Type == "text" && item.Text != "":
			content = append(content, model.Text{
				Text: item.Text,
			})

		case item.Type == "thinking" && item.Thinking != "":
			content = append(content, model.Reasoning{
				Text: item.Thinking,
			})
		}
	}

	for _, citation := range message.Citations {
		content = append(content, m.citationSource(citation))
	}

	for _, toolCall := range message.ToolCalls {
		content = append(content, model.ToolCall{
			ToolCallID: toolCall.ID,
			ToolName:   toolCall.Function.Name,
			Input:      toolArguments(toolCall.Function.Arguments),
		})
	}

	return content
}

// citationSource converts a citation into a document source titled after
// its first cited document.
func (m *ChatModel) citationSource(citation responseCitation) model.Source {
	title := defaultCitationTitle
	if len(citation.Sources) > 0 && citation.Sources[0].Document.Title != "" {
		title = citation.Sources[0].Document.Title
	}

	return model.Source{
		SourceType: model.SourceDocument,
		ID:         m.provider.generateID(),
		MediaType:  "text/plain",
		Title:      title,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: CitationMetadata{
				Start:        citation.Start,
				End:          citation.End,
				Text:         citation.Text,
				Sources:      citation.Sources,
				CitationType: citation.Type,
			},
		},
	}
}

// toolArguments replaces the null arguments Cohere sends for tools without
// parameters with an empty object.
func toolArguments(arguments string) string {
	if arguments == "null" {
		return "{}"
	}

	return arguments
}

var _ model.LanguageModel = (*ChatModel)(nil)
