package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// responsesPath is the Responses API endpoint.
const responsesPath = "/responses"

// listToolsName is the tool name of MCP tool listings.
const listToolsName = "list_tools"

// supportedURLs are the image URLs that Hugging Face downloads itself.
var supportedURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
}

// ResponsesModel is a Hugging Face Responses API language model.
type ResponsesModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "huggingface.responses" by
// default.
func (m *ResponsesModel) Provider() string {
	return m.provider.name + ".responses"
}

// ModelID returns the model identifier.
func (m *ResponsesModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Hugging Face downloads image URLs itself.
func (m *ResponsesModel) SupportedURLs() map[string][]*regexp.Regexp {
	return supportedURLs
}

// Generate performs a non-streaming Responses API call.
func (m *ResponsesModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	body, warnings, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(responsesPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded responsesResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if decoded.Error != nil {
		return nil, &model.APICallError{
			Message:         decoded.Error.Message,
			URL:             request.URL,
			StatusCode:      http.StatusBadRequest,
			ResponseHeaders: response.Headers,
			ResponseBody:    string(response.Body),
		}
	}

	content, err := m.generatedContent(decoded.Output)
	if err != nil {
		return nil, err
	}

	return &model.GenerateResult{
		Content:      content,
		FinishReason: finishReason(decoded.IncompleteDetails),
		Usage:        convertUsage(decoded.Usage),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: ResponsesMetadata{
				ResponseID: new(decoded.ID),
			},
		},
		Warnings: warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID:        decoded.ID,
				Timestamp: unixTime(decoded.CreatedAt),
				ModelID:   decoded.Model,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// Stream performs a streaming Responses API call.
func (m *ResponsesModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	body, warnings, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	body.Stream = true

	request, err := m.provider.apiRequest(responsesPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, body)
	if err != nil {
		return nil, err
	}

	processor := &responsesStream{
		includeRawChunks: options.IncludeRawChunks,
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
	}

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

	unsupported := []struct {
		feature string
		set     bool
	}{
		{
			feature: "topK",
			set:     options.TopK != nil,
		},
		{
			feature: "seed",
			set:     options.Seed != nil,
		},
		{
			feature: "presencePenalty",
			set:     options.PresencePenalty != nil,
		},
		{
			feature: "frequencyPenalty",
			set:     options.FrequencyPenalty != nil,
		},
		{
			feature: "stopSequences",
			set:     options.StopSequences != nil,
		},
	}

	for _, setting := range unsupported {
		if setting.set {
			warnings = append(warnings, model.UnsupportedWarning(setting.feature, ""))
		}
	}

	input, messageWarnings, err := convertPrompt(options.Prompt)
	if err != nil {
		return responsesRequest{}, nil, err
	}

	warnings = append(warnings, messageWarnings...)

	responsesOptions, err := readResponsesOptions(options.ProviderOptions)
	if err != nil {
		return responsesRequest{}, nil, err
	}

	tools, err := prepareTools(options.Tools, options.ToolChoice)
	if err != nil {
		return responsesRequest{}, nil, err
	}

	warnings = append(warnings, tools.warnings...)

	body := responsesRequest{
		Model:           m.modelID,
		Input:           input,
		Temperature:     options.Temperature,
		TopP:            options.TopP,
		MaxOutputTokens: options.MaxOutputTokens,
		Text:            textConfigOf(options.ResponseFormat, responsesOptions),
		Metadata:        responsesOptions.Metadata,
		Instructions:    responsesOptions.Instructions,
		Tools:           tools.tools,
		ToolChoice:      tools.toolChoice,
	}

	if responsesOptions.ReasoningEffort != nil {
		body.Reasoning = &reasoningConfig{
			Effort: *responsesOptions.ReasoningEffort,
		}
	}

	return body, warnings, nil
}

// textConfigOf requests a JSON schema for JSON response formats with a
// schema. JSON without a schema is not constrained.
func textConfigOf(format *model.ResponseFormat, options ResponsesOptions) *textConfig {
	if format == nil || format.Type != model.ResponseFormatJSON || format.Schema == nil {
		return nil
	}

	name := format.Name
	if name == "" {
		name = "response"
	}

	return &textConfig{
		Format: textFormat{
			Type:        "json_schema",
			Strict:      optional.Or(options.StrictJSONSchema, false),
			Name:        name,
			Description: format.Description,
			Schema:      format.Schema,
		},
	}
}

// generatedContent converts the output items in order.
func (m *ResponsesModel) generatedContent(items []outputItem) ([]model.Content, error) {
	var content []model.Content

	for _, item := range items {
		converted, err := m.itemContent(item)
		if err != nil {
			return nil, err
		}

		content = append(content, converted...)
	}

	return content, nil
}

func (m *ResponsesModel) itemContent(item outputItem) ([]model.Content, error) {
	switch {
	case item.Message != nil:
		return m.messageContent(*item.Message), nil

	case item.Reasoning != nil:
		return reasoningContentOf(*item.Reasoning), nil

	case item.MCPCall != nil:
		return toolContent(toolCallSpec{
			toolCallID:       item.MCPCall.ID,
			toolName:         item.MCPCall.Name,
			input:            item.MCPCall.Arguments,
			output:           item.MCPCall.Output,
			providerExecuted: true,
		})

	case item.MCPListTools != nil:
		return listToolsContent(*item.MCPListTools)

	case item.FunctionCall != nil:
		return toolContent(toolCallSpec{
			toolCallID: item.FunctionCall.CallID,
			toolName:   item.FunctionCall.Name,
			input:      item.FunctionCall.Arguments,
			output:     item.FunctionCall.Output,
		})

	default:
		return nil, nil
	}
}

// messageContent returns each output text, followed by its URL citations.
func (m *ResponsesModel) messageContent(item messageItem) []model.Content {
	var content []model.Content

	for _, part := range item.Content {
		content = append(content, model.Text{
			Text:             part.Text,
			ProviderMetadata: itemMetadata(item.ID),
		})

		for _, citation := range part.Annotations {
			content = append(content, model.Source{
				SourceType: model.SourceURL,
				ID:         m.provider.generateID(),
				URL:        citation.URL,
				Title:      citation.Title,
			})
		}
	}

	return content
}

func reasoningContentOf(item reasoningItem) []model.Content {
	var content []model.Content

	for _, part := range item.Content {
		content = append(content, model.Reasoning{
			Text:             part.Text,
			ProviderMetadata: itemMetadata(item.ID),
		})
	}

	return content
}

// toolCallSpec describes a tool call output item.
type toolCallSpec struct {
	toolCallID       string
	toolName         string
	input            string
	output           *string
	providerExecuted bool
}

// toolContent returns the tool call, followed by its result when the item
// carries an output.
func toolContent(spec toolCallSpec) ([]model.Content, error) {
	content := []model.Content{
		model.ToolCall{
			ToolCallID:       spec.toolCallID,
			ToolName:         spec.toolName,
			Input:            spec.input,
			ProviderExecuted: spec.providerExecuted,
		},
	}

	if spec.output == nil || *spec.output == "" {
		return content, nil
	}

	result, err := json.Marshal(*spec.output)
	if err != nil {
		return nil, fmt.Errorf("encode tool result: %w", err)
	}

	return append(content, model.ToolResult{
		ToolCallID: spec.toolCallID,
		ToolName:   spec.toolName,
		Result:     result,
	}), nil
}

type listToolsInput struct {
	ServerLabel string `json:"server_label"`
}

type listToolsResult struct {
	Tools []json.RawMessage `json:"tools"`
}

// listToolsContent returns an MCP tool listing as a provider-executed call
// and its result.
func listToolsContent(item mcpListToolsItem) ([]model.Content, error) {
	input, err := json.Marshal(listToolsInput{
		ServerLabel: item.ServerLabel,
	})
	if err != nil {
		return nil, fmt.Errorf("encode tool input: %w", err)
	}

	content := []model.Content{
		model.ToolCall{
			ToolCallID:       item.ID,
			ToolName:         listToolsName,
			Input:            string(input),
			ProviderExecuted: true,
		},
	}

	if item.Tools == nil {
		return content, nil
	}

	result, err := json.Marshal(listToolsResult{
		Tools: item.Tools,
	})
	if err != nil {
		return nil, fmt.Errorf("encode tool result: %w", err)
	}

	return append(content, model.ToolResult{
		ToolCallID: item.ID,
		ToolName:   listToolsName,
		Result:     result,
	}), nil
}

// finishReason maps the incomplete reason, treating a complete response as
// stopped.
func finishReason(details *incompleteDetails) model.FinishReason {
	if details == nil {
		return model.FinishReason{
			Unified: model.FinishStop,
		}
	}

	return model.FinishReason{
		Unified: unifiedFinishReason(details.Reason),
		Raw:     details.Reason,
	}
}

func unifiedFinishReason(reason string) model.UnifiedFinishReason {
	switch reason {
	case "stop":
		return model.FinishStop
	case "length":
		return model.FinishLength
	case "content_filter":
		return model.FinishContentFilter
	case "tool_calls":
		return model.FinishToolCalls
	case "error":
		return model.FinishError
	default:
		return model.FinishOther
	}
}

// convertUsage converts Responses API usage. Missing usage reports no
// counts.
func convertUsage(usage *responsesUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	cachedTokens := 0
	if usage.InputTokensDetails != nil {
		cachedTokens = usage.InputTokensDetails.CachedTokens
	}

	reasoningTokens := 0
	if usage.OutputTokensDetails != nil {
		reasoningTokens = usage.OutputTokensDetails.ReasoningTokens
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:     new(usage.InputTokens),
			NoCache:   new(usage.InputTokens - cachedTokens),
			CacheRead: new(cachedTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(usage.OutputTokens),
			Text:      new(usage.OutputTokens - reasoningTokens),
			Reasoning: new(reasoningTokens),
		},
		Raw: usage.raw,
	}
}

func unixTime(seconds float64) time.Time {
	return time.Unix(int64(seconds), 0)
}

var _ model.LanguageModel = (*ResponsesModel)(nil)
