package perplexity

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"regexp"
	"slices"
	"strconv"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

// agentPath is the Agent API endpoint.
const agentPath = "/v1/agent"

// failedResponseMessage reports a failed response without an error message.
const failedResponseMessage = "Perplexity response failed"

// presets are the model IDs that select an Agent API preset instead of a
// model.
var presets = []string{
	"fast",
	"low",
	"medium",
	"high",
	"xhigh",
}

// supportedURLs are the image URLs that Perplexity downloads itself.
var supportedURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.*$`),
	},
}

// effortMap maps reasoning levels to Agent API efforts.
var effortMap = map[model.ReasoningEffort]ReasoningEffort{
	model.ReasoningMinimal: ReasoningEffortMinimal,
	model.ReasoningLow:     ReasoningEffortLow,
	model.ReasoningMedium:  ReasoningEffortMedium,
	model.ReasoningHigh:    ReasoningEffortHigh,
	model.ReasoningXHigh:   ReasoningEffortXHigh,
	model.ReasoningMax:     ReasoningEffortXHigh,
}

// AgentModel is a Perplexity Agent API language model.
type AgentModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "perplexity" by default.
func (m *AgentModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model or preset identifier.
func (m *AgentModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Perplexity downloads image URLs itself.
func (m *AgentModel) SupportedURLs() map[string][]*regexp.Regexp {
	return supportedURLs
}

// Generate performs a non-streaming Agent API call.
func (m *AgentModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	body, warnings, err := m.prepareRequest(options, false)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(agentPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded agentResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if decoded.Error != nil || decoded.Status == "failed" {
		message := failedResponseMessage
		if decoded.Error != nil {
			message = decoded.Error.Message
		}

		return nil, &model.APICallError{
			Message:         message,
			URL:             request.URL,
			StatusCode:      http.StatusBadRequest,
			ResponseHeaders: response.Headers,
			ResponseBody:    string(response.Body),
		}
	}

	builder := contentBuilder{
		generateID: m.provider.generateID,
		sourceAt:   make(map[string]int),
	}

	for _, item := range decoded.Output {
		builder.addItem(item)
	}

	return &model.GenerateResult{
		Content:          builder.content,
		FinishReason:     finishReason(&decoded, builder.hasFunctionCall),
		Usage:            convertUsage(decoded.Usage),
		ProviderMetadata: agentMetadata(decoded.Usage),
		Warnings:         warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: responseMetadata(&decoded),
			Headers:          response.Headers,
			Body:             response.Body,
		},
	}, nil
}

// Stream performs a streaming Agent API call.
func (m *AgentModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	body, warnings, err := m.prepareRequest(options, true)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(agentPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, body)
	if err != nil {
		return nil, err
	}

	processor := newAgentStream(agentStreamSettings{
		includeRawChunks: options.IncludeRawChunks,
		generateID:       m.provider.generateID,
	})

	start := model.StreamStart{
		Warnings: warnings,
	}

	parts := chatsse.Parts(ctx, response.Body, start, processor)

	result := model.NewStreamResult(parts, response.Body.Close)
	result.Request = model.RequestInfo{
		Body: response.RequestBody,
	}
	result.Response = model.StreamResponseInfo{
		Headers: response.Headers,
	}

	return result, nil
}

// agentRequestFields are the request members the model sets. They replace
// provider option members of the same name.
type agentRequestFields struct {
	Model           string            `json:"model,omitempty"`
	Preset          string            `json:"preset,omitempty"`
	Input           []inputItem       `json:"input"`
	MaxOutputTokens *int              `json:"max_output_tokens,omitempty"`
	Temperature     *float64          `json:"temperature,omitempty"`
	TopP            *float64          `json:"top_p,omitempty"`
	Reasoning       json.RawMessage   `json:"reasoning,omitempty"`
	ResponseFormat  *responseFormat   `json:"response_format,omitempty"`
	Tools           []json.RawMessage `json:"tools,omitempty"`
	Stream          bool              `json:"stream,omitempty"`
}

// replacedMembers are the provider option members that the model always
// sets or removes.
var replacedMembers = []string{
	"input",
	"max_output_tokens",
	"temperature",
	"top_p",
	"reasoning",
	"response_format",
	"tools",
}

// prepareRequest builds the request body and warnings of a call. The
// provider options are forwarded, with the model's own members replacing
// them.
func (m *AgentModel) prepareRequest(options model.CallOptions, stream bool) (map[string]json.RawMessage, []model.Warning, error) {
	agentOptions, members, err := agentOptionMembers(options.ProviderOptions)
	if err != nil {
		return nil, nil, err
	}

	warnings := unsupportedSettingWarnings(options)

	input, inputWarnings, err := convertPrompt(options.Prompt)
	if err != nil {
		return nil, nil, err
	}

	warnings = append(warnings, inputWarnings...)

	tools, toolWarnings, err := m.agentTools(members["tools"], options)
	if err != nil {
		return nil, nil, err
	}

	warnings = append(warnings, toolWarnings...)

	reasoningConfig, reasoningWarnings, err := reasoningMember(members["reasoning"], agentOptions, options.Reasoning)
	if err != nil {
		return nil, nil, err
	}

	warnings = append(warnings, reasoningWarnings...)

	fields := agentRequestFields{
		Input:           input,
		MaxOutputTokens: options.MaxOutputTokens,
		Temperature:     options.Temperature,
		TopP:            options.TopP,
		Reasoning:       reasoningConfig,
		ResponseFormat:  responseFormatOf(options.ResponseFormat),
		Tools:           tools,
		Stream:          stream,
	}

	body := maps.Clone(members)
	if body == nil {
		body = make(map[string]json.RawMessage)
	}

	for _, key := range replacedMembers {
		delete(body, key)
	}

	if slices.Contains(presets, m.modelID) {
		fields.Preset = m.modelID
		delete(body, "preset")
	} else {
		fields.Model = m.modelID
		delete(body, "model")
	}

	format := options.ResponseFormat
	isJSONWithoutSchema := format != nil && format.Type == model.ResponseFormatJSON && format.Schema == nil

	if isJSONWithoutSchema {
		warnings = append(warnings, model.UnsupportedWarning("JSON response format without a schema", ""))
	}

	encoded, err := jsonobject.Members(fields)
	if err != nil {
		return nil, nil, fmt.Errorf("encode request body: %w", err)
	}

	maps.Copy(body, encoded)

	return body, warnings, nil
}

func unsupportedSettingWarnings(options model.CallOptions) []model.Warning {
	settings := []struct {
		feature string
		set     bool
	}{
		{
			feature: "topK",
			set:     options.TopK != nil,
		},
		{
			feature: "frequencyPenalty",
			set:     options.FrequencyPenalty != nil,
		},
		{
			feature: "presencePenalty",
			set:     options.PresencePenalty != nil,
		},
		{
			feature: "stopSequences",
			set:     options.StopSequences != nil,
		},
		{
			feature: "seed",
			set:     options.Seed != nil,
		},
	}

	var warnings []model.Warning
	for _, setting := range settings {
		if setting.set {
			warnings = append(warnings, model.UnsupportedWarning(setting.feature, ""))
		}
	}

	return warnings
}

// agentTools returns the native tools, as configured, followed by the
// function tools. Provider tools are not supported, and the Agent API
// selects tools automatically.
func (m *AgentModel) agentTools(nativeTools json.RawMessage, options model.CallOptions) ([]json.RawMessage, []model.Warning, error) {
	var tools []json.RawMessage
	if !isAbsent(nativeTools) {
		if err := json.Unmarshal(nativeTools, &tools); err != nil {
			return nil, nil, fmt.Errorf("%w: perplexity provider options: tools: %w", model.ErrInvalidArgument, err)
		}
	}

	var warnings []model.Warning

	for _, tool := range options.Tools {
		switch typed := tool.(type) {
		case model.FunctionTool:
			encoded, err := json.Marshal(functionTool{
				Type:        "function",
				Name:        typed.Name,
				Description: typed.Description,
				Parameters:  typed.InputSchema,
				Strict:      typed.Strict,
			})
			if err != nil {
				return nil, nil, fmt.Errorf("encode function tool: %w", err)
			}

			tools = append(tools, encoded)

		case model.ProviderTool:
			warnings = append(warnings, model.UnsupportedWarning("provider-defined tool "+typed.Name, ""))
		}
	}

	if options.ToolChoice != nil && options.ToolChoice.Type != model.ToolChoiceAuto {
		warnings = append(warnings, model.UnsupportedWarning(
			"toolChoice",
			"The Perplexity Agent API currently selects tools automatically.",
		))
	}

	return tools, warnings, nil
}

type reasoningRequest struct {
	Effort ReasoningEffort `json:"effort"`
}

// reasoningMember forwards the configured reasoning unchanged and otherwise
// maps the call's reasoning level to an effort.
func reasoningMember(configured json.RawMessage, options AgentOptions, level model.ReasoningEffort) (json.RawMessage, []model.Warning, error) {
	if options.Reasoning != nil {
		return configured, nil, nil
	}

	if !reasoning.IsCustom(level) {
		return nil, nil, nil
	}

	if level == model.ReasoningNone {
		return nil, []model.Warning{
			model.UnsupportedWarning(`reasoning "none"`, ""),
		}, nil
	}

	effort, warnings, ok := reasoning.Effort(level, effortMap)
	if !ok {
		return nil, warnings, nil
	}

	encoded, err := json.Marshal(reasoningRequest{
		Effort: effort,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("encode reasoning: %w", err)
	}

	return encoded, warnings, nil
}

// responseFormatOf requests strict JSON schema output for JSON response
// formats with a schema.
func responseFormatOf(format *model.ResponseFormat) *responseFormat {
	hasJSONSchema := format != nil && format.Type == model.ResponseFormatJSON && format.Schema != nil
	if !hasJSONSchema {
		return nil
	}

	name := format.Name
	if name == "" {
		name = "response"
	}

	return &responseFormat{
		Type: "json_schema",
		JSONSchema: jsonSchema{
			Name:        name,
			Description: format.Description,
			Schema:      format.Schema,
			Strict:      true,
		},
	}
}

// contentBuilder collects generated content. Sources are deduplicated by
// URL; a search result with a citation ID replaces an earlier source
// without one.
type contentBuilder struct {
	generateID func() string

	content         []model.Content
	sourceAt        map[string]int
	hasFunctionCall bool
}

// citedSource is a source and whether it carries a search result ID.
type citedSource struct {
	source      model.Source
	hasResultID bool
}

func (b *contentBuilder) addItem(item outputItem) {
	switch {
	case item.Message != nil:
		for _, part := range item.Message.Content {
			if part.Type == "output_text" && part.Text != nil {
				b.content = append(b.content, model.Text{
					Text: *part.Text,
				})
			}

			for _, source := range annotationSources(part.Annotations, b.generateID) {
				b.addSource(source)
			}
		}

	case item.SearchResult != nil:
		for _, result := range item.SearchResult.Results {
			b.addSource(searchSource(result, b.generateID))
		}

	case item.FetchResults != nil:
		for _, fetched := range item.FetchResults.Contents {
			b.addSource(fetchedSource(fetched, b.generateID))
		}

	case item.FunctionCall.complete():
		b.hasFunctionCall = true
		b.content = append(b.content, functionToolCall(*item.FunctionCall))
	}
}

func (b *contentBuilder) addSource(cited citedSource) {
	index, seen := b.sourceAt[cited.source.URL]
	if !seen {
		b.sourceAt[cited.source.URL] = len(b.content)
		b.content = append(b.content, cited.source)

		return
	}

	if !cited.hasResultID {
		return
	}

	existing, ok := b.content[index].(model.Source)
	if ok && !hasResultID(existing) {
		b.content[index] = cited.source
	}
}

// hasResultID reports whether a source came from a search result with a
// citation ID.
func hasResultID(source model.Source) bool {
	metadata, ok := source.ProviderMetadata[Namespace].(SearchSourceMetadata)

	return ok && metadata.ResultID != nil
}

// annotationSources returns the URL citations of a content part.
func annotationSources(annotations []annotation, generateID func() string) []citedSource {
	var sources []citedSource

	for _, citation := range annotations {
		if citation.URL == nil {
			continue
		}

		sources = append(sources, citedSource{
			source: model.Source{
				SourceType: model.SourceURL,
				ID:         generateID(),
				URL:        *citation.URL,
				Title:      optional.Or(citation.Title, ""),
			},
		})
	}

	return sources
}

// searchSource converts a search result, using its citation ID as the
// source ID when it has one.
func searchSource(result searchResult, generateID func() string) citedSource {
	sourceID := generateID()
	if result.ID != nil {
		sourceID = strconv.Itoa(*result.ID)
	}

	return citedSource{
		source: model.Source{
			SourceType: model.SourceURL,
			ID:         sourceID,
			URL:        result.URL,
			Title:      result.Title,
			ProviderMetadata: model.ProviderMetadata{
				Namespace: SearchSourceMetadata{
					ResultID:    result.ID,
					Snippet:     result.Snippet,
					Date:        result.Date,
					LastUpdated: result.LastUpdated,
					Source:      result.Source,
				},
			},
		},
		hasResultID: result.ID != nil,
	}
}

func fetchedSource(fetched fetchedContent, generateID func() string) citedSource {
	return citedSource{
		source: model.Source{
			SourceType: model.SourceURL,
			ID:         generateID(),
			URL:        fetched.URL,
			Title:      fetched.Title,
			ProviderMetadata: model.ProviderMetadata{
				Namespace: FetchedSourceMetadata{
					Snippet: fetched.Snippet,
				},
			},
		},
	}
}

// functionToolCall converts a complete function call item.
func functionToolCall(call functionCallItem) model.ToolCall {
	return model.ToolCall{
		ToolCallID: *call.CallID,
		ToolName:   *call.Name,
		Input:      *call.Arguments,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: ToolCallMetadata{
				ItemID:           call.ID,
				ThoughtSignature: optional.Or(call.ThoughtSignature, ""),
			},
		},
	}
}

var _ model.LanguageModel = (*AgentModel)(nil)
