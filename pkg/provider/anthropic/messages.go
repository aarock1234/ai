package anthropic

import (
	"context"
	"maps"
	"net/http"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

var _ model.LanguageModel = (*MessagesModel)(nil)

// httpURLPattern matches the URLs Anthropic downloads itself.
var httpURLPattern = regexp.MustCompile(`^https?://.*$`)

// MessagesModel is a Claude model served by the Messages API.
type MessagesModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider name, "anthropic.messages" by default.
func (m *MessagesModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model ID.
func (m *MessagesModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Anthropic downloads HTTP and HTTPS images and
// PDF documents itself. A provider configured with WithSupportedURLs
// reports its own patterns instead.
func (m *MessagesModel) SupportedURLs() map[string][]*regexp.Regexp {
	if m.provider.host.supportedURLs != nil {
		return maps.Clone(m.provider.host.supportedURLs)
	}

	return map[string][]*regexp.Regexp{
		"image/*": {
			httpURLPattern,
		},
		"application/pdf": {
			httpURLPattern,
		},
	}
}

// Generate performs one non-streaming Messages API call.
func (m *MessagesModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	call, err := m.prepareCall(ctx, options, false)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, call.api, call.body)
	if err != nil {
		return nil, err
	}

	decoded, nativeUsage, err := decodeMessagesResponse(response.Body)
	if err != nil {
		return nil, err
	}

	converter := newContentConverter(call.request, call.documents, call.generateID)
	content, jsonFromTool := converter.responseContent(decoded.Content)

	metadata := MessagesMetadata{
		Usage:                decoded.Usage,
		StopSequence:         decoded.StopSequence,
		StopDetails:          stopDetailsOf(decoded.StopDetails),
		InputTransformations: decoded.InputTransformations,
		SafeguardResults:     decoded.SafeguardResults,
		Iterations:           convertIterations(nativeUsage.Iterations),
		Container:            containerOf(decoded.Container),
		ContextManagement:    appliedContextManagementOf(decoded.ContextManagement),
	}

	return &model.GenerateResult{
		Content:          content,
		FinishReason:     finishReason(decoded.StopReason, jsonFromTool),
		Usage:            convertUsage(nativeUsage, decoded.Usage),
		ProviderMetadata: call.request.providerMetadata(metadata),
		Warnings:         call.request.warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID:      valueOrZero(decoded.ID),
				ModelID: valueOrZero(decoded.Model),
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// preparedCall is a request with its endpoint, the documents its citations
// may refer to and the generator of source IDs.
type preparedCall struct {
	request *preparedRequest
	api     apicall.Request

	// body is the request body to send: the prepared request, or its JSON
	// after a host transformed it.
	body any

	documents  []citationDocument
	generateID func() string
}

func (m *MessagesModel) prepareCall(ctx context.Context, options model.CallOptions, stream bool) (*preparedCall, error) {
	url, err := m.provider.requestURL(stream)
	if err != nil {
		return nil, err
	}

	configuredHeaders, err := m.provider.configuredHeaders(ctx)
	if err != nil {
		return nil, err
	}

	headerBetas := betasFromHeaders(configuredHeaders, options.Headers)

	request, err := m.prepareRequest(options, stream, headerBetas)
	if err != nil {
		return nil, err
	}

	documents, err := citationDocuments(options.Prompt)
	if err != nil {
		return nil, err
	}

	headers := apicall.MergeHeaders(configuredHeaders, options.Headers)
	if len(request.betas.names) > 0 {
		headers.Set("anthropic-beta", request.betas.header())
	}

	headers, err = m.provider.host.transformedHeaders(headers, options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	body, err := m.provider.host.transformedBody(request.body, request.betas.names)
	if err != nil {
		return nil, err
	}

	return &preparedCall{
		request: request,
		api: apicall.Request{
			Client:     m.provider.httpClient,
			URL:        url,
			Headers:    headers,
			ParseError: parseErrorMessage,
		},
		body:       body,
		documents:  documents,
		generateID: m.provider.generateID,
	}, nil
}

// betasFromHeaders returns the beta features already requested through
// anthropic-beta headers, lowercased.
func betasFromHeaders(headerSets ...http.Header) []string {
	var betas []string

	for _, headers := range headerSets {
		for _, value := range headers.Values("anthropic-beta") {
			for _, beta := range strings.Split(strings.ToLower(value), ",") {
				beta = strings.TrimSpace(beta)
				if beta != "" {
					betas = append(betas, beta)
				}
			}
		}
	}

	return betas
}

// providerMetadata stores the message metadata under "anthropic" and, when
// the call used it, under the provider's own namespace.
func (r *preparedRequest) providerMetadata(metadata MessagesMetadata) model.ProviderMetadata {
	providerMetadata := model.ProviderMetadata{
		Namespace: metadata,
	}

	if r.customNamespace != "" {
		providerMetadata[r.customNamespace] = metadata
	}

	return providerMetadata
}

func stopDetailsOf(details *wireStopDetails) *StopDetails {
	if details == nil {
		return nil
	}

	return &StopDetails{
		Type:             details.Type,
		Category:         valueOrZero(details.Category),
		Explanation:      valueOrZero(details.Explanation),
		RecommendedModel: valueOrZero(details.RecommendedModel),
	}
}

func containerOf(container *wireContainer) *Container {
	if container == nil {
		return nil
	}

	converted := &Container{
		ExpiresAt: container.ExpiresAt,
		ID:        container.ID,
	}

	if container.Skills == nil {
		return converted
	}

	converted.Skills = make([]ContainerSkill, 0, len(container.Skills))
	for _, skill := range container.Skills {
		converted.Skills = append(converted.Skills, ContainerSkill(skill))
	}

	return converted
}

func appliedContextManagementOf(management *wireContextManagement) *AppliedContextManagement {
	if management == nil {
		return nil
	}

	edits := make([]AppliedContextEdit, 0, len(management.AppliedEdits))
	for _, edit := range management.AppliedEdits {
		applied := AppliedContextEdit{
			Type: edit.Type,
		}

		switch edit.Type {
		case ContextEditClearToolUses:
			applied.ClearedToolUses = edit.ClearedToolUses
			applied.ClearedInputTokens = edit.ClearedInputTokens

		case ContextEditClearThinking:
			applied.ClearedThinkingTurns = edit.ClearedThinkingTurns
			applied.ClearedInputTokens = edit.ClearedInputTokens

		case ContextEditCompact:
			// Compaction edits report no counts.

		default:
			continue
		}

		edits = append(edits, applied)
	}

	return &AppliedContextManagement{
		AppliedEdits: edits,
	}
}
