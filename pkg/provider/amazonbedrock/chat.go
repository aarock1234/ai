package amazonbedrock

import (
	"context"
	"fmt"
	"net/http"
	"regexp"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
)

var _ model.LanguageModel = (*ChatModel)(nil)

// s3URLPattern matches the S3 URLs Bedrock reads itself.
var s3URLPattern = regexp.MustCompile(`^s3://`)

// ChatModel is a language model served by the Converse API.
type ChatModel struct {
	modelID  string
	provider *Provider
	family   ModelFamily
}

// ChatModelOption configures a ChatModel.
type ChatModelOption func(*ChatModel)

// WithChatModelFamily sets the family of a model whose ID does not reveal
// it, such as an application inference profile ARN. Only
// ModelFamilyAnthropic applies to chat models.
func WithChatModelFamily(family ModelFamily) ChatModelOption {
	return func(m *ChatModel) {
		m.family = family
	}
}

// Provider returns the provider identifier, "amazon-bedrock" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model ID.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that Bedrock reads S3 images and videos itself.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return map[string][]*regexp.Regexp{
		"image/*": {
			s3URLPattern,
		},
		"video/*": {
			s3URLPattern,
		},
	}
}

// Generate performs one Converse call.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	prepared, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(ctx, bedrock.Runtime, bedrock.ModelPath(m.modelID, "converse"), options.Headers)
	if err != nil {
		return nil, err
	}

	request.ParseError = parseConverseError

	response, err := apicall.PostJSON(ctx, request, prepared.body)
	if err != nil {
		return nil, err
	}

	var decoded converseResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	result, err := m.generateResult(decoded, prepared)
	if err != nil {
		return nil, err
	}

	result.Warnings = prepared.warnings
	result.Request = model.RequestInfo{
		Body: response.RequestBody,
	}
	result.Response = model.ResponseInfo{
		ResponseMetadata: m.responseMetadata(response.Headers),
		Headers:          response.Headers,
		Body:             response.Body,
	}

	return result, nil
}

// responseMetadata reads the request ID and date headers.
func (m *ChatModel) responseMetadata(headers http.Header) model.ResponseMetadata {
	metadata := model.ResponseMetadata{
		ID:      headers.Get("X-Amzn-Requestid"),
		ModelID: m.modelID,
	}

	if date := headers.Get("Date"); date != "" {
		if timestamp, err := http.ParseTime(date); err == nil {
			metadata.Timestamp = timestamp
		}
	}

	return metadata
}

// finishReason maps a Bedrock stop reason. A tool use that carried the JSON
// response is a normal stop.
func finishReason(stopReason string, jsonFromTool bool) model.FinishReason {
	reason := model.FinishReason{
		Unified: model.FinishOther,
		Raw:     stopReason,
	}

	switch stopReason {
	case "stop_sequence", "end_turn":
		reason.Unified = model.FinishStop

	case "max_tokens":
		reason.Unified = model.FinishLength

	case "content_filtered", "guardrail_intervened":
		reason.Unified = model.FinishContentFilter

	case "tool_use":
		reason.Unified = model.FinishToolCalls
		if jsonFromTool {
			reason.Unified = model.FinishStop
		}
	}

	return reason
}

// errMissingField reports a required response field that is absent.
func errMissingField(field string) error {
	return fmt.Errorf("%w: response is missing %s", model.ErrInvalidResponse, field)
}
