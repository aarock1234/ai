package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"regexp"
	"strconv"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// languageModelPath is the language model endpoint.
const languageModelPath = "/language-model"

// supportedURLs lets the gateway download every file URL itself.
var supportedURLs = map[string][]*regexp.Regexp{
	"*/*": {
		regexp.MustCompile(`.*`),
	},
}

// ChatModel is an AI Gateway language model. It forwards the call options
// and returns the gateway's normalized result. It is safe for concurrent
// use.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "gateway" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs reports that the gateway downloads file URLs of every media
// type.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	return supportedURLs
}

// generateResponse is the normalized generate result on the wire.
type generateResponse struct {
	Content          []json.RawMessage `json:"content"`
	FinishReason     wireFinishReason  `json:"finishReason"`
	Usage            wireUsage         `json:"usage"`
	ProviderMetadata wireMetadata      `json:"providerMetadata"`
	Warnings         []wireWarning     `json:"warnings"`
}

// Generate performs one non-streaming call through the gateway.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	body, err := encodeCallOptions(options)
	if err != nil {
		return nil, err
	}

	request, method, err := m.provider.modelRequest(languageModelPath, options.Headers, m.configHeaders(false))
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var decoded generateResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	content, err := decodeContents(decoded.Content)
	if err != nil {
		return nil, err
	}

	return &model.GenerateResult{
		Content:          content,
		FinishReason:     decoded.FinishReason.finishReason(),
		Usage:            decoded.Usage.usage(),
		ProviderMetadata: decoded.ProviderMetadata.providerMetadata(),
		Warnings:         decodeWarnings(decoded.Warnings),
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// Stream performs one streaming call through the gateway, which sends the
// stream parts as server-sent events.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	body, err := encodeCallOptions(options)
	if err != nil {
		return nil, err
	}

	request, method, err := m.provider.modelRequest(languageModelPath, options.Headers, m.configHeaders(true))
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, body)
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	result := model.NewStreamResult(streamParts(ctx, response.Body, options.IncludeRawChunks), response.Body.Close)
	result.Request = model.RequestInfo{
		Body: response.RequestBody,
	}
	result.Response = model.StreamResponseInfo{
		Headers: response.Headers,
	}

	return result, nil
}

// configHeaders describe the call to the gateway.
func (m *ChatModel) configHeaders(streaming bool) http.Header {
	return http.Header{
		"Ai-Language-Model-Specification-Version": {
			"4",
		},
		"Ai-Language-Model-Id": {
			m.modelID,
		},
		"Ai-Language-Model-Streaming": {
			strconv.FormatBool(streaming),
		},
	}
}

var _ model.LanguageModel = (*ChatModel)(nil)
