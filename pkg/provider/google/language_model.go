package google

import (
	"context"
	"regexp"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

var _ model.LanguageModel = (*ChatModel)(nil)

var (
	googleFilesURLPattern  = regexp.MustCompile(`^https://generativelanguage\.googleapis\.com/v1beta/files/.*$`)
	youTubeWatchURLPattern = regexp.MustCompile(`^https://(?:www\.)?youtube\.com/watch\?v=[\w-]+(?:&[\w=&.-]*)?$`)
	youTubeShortURLPattern = regexp.MustCompile(`^https://youtu\.be/[\w-]+(?:\?[\w=&.-]*)?$`)
	externalHTTPSPattern   = regexp.MustCompile(`^https://.*$`)
)

// externalURLMediaTypes are the media types that Gemini models download from
// arbitrary HTTPS URLs.
var externalURLMediaTypes = []string{
	"text/html",
	"text/css",
	"text/plain",
	"text/xml",
	"text/csv",
	"text/rtf",
	"text/javascript",
	"application/json",
	"application/pdf",
	"image/bmp",
	"image/jpeg",
	"image/png",
	"image/webp",
	"video/mp4",
	"video/mpeg",
	"video/quicktime",
	"video/avi",
	"video/x-flv",
	"video/mpg",
	"video/webm",
	"video/wmv",
	"video/3gpp",
}

// ChatModel is a Gemini model that uses the generateContent API. It is
// safe for concurrent use.
type ChatModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider name, "google.generative-ai" by default.
func (m *ChatModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model ID.
func (m *ChatModel) ModelID() string {
	return m.modelID
}

// SupportedURLs returns the URLs the model downloads itself: Google Files
// API files and YouTube videos for every media type, and HTTPS URLs of
// common document, image and video types for Gemini models after 2.0. Vertex
// AI models download every HTTP(S) and gs:// URL.
func (m *ChatModel) SupportedURLs() map[string][]*regexp.Regexp {
	if m.provider.isVertex() {
		return vertexSupportedURLs()
	}

	supported := map[string][]*regexp.Regexp{
		"*": m.provider.geminiFileURLPatterns(),
	}

	if !supportsExternalFileURLs(m.modelID) {
		return supported
	}

	for _, mediaType := range externalURLMediaTypes {
		supported[mediaType] = []*regexp.Regexp{
			externalHTTPSPattern,
		}
	}

	return supported
}

// geminiFileURLPatterns match the URLs every Gemini API model downloads
// itself: Google Files API files, including files below the base URL, and
// YouTube videos.
func (p *Provider) geminiFileURLPatterns() []*regexp.Regexp {
	baseFilesPattern := regexp.MustCompile("^" + regexp.QuoteMeta(p.baseURL+"/files/") + ".*$")

	return []*regexp.Regexp{
		googleFilesURLPattern,
		baseFilesPattern,
		youTubeWatchURLPattern,
		youTubeShortURLPattern,
	}
}

// Generate performs one generateContent call. Responses that reach the
// per-request output cap while budget remains finish with CONTINUATION; they
// are continued automatically and their content and usage are combined.
func (m *ChatModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	call, err := m.prepareCall(ctx, options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.request(ctx, modelPath(m.modelID)+":generateContent", call.requestHeaders(options.Headers))
	if err != nil {
		return nil, err
	}

	generation, err := m.generateWithContinuations(ctx, request, call)
	if err != nil {
		return nil, err
	}

	last, err := convertResponse(call, generation.last.response, generation.usage)
	if err != nil {
		return nil, err
	}

	if generation.isContinued() {
		last.metadata.InitialPromptTokenCount = generation.firstPromptTokenCount
		last.metadata.GroundingMetadataPerRequest = generation.groundingPerRequest
	}

	return &model.GenerateResult{
		Content:          append(generation.continuedContent, last.content...),
		FinishReason:     last.finishReason,
		Usage:            convertUsage(generation.usage),
		ProviderMetadata: wrapMetadata(call.metadataNamespaces, last.metadata),
		Warnings:         call.warnings,
		Request: model.RequestInfo{
			Body: generation.firstRequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID: generation.last.response.ResponseID,
			},
			Headers: generation.last.raw.Headers,
			Body:    generation.last.raw.Body,
		},
	}, nil
}

// continuedGeneration is the outcome of a request and its automatic
// continuations.
type continuedGeneration struct {
	firstRequestBody      []byte
	firstPromptTokenCount *int

	// continuedContent is the content of every response before the last.
	continuedContent []model.Content

	groundingPerRequest []*GroundingMetadata
	usage               *UsageMetadata
	last                generateExchange
}

func (g continuedGeneration) isContinued() bool {
	return len(g.groundingPerRequest) > 1
}

// generateWithContinuations resends the request with the continuation token
// until a response finishes for another reason. Usage is summed across
// requests, matching billing.
func (m *ChatModel) generateWithContinuations(ctx context.Context, request apicall.Request, call preparedCall) (continuedGeneration, error) {
	exchange, err := m.generateContent(ctx, request, call.body)
	if err != nil {
		return continuedGeneration{}, err
	}

	generation := continuedGeneration{
		firstRequestBody:      exchange.raw.RequestBody,
		firstPromptTokenCount: promptTokenCount(exchange.response.UsageMetadata),
	}

	for {
		candidate := exchange.response.firstCandidate()
		generation.groundingPerRequest = append(generation.groundingPerRequest, candidateGrounding(candidate))
		generation.usage = addUsageMetadata(generation.usage, exchange.response.UsageMetadata)

		token := continuationToken(candidate)
		if token == "" {
			generation.last = exchange

			return generation, nil
		}

		converted, err := call.newContentConverter().convert(candidate.parts())
		if err != nil {
			return continuedGeneration{}, err
		}

		generation.continuedContent = append(generation.continuedContent, converted...)

		continued := call.body
		continued.ContinuationToken = token

		exchange, err = m.generateContent(ctx, request, continued)
		if err != nil {
			return continuedGeneration{}, err
		}
	}
}

type generateExchange struct {
	raw      *apicall.Response
	response generateContentResponse
}

func (m *ChatModel) generateContent(ctx context.Context, request apicall.Request, body generateContentRequest) (generateExchange, error) {
	raw, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return generateExchange{}, err
	}

	var response generateContentResponse
	if err := apicall.DecodeJSON(raw.Body, &response); err != nil {
		return generateExchange{}, err
	}

	return generateExchange{
		raw:      raw,
		response: response,
	}, nil
}

type convertedResponse struct {
	content      []model.Content
	finishReason model.FinishReason
	metadata     ChatMetadata
}

// convertResponse converts the last response of a call. Usage is the sum
// over every request of the call.
func convertResponse(call preparedCall, response generateContentResponse, usage *UsageMetadata) (convertedResponse, error) {
	candidate := response.firstCandidate()

	contents, err := call.newContentConverter().convert(candidate.parts())
	if err != nil {
		return convertedResponse{}, err
	}

	if candidate != nil {
		for _, source := range extractSources(candidate.GroundingMetadata, call.generateID) {
			contents = append(contents, source)
		}
	}

	metadata := ChatMetadata{
		PromptFeedback: response.PromptFeedback,
		UsageMetadata:  usage,
	}

	if usage != nil {
		metadata.ServiceTier = usage.ServiceTier
	}

	if candidate != nil {
		metadata.GroundingMetadata = candidate.GroundingMetadata
		metadata.URLContextMetadata = candidate.URLContextMetadata
		metadata.SafetyRatings = candidate.SafetyRatings
		metadata.FinishMessage = candidate.FinishMessage
		metadata.ContinuationToken = candidate.ContinuationToken
	}

	return convertedResponse{
		content:      contents,
		finishReason: generateFinishReason(response, contents),
		metadata:     metadata,
	}, nil
}

// generateFinishReason maps the candidate's finish reason, or reports a
// content filter when the prompt was blocked before any candidate finished.
func generateFinishReason(response generateContentResponse, contents []model.Content) model.FinishReason {
	candidate := response.firstCandidate()

	raw := ""
	if candidate != nil {
		raw = candidate.FinishReason
	}

	var blockReason *string
	if response.PromptFeedback != nil {
		blockReason = response.PromptFeedback.BlockReason
	}

	if raw == "" && isConfirmedBlockReason(blockReason) {
		return model.FinishReason{
			Unified: model.FinishContentFilter,
			Raw:     *blockReason,
		}
	}

	return model.FinishReason{
		Unified: mapFinishReason(raw, hasClientToolCall(contents)),
		Raw:     raw,
	}
}

// hasClientToolCall reports tool calls that the application executes.
func hasClientToolCall(contents []model.Content) bool {
	for _, item := range contents {
		call, ok := item.(model.ToolCall)
		if ok && !call.ProviderExecuted {
			return true
		}
	}

	return false
}

func (c preparedCall) newContentConverter() *contentConverter {
	return &contentConverter{
		toolNames:            c.toolNames,
		jsonResponseToolName: c.jsonResponseToolName,
		generateID:           c.generateID,
		metadataNamespaces:   c.metadataNamespaces,
	}
}

// continuationToken returns the token of a candidate that must be
// continued, or an empty string.
func continuationToken(candidate *candidate) string {
	if candidate == nil || candidate.FinishReason != finishReasonContinuation {
		return ""
	}

	return candidate.ContinuationToken
}

func candidateGrounding(candidate *candidate) *GroundingMetadata {
	if candidate == nil {
		return nil
	}

	return candidate.GroundingMetadata
}

func promptTokenCount(usage *UsageMetadata) *int {
	if usage == nil {
		return nil
	}

	return usage.PromptTokenCount
}
