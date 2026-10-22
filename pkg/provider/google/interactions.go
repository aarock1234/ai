package google

import (
	"context"
	"fmt"
	"maps"
	"net/http"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// serviceTierHeader reports the applied service tier on some responses.
const serviceTierHeader = "X-Gemini-Service-Tier"

var _ model.LanguageModel = (*InteractionsModel)(nil)

// InteractionsAgent names a Gemini agent of the Interactions API.
type InteractionsAgent string

// Gemini agents.
const (
	InteractionsDeepResearchPro    InteractionsAgent = "deep-research-pro-preview-12-2025"
	InteractionsDeepResearch       InteractionsAgent = "deep-research-preview-04-2026"
	InteractionsDeepResearchMax    InteractionsAgent = "deep-research-max-preview-04-2026"
	InteractionsAntigravityPreview InteractionsAgent = "antigravity-preview-05-2026"
)

// interactionsSupportedURLs are the URLs the Interactions API downloads
// itself.
var interactionsSupportedURLs = map[string][]*regexp.Regexp{
	"image/*": {
		regexp.MustCompile(`^https?://.+`),
	},
	"application/pdf": {
		regexp.MustCompile(`^https?://.+`),
	},
	"audio/*": {
		regexp.MustCompile(`^https?://.+`),
	},
	"video/*": {
		regexp.MustCompile(`^https?://(www\.)?youtube\.com/watch\?v=.+`),
		regexp.MustCompile(`^https?://youtu\.be/.+`),
		regexp.MustCompile(`^gs://.+`),
	},
}

// InteractionsModel is a language model of the Gemini Interactions API
// (POST /interactions): a Gemini model, a Gemini agent such as deep
// research, or a managed agent created with the agents endpoint. Agent and
// background calls run on the server; Generate polls them until they end,
// and Stream follows their event stream. It is safe for concurrent use.
type InteractionsModel struct {
	modelID string

	// agent is set for agent calls, which send agent instead of model.
	agent string

	provider *Provider
}

// Interactions returns an Interactions API language model for a Gemini
// model ID.
func (p *Provider) Interactions(modelID string) *InteractionsModel {
	return &InteractionsModel{
		modelID:  modelID,
		provider: p,
	}
}

// InteractionsAgent returns an Interactions API language model that runs a
// Gemini agent. Agent calls may require InteractionsOptions.Background.
func (p *Provider) InteractionsAgent(agent InteractionsAgent) *InteractionsModel {
	return &InteractionsModel{
		modelID:  string(agent),
		agent:    string(agent),
		provider: p,
	}
}

// InteractionsManagedAgent returns an Interactions API language model that
// runs a user-defined agent created with the agents endpoint.
func (p *Provider) InteractionsManagedAgent(name string) *InteractionsModel {
	return &InteractionsModel{
		modelID:  name,
		agent:    name,
		provider: p,
	}
}

// Provider returns the provider identifier,
// "google.generative-ai.interactions" by default.
func (m *InteractionsModel) Provider() string {
	return m.provider.name + ".interactions"
}

// ModelID returns the model ID, or the agent name.
func (m *InteractionsModel) ModelID() string {
	return m.modelID
}

// SupportedURLs returns HTTP(S) URLs of images, PDFs and audio, and
// YouTube and Cloud Storage URLs of videos.
func (m *InteractionsModel) SupportedURLs() map[string][]*regexp.Regexp {
	return maps.Clone(interactionsSupportedURLs)
}

// Generate performs one interaction. Agent and background interactions
// that are still running are polled until they end.
func (m *InteractionsModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	call, err := m.prepareInteraction(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.request(ctx, "interactions", options.Headers)
	if err != nil {
		return nil, err
	}

	raw, err := apicall.PostJSON(ctx, request, call.body)
	if err != nil {
		return nil, err
	}

	response, err := decodeInteraction(raw.Body)
	if err != nil {
		return nil, err
	}

	headers := raw.Headers
	body := raw.Body

	if (call.isAgent || call.isBackground) && !isTerminalInteraction(response.Status) {
		polled, err := m.pollInteraction(ctx, options.Headers, nonEmptyID(response.ID), call.pollTimeout)
		if err != nil {
			return nil, err
		}

		response = polled.response
		body = polled.raw.Body

		if polled.raw.Headers != nil {
			headers = polled.raw.Headers
		}
	}

	return m.generateResult(response, call, raw.RequestBody, headers, body)
}

func (m *InteractionsModel) generateResult(
	response interactionsResponse,
	call preparedInteraction,
	requestBody []byte,
	headers http.Header,
	body []byte,
) (*model.GenerateResult, error) {
	interactionID := nonEmptyID(response.ID)

	content, hasFunctionCall, err := parseInteractionsOutputs(response.Steps, m.provider.generateID, interactionID)
	if err != nil {
		return nil, err
	}

	usage, decodedUsage, err := convertInteractionsUsage(response.Usage)
	if err != nil {
		return nil, err
	}

	serviceTier := valueOrEmpty(response.ServiceTier)
	if serviceTier == "" {
		serviceTier = headers.Get(serviceTierHeader)
	}

	return &model.GenerateResult{
		Content:      content,
		FinishReason: interactionsFinishReason(response.Status, hasFunctionCall),
		Usage:        usage,
		Warnings:     call.warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: InteractionsMetadata{
				InteractionID:          interactionID,
				ServiceTier:            serviceTier,
				OutputTokensByModality: outputTokensByModality(decodedUsage),
			},
		},
		Request: model.RequestInfo{
			Body: requestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				ID:        interactionID,
				Timestamp: interactionsTimestamp(response.Created),
				ModelID:   valueOrEmpty(response.Model),
			},
			Headers: headers,
			Body:    body,
		},
	}, nil
}

// Stream performs one streaming interaction. Background interactions,
// which cannot stream when they are created, are created first and then
// followed through their event stream.
func (m *InteractionsModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	call, err := m.prepareInteraction(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.request(ctx, "interactions", options.Headers)
	if err != nil {
		return nil, err
	}

	if call.isBackground {
		return m.streamBackground(ctx, request, options.Headers, call, options.IncludeRawChunks)
	}

	body := call.body
	body.Stream = true

	response, err := apicall.PostJSONStream(ctx, request, body)
	if err != nil {
		return nil, err
	}

	state := newInteractionsStreamState(interactionsStreamSetup{
		warnings:         call.warnings,
		generateID:       m.provider.generateID,
		includeRawChunks: options.IncludeRawChunks,
		serviceTier:      response.Headers.Get(serviceTierHeader),
	})

	events := bodyEvents(ctx, response.Body)

	result := model.NewStreamResult(state.parts(events), response.Body.Close)
	result.Request = model.RequestInfo{
		Body: response.RequestBody,
	}
	result.Response = model.StreamResponseInfo{
		Headers: response.Headers,
	}

	return result, nil
}

// decodeInteraction decodes an interaction and checks its status.
func decodeInteraction(body []byte) (interactionsResponse, error) {
	var response interactionsResponse
	if err := apicall.DecodeJSON(body, &response); err != nil {
		return interactionsResponse{}, err
	}

	if !isInteractionStatus(response.Status) {
		return interactionsResponse{}, fmt.Errorf("%w: unknown interaction status %q", model.ErrInvalidResponse, response.Status)
	}

	return response, nil
}

func isInteractionStatus(status string) bool {
	return slices.Contains(interactionStatuses, status)
}
