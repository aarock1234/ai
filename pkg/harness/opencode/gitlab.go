package opencode

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
)

// defaultGitLabAIGatewayURL is the GitLab AI gateway used when
// GITLAB_AI_GATEWAY_URL is unset.
const defaultGitLabAIGatewayURL = "https://cloud.gitlab.com"

// maxDirectAccessResponseBytes bounds a direct access response.
const maxDirectAccessResponseBytes = 1 << 20

// gitLabProviderID is an OpenCode provider the adapter configures for a
// brokered GitLab login, or the workflow provider, which cannot be brokered.
type gitLabProviderID string

// GitLab providers.
const (
	gitLabAnthropic       gitLabProviderID = "ai-sdk-gitlab-anthropic"
	gitLabOpenAIChat      gitLabProviderID = "ai-sdk-gitlab-openai-chat"
	gitLabOpenAIResponses gitLabProviderID = "ai-sdk-gitlab-openai-responses"
	gitLabWorkflow        gitLabProviderID = "workflow"
)

// gitLabModel is the provider and upstream model of a GitLab Duo model.
type gitLabModel struct {
	providerID gitLabProviderID
	modelID    string
}

// gitLabModels maps OpenCode's GitLab Duo models to the providers and
// models of the GitLab AI gateway.
var gitLabModels = map[string]gitLabModel{
	"duo-chat-fable-5": {
		providerID: gitLabAnthropic,
		modelID:    "claude-fable-5",
	},
	"duo-chat-opus-5": {
		providerID: gitLabAnthropic,
		modelID:    "claude-opus-5",
	},
	"duo-chat-opus-4-8": {
		providerID: gitLabAnthropic,
		modelID:    "claude-opus-4-8",
	},
	"duo-chat-opus-4-7": {
		providerID: gitLabAnthropic,
		modelID:    "claude-opus-4-7",
	},
	"duo-chat-opus-4-6": {
		providerID: gitLabAnthropic,
		modelID:    "claude-opus-4-6",
	},
	"duo-chat-sonnet-5": {
		providerID: gitLabAnthropic,
		modelID:    "claude-sonnet-5",
	},
	"duo-chat-sonnet-4-6": {
		providerID: gitLabAnthropic,
		modelID:    "claude-sonnet-4-6",
	},
	"duo-chat-opus-4-5": {
		providerID: gitLabAnthropic,
		modelID:    "claude-opus-4-5-20251101",
	},
	"duo-chat-sonnet-4-5": {
		providerID: gitLabAnthropic,
		modelID:    "claude-sonnet-4-5-20250929",
	},
	"duo-chat-haiku-4-5": {
		providerID: gitLabAnthropic,
		modelID:    "claude-haiku-4-5-20251001",
	},
	"duo-chat-gpt-5-1": {
		providerID: gitLabOpenAIChat,
		modelID:    "gpt-5.1-2025-11-13",
	},
	"duo-chat-gpt-5-2": {
		providerID: gitLabOpenAIChat,
		modelID:    "gpt-5.2-2025-12-11",
	},
	"duo-chat-gpt-5-4": {
		providerID: gitLabOpenAIChat,
		modelID:    "gpt-5.4-2026-03-05",
	},
	"duo-chat-gpt-5-5": {
		providerID: gitLabOpenAIChat,
		modelID:    "gpt-5.5-2026-04-23",
	},
	"duo-chat-gpt-5-mini": {
		providerID: gitLabOpenAIChat,
		modelID:    "gpt-5-mini-2025-08-07",
	},
	"duo-chat-gpt-5-4-mini": {
		providerID: gitLabOpenAIChat,
		modelID:    "gpt-5.4-mini",
	},
	"duo-chat-gpt-5-4-nano": {
		providerID: gitLabOpenAIChat,
		modelID:    "gpt-5.4-nano",
	},
	"duo-chat-gpt-5-6-sol": {
		providerID: gitLabOpenAIResponses,
		modelID:    "gpt-5.6-sol",
	},
	"duo-chat-gpt-5-6-terra": {
		providerID: gitLabOpenAIResponses,
		modelID:    "gpt-5.6-terra",
	},
	"duo-chat-gpt-5-6-luna": {
		providerID: gitLabOpenAIResponses,
		modelID:    "gpt-5.6-luna",
	},
	"duo-chat-gpt-5-codex": {
		providerID: gitLabOpenAIResponses,
		modelID:    "gpt-5-codex",
	},
	"duo-chat-gpt-5-2-codex": {
		providerID: gitLabOpenAIResponses,
		modelID:    "gpt-5.2-codex",
	},
	"duo-chat-gpt-5-3-codex": {
		providerID: gitLabOpenAIResponses,
		modelID:    "gpt-5.3-codex",
	},
	"duo-workflow": {
		providerID: gitLabWorkflow,
		modelID:    "default",
	},
	"duo-workflow-default": {
		providerID: gitLabWorkflow,
		modelID:    "default",
	},
	"duo-workflow-sonnet-4-5": {
		providerID: gitLabWorkflow,
		modelID:    "anthropic/claude-sonnet-4-5-20250929",
	},
	"duo-workflow-sonnet-5": {
		providerID: gitLabWorkflow,
		modelID:    "claude_sonnet_5",
	},
	"duo-workflow-opus-5": {
		providerID: gitLabWorkflow,
		modelID:    "claude_opus_5",
	},
	"duo-workflow-sonnet-4-6": {
		providerID: gitLabWorkflow,
		modelID:    "claude_sonnet_4_6",
	},
	"duo-workflow-opus-4-5": {
		providerID: gitLabWorkflow,
		modelID:    "anthropic/claude-opus-4-5-20251101",
	},
	"duo-workflow-haiku-4-5": {
		providerID: gitLabWorkflow,
		modelID:    "claude_haiku_4_5_20251001",
	},
	"duo-workflow-opus-4-6": {
		providerID: gitLabWorkflow,
		modelID:    "claude_opus_4_6_20260205",
	},
}

// gitLabDirectAccess is a short-lived credential of the GitLab AI gateway.
type gitLabDirectAccess struct {
	accessToken  string
	headers      map[string]string
	aiGatewayURL string
}

// directAccessRequest configures requestGitLabDirectAccess.
type directAccessRequest struct {
	accessToken  string
	instanceURL  string
	aiGatewayURL string
	client       *http.Client
}

// directAccessResponse is the part of a direct access response the adapter
// reads.
type directAccessResponse struct {
	Token   *string                    `json:"token"`
	Headers map[string]json.RawMessage `json:"headers"`
}

// requestGitLabDirectAccess exchanges a GitLab login for a direct access
// credential of the GitLab AI gateway.
func requestGitLabDirectAccess(ctx context.Context, request directAccessRequest) (gitLabDirectAccess, error) {
	body, err := json.Marshal(map[string]map[string]bool{
		"feature_flags": {
			"DuoAgentPlatformNext": true,
		},
	})
	if err != nil {
		return gitLabDirectAccess{}, fmt.Errorf("encode opencode gitlab direct access request: %w", err)
	}

	endpoint := strings.TrimRight(request.instanceURL, "/") + "/api/v4/ai/third_party_agents/direct_access"

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return gitLabDirectAccess{}, fmt.Errorf("create opencode gitlab direct access request: %w", err)
	}

	httpRequest.Header.Set("Authorization", "Bearer "+request.accessToken)
	httpRequest.Header.Set("Content-Type", "application/json")

	responseBody, err := sendDirectAccessRequest(httpRequest, request.client)
	if err != nil {
		return gitLabDirectAccess{}, err
	}

	return parseDirectAccess(responseBody, request.aiGatewayURL)
}

// sendDirectAccessRequest sends the request and returns the body of a
// successful response.
func sendDirectAccessRequest(request *http.Request, client *http.Client) ([]byte, error) {
	if client == nil {
		client = http.DefaultClient
	}

	response, err := client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("opencode gitlab direct access request: %w", err)
	}

	defer func() {
		_ = response.Body.Close()
	}()

	body, err := io.ReadAll(io.LimitReader(response.Body, maxDirectAccessResponseBytes))
	if err != nil {
		return nil, fmt.Errorf("read opencode gitlab direct access response: %w", err)
	}

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("opencode gitlab direct access request failed with status %d", response.StatusCode)
	}

	return body, nil
}

// parseDirectAccess validates a direct access response.
func parseDirectAccess(body []byte, aiGatewayURL string) (gitLabDirectAccess, error) {
	var response directAccessResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return gitLabDirectAccess{}, errors.New("opencode gitlab direct access returned invalid json")
	}

	invalidCredentials := errors.New("opencode gitlab direct access returned invalid credentials")

	if response.Token == nil || *response.Token == "" || response.Headers == nil {
		return gitLabDirectAccess{}, invalidCredentials
	}

	headers := make(map[string]string, len(response.Headers))

	for name, value := range response.Headers {
		var text string
		if err := json.Unmarshal(value, &text); err != nil {
			return gitLabDirectAccess{}, invalidCredentials
		}

		headers[name] = text
	}

	return gitLabDirectAccess{
		accessToken:  *response.Token,
		headers:      headers,
		aiGatewayURL: strings.TrimRight(aiGatewayURL, "/"),
	}, nil
}

// gitLabTransformations replace the placeholder of a GitLab login with the
// direct access credential on the gateway's Anthropic and OpenAI proxies.
func gitLabTransformations(access gitLabDirectAccess, placeholder string) ([]harness.RequestTransformation, error) {
	transformHeaders := make(map[string]string, len(access.headers)+1)

	for name, value := range access.headers {
		lowered := strings.ToLower(name)
		if lowered == "authorization" || lowered == "x-api-key" {
			continue
		}

		transformHeaders[name] = value
	}

	transformHeaders["Authorization"] = "Bearer " + access.accessToken

	proxies := []string{
		access.aiGatewayURL + "/ai/v1/proxy/anthropic",
		access.aiGatewayURL + "/ai/v1/proxy/openai/v1",
	}

	built := make([]harness.RequestTransformation, 0, len(proxies))

	for _, proxy := range proxies {
		transformation, err := harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
			MatchURL: proxy,
			MatchHeaders: map[string]string{
				"Authorization": "Bearer " + placeholder,
			},
			TransformHeaders: maps.Clone(transformHeaders),
		})
		if err != nil {
			return nil, err
		}

		built = append(built, transformation)
	}

	return built, nil
}

// gitLabProviderConfig is the OpenCode configuration of a GitLab gateway
// provider.
type gitLabProviderConfig struct {
	Name    string                          `json:"name"`
	NPM     string                          `json:"npm"`
	API     string                          `json:"api"`
	Options gitLabProviderOptions           `json:"options"`
	Models  map[string]gitLabModelReference `json:"models"`
}

// gitLabProviderOptions authenticate a GitLab gateway provider with the
// sandbox's placeholder, which the request transformations replace.
type gitLabProviderOptions struct {
	AuthToken string            `json:"authToken,omitempty"`
	APIKey    string            `json:"apiKey,omitempty"`
	Headers   map[string]string `json:"headers,omitempty"`
}

// gitLabModelReference names a gateway model in the provider configuration.
type gitLabModelReference struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// gitLabConfigOptions configure gitLabSubscriptionConfig.
type gitLabConfigOptions struct {
	openCodeConfig map[string]any
	placeholder    string
	aiGatewayURL   string
	headers        map[string]string
}

// gitLabSubscriptionConfig returns the OpenCode configuration with the
// GitLab gateway providers, which take precedence over configured providers
// of the same name.
func gitLabSubscriptionConfig(options gitLabConfigOptions) map[string]any {
	providers := make(map[string]any)

	if configured, ok := options.openCodeConfig["provider"].(map[string]any); ok {
		maps.Copy(providers, configured)
	}

	providers[string(gitLabAnthropic)] = gitLabProviderConfig{
		Name: "GitLab Anthropic",
		NPM:  "@ai-sdk/anthropic",
		API:  options.aiGatewayURL + "/ai/v1/proxy/anthropic/v1",
		Options: gitLabProviderOptions{
			AuthToken: options.placeholder,
			Headers:   options.headers,
		},
		Models: gitLabProviderModels(gitLabAnthropic),
	}

	providers[string(gitLabOpenAIChat)] = gitLabProviderConfig{
		Name: "GitLab OpenAI Chat",
		NPM:  "@ai-sdk/openai-compatible",
		API:  options.aiGatewayURL + "/ai/v1/proxy/openai/v1",
		Options: gitLabProviderOptions{
			APIKey:  options.placeholder,
			Headers: options.headers,
		},
		Models: gitLabProviderModels(gitLabOpenAIChat),
	}

	providers[string(gitLabOpenAIResponses)] = gitLabProviderConfig{
		Name: "GitLab OpenAI Responses",
		NPM:  "@ai-sdk/openai",
		API:  options.aiGatewayURL + "/ai/v1/proxy/openai/v1",
		Options: gitLabProviderOptions{
			APIKey:  options.placeholder,
			Headers: options.headers,
		},
		Models: gitLabProviderModels(gitLabOpenAIResponses),
	}

	config := maps.Clone(options.openCodeConfig)
	if config == nil {
		config = make(map[string]any)
	}

	config["provider"] = providers

	return config
}

// gitLabProviderModels returns the models of a gateway provider, keyed by
// their GitLab Duo names.
func gitLabProviderModels(providerID gitLabProviderID) map[string]gitLabModelReference {
	models := make(map[string]gitLabModelReference)

	for name, entry := range gitLabModels {
		if entry.providerID != providerID {
			continue
		}

		models[name] = gitLabModelReference{
			ID:   entry.modelID,
			Name: name,
		}
	}

	return models
}

// gitLabSubscriptionModel maps a GitLab Duo model to the gateway provider
// the brokered login uses. Models of other providers pass through.
func gitLabSubscriptionModel(modelName string, provider string) (string, error) {
	if modelName == "" {
		if provider == string(resolvedGitLab) {
			return "", unsupported("Harness 'opencode' requires an explicit GitLab model when native GitLab subscription credentials are brokered.")
		}

		return "", nil
	}

	modelProvider := provider
	modelID := modelName

	if prefix, rest, found := strings.Cut(modelName, "/"); found {
		modelProvider = prefix
		modelID = rest
	}

	if modelProvider != string(resolvedGitLab) {
		return modelName, nil
	}

	mapping, ok := gitLabModels[modelID]
	if !ok {
		return "", unsupported(fmt.Sprintf(
			"Harness 'opencode' cannot broker native GitLab subscription credentials for unknown model '%s'.",
			modelID,
		))
	}

	if mapping.providerID == gitLabWorkflow {
		return "", unsupported(fmt.Sprintf(
			"Harness 'opencode' cannot broker native GitLab subscription credentials for workflow model '%s' because it authenticates over WebSocket. Use a sandbox without credential brokering to use the credential-forwarding fallback.",
			modelID,
		))
	}

	return string(mapping.providerID) + "/" + modelID, nil
}
