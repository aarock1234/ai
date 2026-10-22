package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// RealtimeToolDefinitions declares the function and dynamic tools for a
// realtime session, in declaration order. Provider tools are skipped:
// realtime sessions only call functions.
func RealtimeToolDefinitions(tools []Tool) ([]model.RealtimeToolDefinition, error) {
	if err := validateTools(tools); err != nil {
		return nil, err
	}

	var definitions []model.RealtimeToolDefinition

	for _, tool := range tools {
		if tool.kind() == ToolKindProvider {
			continue
		}

		definitions = append(definitions, model.RealtimeToolDefinition{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.InputSchema,
		})
	}

	return definitions, nil
}

// RealtimeSetupResponse is what a server hands to a client that connects
// to a realtime model directly: a short-lived token, the WebSocket URL and
// the tools to declare. Its JSON form matches the upstream setup endpoint.
type RealtimeSetupResponse struct {
	Token string `json:"token"`
	URL   string `json:"url"`

	// ExpiresAt is in Unix seconds, or zero when unknown.
	ExpiresAt int64 `json:"expiresAt,omitempty"`

	Tools []RealtimeSetupTool `json:"tools,omitempty"`
}

// RealtimeSetupTool is the JSON form of a realtime tool definition.
type RealtimeSetupTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// CreateRealtimeSetupOptions configures CreateRealtimeSetup.
type CreateRealtimeSetupOptions struct {
	// Model must implement model.RealtimeClientSecretCreator.
	Model model.RealtimeModel

	// ExpiresAfter is the lifetime of the token, or zero for the provider
	// default.
	ExpiresAfter time.Duration

	// SessionConfig is embedded in the token request when set.
	SessionConfig *model.RealtimeSessionConfig

	// Tools are declared to the client. Their Execute functions stay on
	// the server.
	Tools []Tool
}

// CreateRealtimeSetup creates a short-lived client secret for a realtime
// model, for a server endpoint that lets clients connect directly without
// exposing the API key. The realtime API is experimental and may change.
func CreateRealtimeSetup(ctx context.Context, options CreateRealtimeSetupOptions) (*RealtimeSetupResponse, error) {
	if options.Model == nil {
		return nil, &InvalidArgumentError{
			Parameter: "model",
			Message:   "model is required",
		}
	}

	creator, ok := options.Model.(model.RealtimeClientSecretCreator)
	if !ok {
		return nil, fmt.Errorf(
			"%w: the %s realtime model %q does not issue client secrets",
			model.ErrUnsupported,
			options.Model.Provider(),
			options.Model.ModelID(),
		)
	}

	definitions, err := RealtimeToolDefinitions(options.Tools)
	if err != nil {
		return nil, err
	}

	secret, err := creator.CreateClientSecret(ctx, model.RealtimeClientSecretOptions{
		ExpiresAfter:  options.ExpiresAfter,
		SessionConfig: options.SessionConfig,
	})
	if err != nil {
		return nil, err
	}

	return &RealtimeSetupResponse{
		Token:     secret.Token,
		URL:       secret.URL,
		ExpiresAt: unixSeconds(secret.ExpiresAt),
		Tools:     realtimeSetupTools(definitions),
	}, nil
}

func realtimeSetupTools(definitions []model.RealtimeToolDefinition) []RealtimeSetupTool {
	if len(definitions) == 0 {
		return nil
	}

	tools := make([]RealtimeSetupTool, len(definitions))
	for index, definition := range definitions {
		tools[index] = RealtimeSetupTool{
			Type:        "function",
			Name:        definition.Name,
			Description: definition.Description,
			Parameters:  definition.Parameters,
		}
	}

	return tools
}

func unixSeconds(moment time.Time) int64 {
	if moment.IsZero() {
		return 0
	}

	return moment.Unix()
}
