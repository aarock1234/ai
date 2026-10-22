package baseten

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

// Default model IDs. Dedicated deployments ignore the model ID.
const (
	defaultChatModelID       = "chat"
	defaultDeploymentModelID = "placeholder"
)

// ChatModel is a Baseten chat model. It sends OpenAI-compatible chat
// completions with structured outputs and reports usage in streams.
type ChatModel struct {
	*openaicompatible.ChatModel

	provider string
}

// Provider returns the provider identifier, "baseten.chat" by default.
func (m *ChatModel) Provider() string {
	return m.provider
}

// Chat returns a chat model. Without a model URL it uses the Model APIs,
// where model IDs such as "deepseek-ai/DeepSeek-V3.1" select the model. With
// a "/sync/v1" model URL it uses that dedicated deployment, and model URLs
// ending in "/predict" are rejected. An empty modelID uses a default.
func (p *Provider) Chat(modelID string) (*ChatModel, error) {
	baseURL := p.baseURL
	defaultModelID := defaultChatModelID

	switch {
	case p.modelURL != "" && strings.Contains(p.modelURL, "/sync/v1"):
		baseURL = p.modelURL
		defaultModelID = defaultDeploymentModelID

	case p.modelURL != "" && strings.Contains(p.modelURL, "/predict"):
		return nil, fmt.Errorf("%w: you must use a /sync/v1 endpoint for chat models", model.ErrUnsupported)
	}

	if modelID == "" {
		modelID = defaultModelID
	}

	chat := openaicompatible.NewChatModel(modelID, openaicompatible.ChatConfig{
		Provider: Namespace + ".chat",
		BuildURL: func(path string) (string, error) {
			return baseURL + path, nil
		},
		Headers:                   p.requestHeaders,
		HTTPClient:                p.httpClient,
		ParseError:                parseError,
		IncludeUsage:              true,
		SupportsStructuredOutputs: true,
		GenerateID:                p.generateID,
	})

	return &ChatModel{
		ChatModel: chat,
		provider:  p.name + ".chat",
	}, nil
}
