package openrouter

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
	oai "github.com/aarock1234/ai/pkg/provider/openai"
)

// Compile-time check that ChatModel implements model.Model.
var _ model.Model = (*ChatModel)(nil)

// ChatModel is an OpenRouter chat completion model that implements
// model.Model. It delegates to the OpenAI codec since OpenRouter
// uses the OpenAI-compatible wire format.
type ChatModel struct {
	modelID      string
	clientConfig *oai.ClientConfig
}

// Provider returns "openrouter".
func (m *ChatModel) Provider() string { return "openrouter" }

// ModelID returns the model identifier (e.g. "openai/gpt-4o").
func (m *ChatModel) ModelID() string { return m.modelID }

// DoGenerate sends a non-streaming chat completion request and returns
// the normalized response.
func (m *ChatModel) DoGenerate(ctx context.Context, req *model.Request) (*model.Response, error) {
	chatReq, err := oai.EncodeRequest(m.modelID, req, false)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	chatResp, err := oai.DoRequest(ctx, m.clientConfig, chatReq)
	if err != nil {
		return nil, err
	}

	return oai.DecodeResponse(chatResp), nil
}

// DoStream sends a streaming chat completion request and returns a
// StreamResponse whose iterator yields chunks as they arrive. The
// HTTP response body is closed automatically when the stream is fully
// consumed or the caller breaks out of the range loop.
func (m *ChatModel) DoStream(ctx context.Context, req *model.Request) (*model.StreamResponse, error) {
	chatReq, err := oai.EncodeRequest(m.modelID, req, true)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	body, err := oai.DoStreamRequest(ctx, m.clientConfig, chatReq)
	if err != nil {
		return nil, err
	}

	raw := oai.DecodeStream(body)

	// Wrap the iterator to close the response body when done.
	wrapped := func(yield func(model.Chunk) bool) {
		defer func() { _ = body.Close() }()

		for chunk := range raw {
			if !yield(chunk) {
				return
			}
		}
	}

	return &model.StreamResponse{Stream: wrapped}, nil
}
