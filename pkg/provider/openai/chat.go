package openai

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Compile-time check that ChatModel implements model.Model.
var _ model.Model = (*ChatModel)(nil)

// ChatModel is an OpenAI chat completion model that implements
// model.Model.
type ChatModel struct {
	modelID      string
	clientConfig *ClientConfig
}

// Provider returns "openai".
func (m *ChatModel) Provider() string { return "openai" }

// ModelID returns the model identifier (e.g. "gpt-4o").
func (m *ChatModel) ModelID() string { return m.modelID }

// DoGenerate sends a non-streaming chat completion request and returns
// the normalized response.
func (m *ChatModel) DoGenerate(ctx context.Context, req *model.Request) (*model.Response, error) {
	chatReq, err := EncodeRequest(m.modelID, req, false)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	chatResp, err := DoRequest(ctx, m.clientConfig, chatReq)
	if err != nil {
		return nil, err
	}

	return DecodeResponse(chatResp), nil
}

// DoStream sends a streaming chat completion request and returns a
// StreamResponse whose iterator yields chunks as they arrive. The
// HTTP response body is closed automatically when the stream is fully
// consumed or the caller breaks out of the range loop.
func (m *ChatModel) DoStream(ctx context.Context, req *model.Request) (*model.StreamResponse, error) {
	chatReq, err := EncodeRequest(m.modelID, req, true)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	body, err := DoStreamRequest(ctx, m.clientConfig, chatReq)
	if err != nil {
		return nil, err
	}

	raw := DecodeStream(body)

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
