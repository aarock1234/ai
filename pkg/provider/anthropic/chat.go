package anthropic

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Compile-time check that ChatModel implements model.Model.
var _ model.Model = (*ChatModel)(nil)

// ChatModel is an Anthropic Messages API model that implements
// model.Model.
type ChatModel struct {
	modelID      string
	clientConfig *ClientConfig
}

// Provider returns "anthropic".
func (m *ChatModel) Provider() string { return "anthropic" }

// ModelID returns the model identifier.
func (m *ChatModel) ModelID() string { return m.modelID }

// SupportsResponseFormat reports whether this model supports native structured output.
func (m *ChatModel) SupportsResponseFormat(_ *model.ResponseFormat) bool { return false }

// DoGenerate sends a non-streaming Messages API request and returns the
// normalized response.
func (m *ChatModel) DoGenerate(ctx context.Context, req *model.Request) (*model.Response, error) {
	msgReq, err := EncodeRequest(m.modelID, req, false)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	msgResp, err := DoRequest(ctx, m.clientConfig, msgReq)
	if err != nil {
		return nil, err
	}

	return DecodeResponse(msgResp), nil
}

// DoStream sends a streaming Messages API request and returns a
// StreamResponse whose iterator yields chunks as they arrive. The HTTP
// response body is closed automatically when the stream is fully
// consumed or the caller breaks out of the range loop.
func (m *ChatModel) DoStream(ctx context.Context, req *model.Request) (*model.StreamResponse, error) {
	msgReq, err := EncodeRequest(m.modelID, req, true)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	body, err := DoStreamRequest(ctx, m.clientConfig, msgReq)
	if err != nil {
		return nil, err
	}

	raw := DecodeStream(body)
	wrapped := func(yield func(model.Chunk) bool) {
		defer func() { _ = body.Close() }()

		for chunk := range raw {
			if !yield(chunk) {
				return
			}
		}
	}

	return &model.StreamResponse{
		Stream:   wrapped,
		Metadata: model.ProviderMetadata{ModelID: m.modelID},
	}, nil
}
