package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/ai/ui"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	instructions    = "You are a helpful assistant. Answer concisely."
	maxRequestBytes = 1 << 20
)

// chatRequest is the request body that useChat sends. Other fields, such as
// the chat ID and trigger, are not needed here.
type chatRequest struct {
	Messages []ui.Message `json:"messages"`
}

// chatHandler answers a chat conversation with a streamed UI message.
type chatHandler struct {
	model model.LanguageModel
}

func (h *chatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	messages, err := readMessages(w, r)
	if err != nil {
		slog.WarnContext(ctx, "rejecting chat request", slog.Any("error", err))
		http.Error(w, "invalid chat request", http.StatusBadRequest)

		return
	}

	prompt, err := ui.ConvertToModelMessages(ctx, messages, ui.ConvertToModelMessagesOptions{})
	if err != nil {
		slog.WarnContext(ctx, "rejecting chat messages", slog.Any("error", err))
		http.Error(w, "invalid chat messages", http.StatusBadRequest)

		return
	}

	// The request context ends the generation when the client disconnects.
	stream, err := ai.StreamText(ctx, ai.StreamTextOptions{
		Model:    h.model,
		System:   instructions,
		Messages: prompt,
	})
	if err != nil {
		slog.ErrorContext(ctx, "starting chat stream", slog.Any("error", err))
		http.Error(w, "chat unavailable", http.StatusInternalServerError)

		return
	}
	defer closeStream(ctx, stream)

	err = ui.PipeMessageStreamToResponse(w, ui.MessageStreamResponseOptions{
		Stream: ui.ToMessageStream(ctx, stream.FullStream(), ui.MessageStreamOptions{}),
	})
	if err != nil {
		slog.WarnContext(ctx, "chat stream ended early", slog.Any("error", err))
	}
}

// closeStream releases the chat stream and logs a failure to close it.
func closeStream(ctx context.Context, stream io.Closer) {
	if err := stream.Close(); err != nil {
		slog.WarnContext(ctx, "closing chat stream", slog.Any("error", err))
	}
}

// readMessages decodes and validates the UI messages of a chat request.
func readMessages(w http.ResponseWriter, r *http.Request) ([]ui.Message, error) {
	body := http.MaxBytesReader(w, r.Body, maxRequestBytes)

	var request chatRequest
	if err := json.NewDecoder(body).Decode(&request); err != nil {
		return nil, fmt.Errorf("decode request: %w", err)
	}

	messages, err := ui.ValidateMessages(r.Context(), ui.ValidateMessagesOptions{
		Messages: request.Messages,
	})
	if err != nil {
		return nil, fmt.Errorf("validate messages: %w", err)
	}

	return messages, nil
}
