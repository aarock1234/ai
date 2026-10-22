package openaicompatible

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// completionTextID is the ID of the single text block of a completion stream.
const completionTextID = "0"

// completionStream converts completion chunks into stream parts.
type completionStream struct {
	includeRawChunks bool

	finishReason model.FinishReason
	usage        *completionUsage
	started      bool
}

// Process converts one server-sent event into stream parts.
func (s *completionStream) Process(event sse.Event) ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if s.includeRawChunks {
		if raw, ok := chatsse.Raw(event); ok {
			parts = append(parts, raw)
		}
	}

	var chunk completionResponse
	err := json.Unmarshal([]byte(event.Data), &chunk)
	if err == nil && chunk.Error == nil && chunk.Choices == nil {
		err = errMissingChoices
	}

	if err != nil {
		return append(parts, s.Fail(fmt.Errorf("%w: %w", model.ErrInvalidResponse, err))), nil
	}

	if chunk.Error != nil {
		return append(parts, s.Fail(chunkError(chunk.Error))), nil
	}

	if !s.started {
		s.started = true
		parts = append(parts,
			responseMetadata(chunk.ID, chunk.Model, chunk.Created),
			model.TextStart{
				ID: completionTextID,
			},
		)
	}

	if chunk.Usage != nil {
		s.usage = chunk.Usage
	}

	if len(chunk.Choices) == 0 {
		return parts, nil
	}

	choice := chunk.Choices[0]
	if choice.FinishReason != nil {
		s.finishReason = MapFinishReason(*choice.FinishReason)
	}

	if choice.Text != nil {
		parts = append(parts, model.TextDelta{
			ID:    completionTextID,
			Delta: *choice.Text,
		})
	}

	return parts, nil
}

// Fail records an error finish reason and returns the error part.
func (s *completionStream) Fail(err error) model.StreamPart {
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}

	return model.StreamError{
		Err: err,
	}
}

// Flush returns the parts that close the stream.
func (s *completionStream) Flush() ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if s.started {
		parts = append(parts, model.TextEnd{
			ID: completionTextID,
		})
	}

	return append(parts, model.Finish{
		Usage:        convertCompletionUsage(s.usage),
		FinishReason: s.finishReason,
	}), nil
}
