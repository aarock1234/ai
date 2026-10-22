package middleware

import (
	"context"
	"iter"
	"strconv"

	"github.com/aarock1234/ai/pkg/model"
)

// SimulateStreaming returns middleware that serves streaming calls with a
// non-streaming call, replaying its content as stream parts.
func SimulateStreaming() Language {
	return Language{
		WrapStream: func(ctx context.Context, call LanguageCall) (*model.StreamResult, error) {
			result, err := call.DoGenerate(ctx)
			if err != nil {
				return nil, err
			}

			stream := model.NewStreamResult(simulatedParts(result), nil)
			stream.Request = result.Request
			stream.Response = model.StreamResponseInfo{
				Headers: result.Response.Headers,
			}

			return stream, nil
		},
	}
}

// simulatedParts replays a generate result: stream start, response metadata,
// one start/delta/end block per text and reasoning item, the other content
// as is, and the finish part.
func simulatedParts(result *model.GenerateResult) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		parts := []model.StreamPart{
			model.StreamStart{
				Warnings: result.Warnings,
			},
			result.Response.ResponseMetadata,
		}

		nextID := 0
		for _, content := range result.Content {
			blockID := strconv.Itoa(nextID)

			switch typed := content.(type) {
			case model.Text:
				if typed.Text == "" && len(typed.Citations) == 0 {
					continue
				}

				parts = append(parts, simulatedText(blockID, typed)...)
				nextID++

			case model.Reasoning:
				parts = append(parts, simulatedReasoning(blockID, typed)...)
				nextID++

			case model.StreamPart:
				parts = append(parts, typed)
			}
		}

		parts = append(parts, model.Finish{
			Usage:            result.Usage,
			FinishReason:     result.FinishReason,
			ProviderMetadata: result.ProviderMetadata,
		})

		for _, part := range parts {
			if !yield(part) {
				return
			}
		}
	}
}

func simulatedText(blockID string, text model.Text) []model.StreamPart {
	return []model.StreamPart{
		model.TextStart{
			ID:               blockID,
			ProviderMetadata: text.ProviderMetadata,
		},
		model.TextDelta{
			ID:    blockID,
			Delta: text.Text,
		},
		model.TextEnd{
			ID:        blockID,
			Citations: text.Citations,
		},
	}
}

func simulatedReasoning(blockID string, reasoning model.Reasoning) []model.StreamPart {
	return []model.StreamPart{
		model.ReasoningStart{
			ID:               blockID,
			ProviderMetadata: reasoning.ProviderMetadata,
		},
		model.ReasoningDelta{
			ID:    blockID,
			Delta: reasoning.Text,
		},
		model.ReasoningEnd{
			ID: blockID,
		},
	}
}
