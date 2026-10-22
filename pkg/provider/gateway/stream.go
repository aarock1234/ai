package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"time"

	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// Stream part types that are not content.
const (
	streamTextStart        = "text-start"
	streamTextDelta        = "text-delta"
	streamTextEnd          = "text-end"
	streamReasoningStart   = "reasoning-start"
	streamReasoningDelta   = "reasoning-delta"
	streamReasoningEnd     = "reasoning-end"
	streamToolInputStart   = "tool-input-start"
	streamToolInputDelta   = "tool-input-delta"
	streamToolInputEnd     = "tool-input-end"
	streamStart            = "stream-start"
	streamResponseMetadata = "response-metadata"
	streamFinish           = "finish"
	streamRaw              = "raw"
	streamError            = "error"
)

// doneMarker is a terminator some event streams send after the last event.
const doneMarker = "[DONE]"

// errStreamFailed reports an error part sent by the gateway.
var errStreamFailed = errors.New("gateway stream failed")

type wireBlock struct {
	ID               string       `json:"id"`
	ProviderMetadata wireMetadata `json:"providerMetadata"`
}

type wireDelta struct {
	ID               string       `json:"id"`
	Delta            string       `json:"delta"`
	ProviderMetadata wireMetadata `json:"providerMetadata"`
}

type wireTextEnd struct {
	ID               string         `json:"id"`
	Citations        []wireCitation `json:"citations"`
	ProviderMetadata wireMetadata   `json:"providerMetadata"`
}

type wireToolInputStart struct {
	ID               string                  `json:"id"`
	ToolName         string                  `json:"toolName"`
	InputEncoding    model.ToolInputEncoding `json:"inputEncoding"`
	ProviderMetadata wireMetadata            `json:"providerMetadata"`
	ProviderExecuted bool                    `json:"providerExecuted"`
	Dynamic          bool                    `json:"dynamic"`
	Title            string                  `json:"title"`
}

type wireStreamStart struct {
	Warnings []wireWarning `json:"warnings"`
}

type wireResponseMetadata struct {
	ID        string `json:"id"`
	Timestamp string `json:"timestamp"`
	ModelID   string `json:"modelId"`
}

type wireFinish struct {
	Usage            wireUsage        `json:"usage"`
	FinishReason     wireFinishReason `json:"finishReason"`
	ProviderMetadata wireMetadata     `json:"providerMetadata"`
}

type wireRaw struct {
	RawValue json.RawMessage `json:"rawValue"`
}

type wireStreamError struct {
	Error json.RawMessage `json:"error"`
}

type errorMessage struct {
	Message *string `json:"message"`
}

// streamParts reads the gateway's event stream. Each event holds one stream
// part in the specification's JSON form. Raw parts are dropped unless the
// call asked for them.
func streamParts(ctx context.Context, body io.Reader, includeRawChunks bool) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		for event, err := range sse.Read(body) {
			if err != nil {
				yield(model.StreamError{
					Err: readError(ctx, err),
				})

				return
			}

			if event.Data == doneMarker {
				continue
			}

			part, err := decodeStreamPart(json.RawMessage(event.Data))
			if err != nil {
				yield(model.StreamError{
					Err: err,
				})

				return
			}

			if _, isRaw := part.(model.Raw); isRaw && !includeRawChunks {
				continue
			}

			if !yield(part) {
				return
			}
		}
	}
}

// readError reports the context error when ctx ended the read.
func readError(ctx context.Context, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}

	return fmt.Errorf("read stream: %w", err)
}

// decodeStreamPart converts one stream part by its type.
func decodeStreamPart(raw json.RawMessage) (model.StreamPart, error) {
	header, err := decodeAs[typeHeader](raw)
	if err != nil {
		return nil, err
	}

	switch header.Type {
	case streamTextStart:
		return asStreamPart(decodeWith(raw, wireBlock.textStart))
	case streamTextDelta:
		return asStreamPart(decodeWith(raw, wireDelta.textDelta))
	case streamTextEnd:
		return asStreamPart(decodeWith(raw, wireTextEnd.part))
	case streamReasoningStart:
		return asStreamPart(decodeWith(raw, wireBlock.reasoningStart))
	case streamReasoningDelta:
		return asStreamPart(decodeWith(raw, wireDelta.reasoningDelta))
	case streamReasoningEnd:
		return asStreamPart(decodeWith(raw, wireBlock.reasoningEnd))
	case streamToolInputStart:
		return asStreamPart(decodeWith(raw, wireToolInputStart.part))
	case streamToolInputDelta:
		return asStreamPart(decodeWith(raw, wireDelta.toolInputDelta))
	case streamToolInputEnd:
		return asStreamPart(decodeWith(raw, wireBlock.toolInputEnd))
	default:
		return decodeLifecyclePart(header.Type, raw)
	}
}

// decodeLifecyclePart converts stream start, metadata, finish, raw and error
// parts, and complete content parts.
func decodeLifecyclePart(partType string, raw json.RawMessage) (model.StreamPart, error) {
	switch partType {
	case streamStart:
		return asStreamPart(decodeWith(raw, wireStreamStart.part))
	case streamResponseMetadata:
		return asStreamPart(decodeWith(raw, wireResponseMetadata.part))
	case streamFinish:
		return asStreamPart(decodeWith(raw, wireFinish.part))
	case streamRaw:
		return asStreamPart(decodeWith(raw, wireRaw.part))
	case streamError:
		return asStreamPart(decodeWith(raw, wireStreamError.part))
	default:
		return asStreamPart(decodeSharedContent(partType, raw))
	}
}

// asStreamPart returns a decoded value as a stream part, or the decoding
// error.
func asStreamPart[Value model.StreamPart](value Value, err error) (model.StreamPart, error) {
	if err != nil {
		return nil, err
	}

	return value, nil
}

func (b wireBlock) textStart() (model.TextStart, error) {
	return model.TextStart{
		ID:               b.ID,
		ProviderMetadata: b.ProviderMetadata.providerMetadata(),
	}, nil
}

func (b wireBlock) reasoningStart() (model.ReasoningStart, error) {
	return model.ReasoningStart{
		ID:               b.ID,
		ProviderMetadata: b.ProviderMetadata.providerMetadata(),
	}, nil
}

func (b wireBlock) reasoningEnd() (model.ReasoningEnd, error) {
	return model.ReasoningEnd{
		ID:               b.ID,
		ProviderMetadata: b.ProviderMetadata.providerMetadata(),
	}, nil
}

func (b wireBlock) toolInputEnd() (model.ToolInputEnd, error) {
	return model.ToolInputEnd{
		ID:               b.ID,
		ProviderMetadata: b.ProviderMetadata.providerMetadata(),
	}, nil
}

func (d wireDelta) textDelta() (model.TextDelta, error) {
	return model.TextDelta{
		ID:               d.ID,
		Delta:            d.Delta,
		ProviderMetadata: d.ProviderMetadata.providerMetadata(),
	}, nil
}

func (d wireDelta) reasoningDelta() (model.ReasoningDelta, error) {
	return model.ReasoningDelta{
		ID:               d.ID,
		Delta:            d.Delta,
		ProviderMetadata: d.ProviderMetadata.providerMetadata(),
	}, nil
}

func (d wireDelta) toolInputDelta() (model.ToolInputDelta, error) {
	return model.ToolInputDelta{
		ID:               d.ID,
		Delta:            d.Delta,
		ProviderMetadata: d.ProviderMetadata.providerMetadata(),
	}, nil
}

func (e wireTextEnd) part() (model.TextEnd, error) {
	return model.TextEnd{
		ID:               e.ID,
		Citations:        decodeCitations(e.Citations),
		ProviderMetadata: e.ProviderMetadata.providerMetadata(),
	}, nil
}

func (s wireToolInputStart) part() (model.ToolInputStart, error) {
	return model.ToolInputStart{
		ID:               s.ID,
		ToolName:         s.ToolName,
		InputEncoding:    s.InputEncoding,
		ProviderExecuted: s.ProviderExecuted,
		Dynamic:          s.Dynamic,
		Title:            s.Title,
		ProviderMetadata: s.ProviderMetadata.providerMetadata(),
	}, nil
}

func (s wireStreamStart) part() (model.StreamStart, error) {
	return model.StreamStart{
		Warnings: decodeWarnings(s.Warnings),
	}, nil
}

// part converts response metadata. The timestamp travels as an ISO 8601
// string.
func (m wireResponseMetadata) part() (model.ResponseMetadata, error) {
	metadata := model.ResponseMetadata{
		ID:      m.ID,
		ModelID: m.ModelID,
	}

	if m.Timestamp == "" {
		return metadata, nil
	}

	timestamp, err := time.Parse(time.RFC3339Nano, m.Timestamp)
	if err != nil {
		return model.ResponseMetadata{}, fmt.Errorf("%w: response timestamp: %w", model.ErrInvalidResponse, err)
	}

	metadata.Timestamp = timestamp

	return metadata, nil
}

func (f wireFinish) part() (model.Finish, error) {
	return model.Finish{
		Usage:            f.Usage.usage(),
		FinishReason:     f.FinishReason.finishReason(),
		ProviderMetadata: f.ProviderMetadata.providerMetadata(),
	}, nil
}

func (r wireRaw) part() (model.Raw, error) {
	return model.Raw{
		Value: jsonOrNull(r.RawValue),
	}, nil
}

// part converts an error part. The error's message is used when it has
// one; otherwise the error value itself describes the failure.
func (e wireStreamError) part() (model.StreamError, error) {
	description := string(jsonOrNull(e.Error))

	var text string
	var described errorMessage

	switch {
	case json.Unmarshal(e.Error, &text) == nil:
		description = text
	case json.Unmarshal(e.Error, &described) == nil && described.Message != nil:
		description = *described.Message
	}

	return model.StreamError{
		Err: fmt.Errorf("%w: %s", errStreamFailed, description),
	}, nil
}
