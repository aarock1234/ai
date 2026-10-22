package google

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"sync"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

var errStreamClosed = errors.New("stream closed")

// Stream performs one streamGenerateContent call. When a stream reaches the
// per-request output cap while budget remains, its last chunk finishes with
// CONTINUATION; the request is resent with the continuation token and the
// new stream, which holds only newly generated content, is appended.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	call, err := m.prepareCall(ctx, options)
	if err != nil {
		return nil, err
	}

	call.enableStreamedFunctionCallArguments()

	request, err := m.provider.request(ctx, modelPath(m.modelID)+":streamGenerateContent?alt=sse", call.requestHeaders(options.Headers))
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, call.body)
	if err != nil {
		return nil, err
	}

	body := &streamBody{
		current: response.Body,
	}

	session := &streamSession{
		ctx:              ctx,
		request:          request,
		call:             call,
		body:             body,
		includeRawChunks: options.IncludeRawChunks,
	}

	parts := session.parts()
	if call.jsonResponseToolName != "" {
		parts = convertJSONResponseToolStream(parts, call.jsonResponseToolName)
	}

	result := model.NewStreamResult(parts, body.Close)
	result.Request = model.RequestInfo{
		Body: response.RequestBody,
	}
	result.Response = model.StreamResponseInfo{
		Headers: response.Headers,
	}

	return result, nil
}

// streamSession reads the event streams of one call, including the streams
// of automatic continuations.
type streamSession struct {
	ctx              context.Context
	request          apicall.Request
	call             preparedCall
	body             *streamBody
	includeRawChunks bool

	// pendingToken is the continuation token of the current stream.
	pendingToken string

	// requestGrounding is the latest grounding of the current request, and
	// groundingPerRequest holds that of the completed requests.
	requestGrounding    *GroundingMetadata
	groundingPerRequest []*GroundingMetadata
}

func (s *streamSession) parts() iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		state := newStreamState(s.call)

		start := model.StreamStart{
			Warnings: s.call.warnings,
		}

		if !yield(start) {
			return
		}

		for {
			if !s.readCurrentStream(state, yield) {
				return
			}

			if s.pendingToken == "" {
				break
			}

			// A failed continuation ends the stream; the finish part keeps
			// the last continuation token so the caller can resume.
			err := s.continueStream()
			if err == nil {
				continue
			}

			failure := model.StreamError{
				Err: err,
			}

			if !yield(failure) {
				return
			}

			break
		}

		finish := state.finish(s.groundingPerRequest, s.requestGrounding)
		for _, part := range finish {
			if !yield(part) {
				return
			}
		}
	}
}

// readCurrentStream yields the parts of the current event stream. It
// reports false when the consumer stopped.
func (s *streamSession) readCurrentStream(state *streamState, yield func(model.StreamPart) bool) bool {
	for event, err := range sse.Read(s.body.reader()) {
		if err != nil {
			state.fail()
			s.pendingToken = ""

			return yield(model.StreamError{
				Err: s.readError(err),
			})
		}

		for _, part := range s.handleEvent(state, event) {
			if !yield(part) {
				return false
			}
		}
	}

	return true
}

func (s *streamSession) handleEvent(state *streamState, event sse.Event) []model.StreamPart {
	var parts []model.StreamPart

	if s.includeRawChunks && json.Valid([]byte(event.Data)) {
		parts = append(parts, model.Raw{
			Value: json.RawMessage(event.Data),
		})
	}

	var chunk generateContentResponse
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		return append(parts, model.StreamError{
			Err: fmt.Errorf("%w: decode stream chunk: %w", model.ErrInvalidResponse, err),
		})
	}

	s.trackContinuation(chunk)

	return append(parts, state.process(chunk)...)
}

func (s *streamSession) trackContinuation(chunk generateContentResponse) {
	candidate := chunk.firstCandidate()
	if candidate == nil {
		return
	}

	if candidate.GroundingMetadata != nil {
		s.requestGrounding = candidate.GroundingMetadata
	}

	if token := continuationToken(candidate); token != "" {
		s.pendingToken = token
	}
}

// continueStream sends the continuation request and replaces the current
// body with its stream.
func (s *streamSession) continueStream() error {
	body := s.call.body
	body.ContinuationToken = s.pendingToken
	s.pendingToken = ""

	response, err := apicall.PostJSONStream(s.ctx, s.request, body)
	if err != nil {
		return err
	}

	if err := s.body.replace(response.Body); err != nil {
		return err
	}

	s.groundingPerRequest = append(s.groundingPerRequest, s.requestGrounding)
	s.requestGrounding = nil

	return nil
}

func (s *streamSession) readError(err error) error {
	if ctxErr := s.ctx.Err(); ctxErr != nil {
		return ctxErr
	}

	return fmt.Errorf("read stream: %w", err)
}

// streamBody owns the response body of the current request. Continuations
// replace it; Close releases whichever body is current.
type streamBody struct {
	mu      sync.Mutex
	current io.ReadCloser
	closed  bool
}

func (b *streamBody) reader() io.Reader {
	b.mu.Lock()
	defer b.mu.Unlock()

	return b.current
}

// replace closes the current body and makes next current. It closes next
// instead when the stream was already closed.
func (b *streamBody) replace(next io.ReadCloser) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		_ = next.Close()

		return errStreamClosed
	}

	previous := b.current
	b.current = next

	// The previous body was read to its end, so its close result is moot.
	_ = previous.Close()

	return nil
}

// Close closes the current body.
func (b *streamBody) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return nil
	}

	b.closed = true

	return b.current.Close()
}
