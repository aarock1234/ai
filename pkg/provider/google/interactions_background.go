package google

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// The polling of a background interaction starts at one second and
	// doubles up to ten seconds.
	interactionPollInitialDelay = time.Second
	interactionPollMaxDelay     = 10 * time.Second

	// An interrupted event stream reconnects up to three times, waiting
	// half a second longer before each attempt.
	interactionStreamMaxAttempts = 3
	interactionStreamRetryDelay  = 500 * time.Millisecond

	// interactionCancelTimeout bounds the best-effort cancellation of an
	// abandoned background interaction.
	interactionCancelTimeout = 10 * time.Second
)

var errNoStreamEvents = errors.New("google.interactions: SSE stream closed without producing any events")

// ErrInteractionPollTimeout reports a background interaction that did not
// end within the polling timeout. The interaction is cancelled on the server.
var ErrInteractionPollTimeout = errors.New("google interaction polling timed out")

// polledInteraction is the terminal state of a polled interaction.
type polledInteraction struct {
	response interactionsResponse
	raw      *apicall.Response
}

// pollInteraction polls the interaction until it ends or the timeout
// passes. When the timeout passes or ctx ends first, the interaction is
// cancelled on the server so that it stops running.
func (m *InteractionsModel) pollInteraction(ctx context.Context, headers http.Header, interactionID string, timeout time.Duration) (polledInteraction, error) {
	if interactionID == "" {
		return polledInteraction{}, fmt.Errorf(
			"%w: google.interactions: cannot poll a background interaction without an id; the POST response did not include an interaction id",
			model.ErrInvalidResponse,
		)
	}

	started := time.Now()
	delay := interactionPollInitialDelay

	for {
		if time.Since(started) > timeout {
			m.cancelInteraction(ctx, headers, interactionID)

			return polledInteraction{}, fmt.Errorf("%w: interaction %s after %s", ErrInteractionPollTimeout, interactionID, timeout)
		}

		polled, err := m.pollOnce(ctx, headers, interactionID, delay)
		if err != nil {
			if ctx.Err() != nil {
				m.cancelInteraction(ctx, headers, interactionID)

				return polledInteraction{}, context.Cause(ctx)
			}

			return polledInteraction{}, err
		}

		if isTerminalInteraction(polled.response.Status) {
			return polled, nil
		}

		delay = min(2*delay, interactionPollMaxDelay)
	}
}

// pollOnce waits for delay and reads the interaction.
func (m *InteractionsModel) pollOnce(ctx context.Context, headers http.Header, interactionID string, delay time.Duration) (polledInteraction, error) {
	if err := poll.Wait(ctx, delay); err != nil {
		return polledInteraction{}, context.Cause(ctx)
	}

	request, err := m.provider.request(ctx, interactionPath(interactionID), headers)
	if err != nil {
		return polledInteraction{}, err
	}

	raw, err := apicall.Get(ctx, request)
	if err != nil {
		return polledInteraction{}, err
	}

	response, err := decodeInteraction(raw.Body)
	if err != nil {
		return polledInteraction{}, err
	}

	return polledInteraction{
		response: response,
		raw:      raw,
	}, nil
}

// cancelInteraction asks the server to stop an interaction the caller
// abandoned. It is best effort: failures are ignored, so that they cannot
// hide the reason the caller stopped.
func (m *InteractionsModel) cancelInteraction(ctx context.Context, headers http.Header, interactionID string) {
	if interactionID == "" {
		return
	}

	cancelCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), interactionCancelTimeout)
	defer cancel()

	request, err := m.provider.request(cancelCtx, interactionPath(interactionID)+"/cancel", headers)
	if err != nil {
		return
	}

	_, _ = apicall.PostJSON(cancelCtx, request, emptyObject{})
}

func interactionPath(interactionID string) string {
	return "interactions/" + url.PathEscape(interactionID)
}

// streamBackground creates a background interaction and follows its event
// stream. An interaction that already ended is replayed from the response.
func (m *InteractionsModel) streamBackground(
	ctx context.Context,
	request apicall.Request,
	headers http.Header,
	call preparedInteraction,
	includeRawChunks bool,
) (*model.StreamResult, error) {
	raw, err := apicall.PostJSON(ctx, request, call.body)
	if err != nil {
		return nil, err
	}

	response, err := decodeInteraction(raw.Body)
	if err != nil {
		return nil, err
	}

	interactionID := nonEmptyID(response.ID)
	if interactionID == "" {
		return nil, fmt.Errorf(
			"%w: google.interactions: background POST response did not include an interaction id; cannot stream the result",
			model.ErrInvalidResponse,
		)
	}

	setup := interactionsStreamSetup{
		warnings:         call.warnings,
		generateID:       m.provider.generateID,
		includeRawChunks: includeRawChunks,
		serviceTier:      raw.Headers.Get(serviceTierHeader),
	}

	var result *model.StreamResult

	if isTerminalInteraction(response.Status) {
		replay := interactionReplay{
			interactionsStreamSetup: setup,
			response:                response,
			raw:                     raw.Body,
		}

		result = model.NewStreamResult(replay.parts(), nil)
	} else {
		follower := &interactionFollower{
			ctx:           ctx,
			interactions:  m,
			headers:       headers,
			interactionID: interactionID,
		}

		result = model.NewStreamResult(newInteractionsStreamState(setup).parts(follower.events()), follower.close)
	}

	result.Request = model.RequestInfo{
		Body: raw.RequestBody,
	}
	result.Response = model.StreamResponseInfo{
		Headers: raw.Headers,
	}

	return result, nil
}

// interactionFollower reads the event stream of a background interaction.
// Idle agents may drop the connection between events, so it reconnects
// with the last event ID. It owns the body of the current connection, and
// cancels the interaction when the caller abandons the stream before the
// interaction ended.
type interactionFollower struct {
	ctx           context.Context
	interactions  *InteractionsModel
	headers       http.Header
	interactionID string

	mu   sync.Mutex
	body io.ReadCloser

	// ended is set once the stream reported the end of the interaction or
	// gave up reconnecting; abandoned interactions are cancelled otherwise.
	ended  bool
	closed bool
}

// interactionEventHeader holds the fields that resume a stream.
type interactionEventHeader struct {
	EventID   *string `json:"event_id"`
	EventType string  `json:"event_type"`
}

// events yields the events of every connection until the interaction
// completes or fails, or reconnecting fails three times in a row.
func (f *interactionFollower) events() iter.Seq2[sse.Event, error] {
	return func(yield func(sse.Event, error) bool) {
		attempts := 0
		lastEventID := ""

		for {
			body, err := f.open(lastEventID)
			if err != nil {
				if !f.retry(&attempts, err, yield) {
					return
				}

				continue
			}

			outcome := f.read(body, &lastEventID, yield)
			f.closeBody()

			switch {
			case outcome.stopped || outcome.complete:
				return

			case outcome.err != nil:
				if !f.retry(&attempts, outcome.err, yield) {
					return
				}

			case !outcome.receivedEvents:
				if !f.retry(&attempts, errNoStreamEvents, yield) {
					return
				}

			default:
				attempts = 0
			}
		}
	}
}

// readOutcome reports how reading one connection ended.
type readOutcome struct {
	receivedEvents bool
	complete       bool
	stopped        bool
	err            error
}

// read yields the events of one connection and tracks the last event ID.
// It stops after the event that ends the interaction.
func (f *interactionFollower) read(body io.Reader, lastEventID *string, yield func(sse.Event, error) bool) readOutcome {
	var outcome readOutcome

	for event, err := range sse.Read(body) {
		if err != nil {
			outcome.err = fmt.Errorf("read stream: %w", err)

			return outcome
		}

		outcome.receivedEvents = true

		var header interactionEventHeader
		if json.Unmarshal([]byte(event.Data), &header) == nil {
			if id := valueOrEmpty(header.EventID); id != "" {
				*lastEventID = id
			}

			outcome.complete = header.EventType == "interaction.completed" || header.EventType == "error"
		}

		if outcome.complete {
			f.markEnded()
		}

		if !yield(event, nil) {
			outcome.stopped = true

			return outcome
		}

		if outcome.complete {
			return outcome
		}
	}

	return outcome
}

// retry counts a failed attempt and waits before the next one. It reports
// false, after yielding the error, when the stream must end.
func (f *interactionFollower) retry(attempts *int, err error, yield func(sse.Event, error) bool) bool {
	if f.ctx.Err() != nil {
		yield(sse.Event{}, context.Cause(f.ctx))

		return false
	}

	*attempts++

	if *attempts >= interactionStreamMaxAttempts {
		f.markEnded()
		yield(sse.Event{}, err)

		return false
	}

	if waitErr := poll.Wait(f.ctx, time.Duration(*attempts)*interactionStreamRetryDelay); waitErr != nil {
		yield(sse.Event{}, context.Cause(f.ctx))

		return false
	}

	return true
}

// open connects to the event stream, resuming after lastEventID.
func (f *interactionFollower) open(lastEventID string) (io.ReadCloser, error) {
	query := url.Values{
		"stream": {
			strconv.FormatBool(true),
		},
	}

	if lastEventID != "" {
		query.Set("last_event_id", lastEventID)
	}

	headers := apicall.MergeHeaders(f.headers, http.Header{
		"Accept": {
			"text/event-stream",
		},
	})

	request, err := f.interactions.provider.request(f.ctx, interactionPath(f.interactionID)+"?"+query.Encode(), headers)
	if err != nil {
		return nil, err
	}

	stream, err := apicall.GetStream(f.ctx, request)
	if err != nil {
		return nil, err
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	if f.closed {
		_ = stream.Body.Close()

		return nil, errStreamClosed
	}

	f.body = stream.Body

	return stream.Body, nil
}

func (f *interactionFollower) markEnded() {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.ended = true
}

func (f *interactionFollower) closeBody() {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.body != nil {
		// The body was read to its end or abandoned, so its close result
		// is moot.
		_ = f.body.Close()
		f.body = nil
	}
}

// close releases the current connection and cancels an interaction that
// had not ended.
func (f *interactionFollower) close() error {
	f.mu.Lock()

	if f.closed {
		f.mu.Unlock()

		return nil
	}

	f.closed = true
	body := f.body
	f.body = nil
	ended := f.ended

	f.mu.Unlock()

	var err error
	if body != nil {
		err = body.Close()
	}

	if !ended {
		f.interactions.cancelInteraction(f.ctx, f.headers, f.interactionID)
	}

	return err
}

// interactionReplay replays an interaction that ended before it could be
// streamed, emitting each block as one delta.
type interactionReplay struct {
	interactionsStreamSetup

	response interactionsResponse
	raw      json.RawMessage
}

func (r interactionReplay) parts() iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		for _, part := range r.replayParts() {
			if !yield(part) {
				return
			}
		}
	}
}

func (r interactionReplay) replayParts() []model.StreamPart {
	interactionID := nonEmptyID(r.response.ID)

	parts := []model.StreamPart{
		model.StreamStart{
			Warnings: r.warnings,
		},
		model.ResponseMetadata{
			ID:        interactionID,
			Timestamp: interactionsTimestamp(r.response.Created),
			ModelID:   valueOrEmpty(r.response.Model),
		},
	}

	if r.includeRawChunks {
		parts = append(parts, model.Raw{
			Value: r.raw,
		})
	}

	content, hasFunctionCall, err := parseInteractionsOutputs(r.response.Steps, r.generateID, interactionID)
	if err != nil {
		parts = append(parts, model.StreamError{
			Err: err,
		})
	}

	blockPrefix := firstNonEmpty(interactionID, "agent")
	blockCount := 0

	for _, item := range content {
		blockID := ""

		switch item.(type) {
		case model.Text, model.Reasoning:
			blockID = blockPrefix + ":" + strconv.Itoa(blockCount)
			blockCount++
		}

		parts = append(parts, replayContent(item, blockID)...)
	}

	usage, _, err := convertInteractionsUsage(r.response.Usage)
	if err != nil {
		parts = append(parts, model.StreamError{
			Err: err,
		})
	}

	serviceTier := valueOrEmpty(r.response.ServiceTier)
	if serviceTier == "" {
		serviceTier = r.serviceTier
	}

	return append(parts, model.Finish{
		Usage:        usage,
		FinishReason: interactionsFinishReason(r.response.Status, hasFunctionCall),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: InteractionsMetadata{
				InteractionID: interactionID,
				ServiceTier:   serviceTier,
			},
		},
	})
}

// replayContent returns the stream parts of one content part. Custom
// content is not replayed.
func replayContent(item model.Content, blockID string) []model.StreamPart {
	switch typed := item.(type) {
	case model.Text:
		parts := []model.StreamPart{
			model.TextStart{
				ID: blockID,
			},
		}

		if typed.Text != "" {
			parts = append(parts, model.TextDelta{
				ID:    blockID,
				Delta: typed.Text,
			})
		}

		return append(parts, model.TextEnd{
			ID:               blockID,
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.Reasoning:
		parts := []model.StreamPart{
			model.ReasoningStart{
				ID: blockID,
			},
		}

		if typed.Text != "" {
			parts = append(parts, model.ReasoningDelta{
				ID:    blockID,
				Delta: typed.Text,
			})
		}

		return append(parts, model.ReasoningEnd{
			ID:               blockID,
			ProviderMetadata: typed.ProviderMetadata,
		})

	case model.ToolCall:
		return []model.StreamPart{
			model.ToolInputStart{
				ID:               typed.ToolCallID,
				ToolName:         typed.ToolName,
				ProviderExecuted: typed.ProviderExecuted,
			},
			model.ToolInputDelta{
				ID:    typed.ToolCallID,
				Delta: typed.Input,
			},
			model.ToolInputEnd{
				ID: typed.ToolCallID,
			},
			typed,
		}

	case model.ToolResult:
		return []model.StreamPart{
			model.ToolResult{
				ToolCallID: typed.ToolCallID,
				ToolName:   typed.ToolName,
				Result:     typed.Result,
			},
		}

	case model.Source:
		return []model.StreamPart{
			typed,
		}

	case model.File:
		return []model.StreamPart{
			typed,
		}

	default:
		return nil
	}
}
