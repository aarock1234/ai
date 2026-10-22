package bridge

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"sync"

	"github.com/aarock1234/ai/internal/id"
)

// Errors of UserMessageSubmitter.
var (
	// ErrUserMessagesClosed reports a submission after the turn stopped
	// accepting user messages.
	ErrUserMessagesClosed = errors.New("the bridge turn is no longer accepting user messages")

	// ErrUserMessageTurnEnded reports a turn that ended before accepting a
	// message.
	ErrUserMessageTurnEnded = errors.New("the bridge turn ended before accepting the user message")
)

// UserMessageSubmitter correlates user messages sent to a running turn with
// the bridge's UserMessageResponse frames, resending unanswered messages
// after reconnects. This API is experimental.
type UserMessageSubmitter struct {
	send func(command UserMessageCommand) error

	mu      sync.Mutex
	pending map[string]*pendingUserMessage
	order   []string
	closed  bool
}

// pendingUserMessage is a message waiting for its response.
type pendingUserMessage struct {
	command UserMessageCommand
	done    chan error
}

// NewUserMessageSubmitter returns a submitter that sends commands with send,
// typically Channel.Send. Feed it the bridge's responses with
// HandleResponse, and call Resend after each reconnect.
func NewUserMessageSubmitter(send func(command UserMessageCommand) error) *UserMessageSubmitter {
	return &UserMessageSubmitter{
		send:    send,
		pending: make(map[string]*pendingUserMessage),
	}
}

// Submit sends text and waits until the bridge accepted or rejected it.
func (s *UserMessageSubmitter) Submit(ctx context.Context, text string) error {
	message := &pendingUserMessage{
		command: UserMessageCommand{
			Type:      "user-message",
			MessageID: id.New("message"),
			Text:      text,
		},
		done: make(chan error, 1),
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()

		return ErrUserMessagesClosed
	}

	s.pending[message.command.MessageID] = message
	s.order = append(s.order, message.command.MessageID)
	s.mu.Unlock()

	s.sendPending(message)

	select {
	case err := <-message.done:
		return err
	case <-ctx.Done():
		s.settle(message.command.MessageID, context.Cause(ctx))

		return context.Cause(ctx)
	}
}

// HandleResponse settles the message a response answers.
func (s *UserMessageSubmitter) HandleResponse(response UserMessageResponse) {
	if response.Accepted {
		s.settle(response.MessageID, nil)

		return
	}

	reason := response.Error
	if strings.TrimSpace(reason) == "" {
		reason = "The runtime rejected the user message."
	}

	s.settle(response.MessageID, errors.New(reason))
}

// Resend sends every unanswered message again, after a reconnect.
func (s *UserMessageSubmitter) Resend() {
	s.mu.Lock()
	messages := make([]*pendingUserMessage, 0, len(s.order))
	for _, messageID := range s.order {
		messages = append(messages, s.pending[messageID])
	}
	s.mu.Unlock()

	for _, message := range messages {
		s.sendPending(message)
	}
}

// Close rejects every unanswered message with err, or with
// ErrUserMessageTurnEnded when err is nil, and rejects later submissions.
func (s *UserMessageSubmitter) Close(err error) {
	if err == nil {
		err = ErrUserMessageTurnEnded
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()

		return
	}

	s.closed = true
	pending := s.pending
	s.pending = make(map[string]*pendingUserMessage)
	s.order = nil
	s.mu.Unlock()

	for _, message := range pending {
		message.done <- err
	}
}

// sendPending sends a message, rejecting it when sending fails.
func (s *UserMessageSubmitter) sendPending(message *pendingUserMessage) {
	if err := s.send(message.command); err != nil {
		s.settle(message.command.MessageID, err)
	}
}

// settle removes a pending message and delivers its outcome.
func (s *UserMessageSubmitter) settle(messageID string, err error) {
	s.mu.Lock()
	message, ok := s.pending[messageID]
	if ok {
		delete(s.pending, messageID)

		for index, pendingID := range s.order {
			if pendingID == messageID {
				s.order = append(s.order[:index], s.order[index+1:]...)

				break
			}
		}
	}
	s.mu.Unlock()

	if ok {
		message.done <- err
	}
}

// DiskLogRecoveryMode is how a respawned bridge recovers a turn from its
// persisted event log when attaching is not possible.
type DiskLogRecoveryMode string

// Disk log recovery modes.
const (
	// RecoverByReplay replays a complete turn from the log.
	RecoverByReplay DiskLogRecoveryMode = "replay"

	// RecoverByRerun reruns the turn, because the log is missing or ends
	// before the turn finished.
	RecoverByRerun DiskLogRecoveryMode = "rerun"
)

// ClassifyDiskLog decides whether a bridge event log, newline-delimited
// JSON, holds a complete turn: it does when its last line is a finish event.
func ClassifyDiskLog(eventLog string) DiskLogRecoveryMode {
	lines := strings.Split(eventLog, "\n")

	for _, line := range slices.Backward(lines) {
		line := strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var event struct {
			Type string `json:"type"`
		}

		if err := json.Unmarshal([]byte(line), &event); err == nil && event.Type == "finish" {
			return RecoverByReplay
		}

		return RecoverByRerun
	}

	return RecoverByRerun
}
