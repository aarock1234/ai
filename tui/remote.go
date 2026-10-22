package tui

import (
	"context"
	"iter"
	"slices"
	"strconv"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/ai/ui"
)

// remoteConversation runs an agent behind a chat transport, exchanging the
// whole conversation as UI messages on every turn.
type remoteConversation struct {
	transport ui.ChatTransport
	chatID    string
	messages  []ui.Message

	// nextMessageNumber numbers the user messages.
	nextMessageNumber int

	// continuing marks a turn that continues the last assistant message
	// after approvals, so that its response replaces that message.
	continuing bool
}

func newRemoteConversation(transport ui.ChatTransport) *remoteConversation {
	return &remoteConversation{
		transport: transport,
		chatID:    id.New(""),
	}
}

func (c *remoteConversation) addPrompt(prompt string) {
	c.nextMessageNumber++
	c.continuing = false

	c.messages = append(c.messages, ui.Message{
		ID:   "message-" + strconv.Itoa(c.nextMessageNumber),
		Role: ui.RoleUser,
		Parts: []ui.Part{
			ui.TextPart{
				Text: prompt,
			},
		},
	})
}

// addApprovalDecisions records the decisions in the tool parts of the last
// assistant message, which the next turn continues.
func (c *remoteConversation) addApprovalDecisions(decisions []approvalDecision) {
	last, ok := c.lastAssistantMessage()
	if !ok {
		return
	}

	parts := slices.Clone(last.Parts)

	for _, decision := range decisions {
		respondToApproval(parts, decision)
	}

	c.messages[len(c.messages)-1].Parts = parts
	c.continuing = true
}

// respondToApproval moves the tool part of a decided approval to the
// approval-responded state.
func respondToApproval(parts []ui.Part, decision approvalDecision) {
	for index, part := range parts {
		tool, ok := part.(ui.ToolPart)
		if !ok || tool.ToolCallID != decision.request.toolCallID {
			continue
		}

		approved := decision.approved

		tool.State = ui.ToolApprovalResponded
		tool.Approval = &ui.ToolApproval{
			ID:       decision.request.approvalID,
			Approved: &approved,
			Reason:   decision.reason,
		}

		parts[index] = tool

		return
	}
}

func (c *remoteConversation) nextTurn() turnFunc {
	turn := remoteTurn{
		transport: c.transport,
		chatID:    c.chatID,
		messages:  slices.Clone(c.messages),
	}

	if last, ok := c.lastAssistantMessage(); ok {
		continued := last
		turn.continued = &continued
	}

	return turn.stream
}

// finishTurn adds or replaces the assistant message and returns the
// approvals that it waits for.
func (c *remoteConversation) finishTurn(finished turnFinishedMsg) []approvalRequest {
	response := finished.responseMessage
	if response == nil || len(response.Parts) == 0 {
		c.continuing = false

		return nil
	}

	if c.continuing && c.lastAssistantIndex() >= 0 {
		c.messages[len(c.messages)-1] = *response
	} else {
		c.messages = append(c.messages, *response)
	}

	c.continuing = false

	return pendingApprovals(*response)
}

func (c *remoteConversation) lastAssistantMessage() (ui.Message, bool) {
	index := c.lastAssistantIndex()
	if index < 0 {
		return ui.Message{}, false
	}

	return c.messages[index], true
}

// lastAssistantIndex returns the index of the last message when it is an
// assistant message, and -1 otherwise.
func (c *remoteConversation) lastAssistantIndex() int {
	if len(c.messages) == 0 || c.messages[len(c.messages)-1].Role != ui.RoleAssistant {
		return -1
	}

	return len(c.messages) - 1
}

// pendingApprovals lists the tool parts of a message that wait for the
// user's decision. Automatic approvals need none.
func pendingApprovals(message ui.Message) []approvalRequest {
	var requests []approvalRequest

	for _, part := range message.Parts {
		tool, ok := part.(ui.ToolPart)
		if !ok || !isPendingApproval(tool) {
			continue
		}

		requests = append(requests, approvalRequest{
			approvalID:       tool.Approval.ID,
			toolCallID:       tool.ToolCallID,
			toolName:         tool.ToolName,
			title:            tool.Title,
			providerExecuted: tool.ProviderExecuted,
		})
	}

	return requests
}

func isPendingApproval(tool ui.ToolPart) bool {
	if tool.State != ui.ToolApprovalRequested || tool.Approval == nil {
		return false
	}

	return !tool.Approval.IsAutomatic
}

// remoteTurn sends the conversation and streams the response.
type remoteTurn struct {
	transport ui.ChatTransport
	chatID    string
	messages  []ui.Message

	// continued is the assistant message that the response continues,
	// when the conversation ends with one.
	continued *ui.Message
}

func (t remoteTurn) stream(ctx context.Context, events chan<- tea.Msg) {
	stream, err := t.transport.SendMessages(ctx, ui.SendMessagesOptions{
		Trigger:  ui.ChatTriggerSubmitMessage,
		ChatID:   t.chatID,
		Messages: t.messages,
	})
	if err != nil {
		send(ctx, events, turnFinishedMsg{
			err: err,
		})

		return
	}

	defer func() {
		_ = stream.Close()
	}()

	forwarder := &chunkForwarder{
		ctx:    ctx,
		events: events,
	}

	var response *ui.Message

	messages := ui.ReadMessageStream(ui.ReadMessageStreamOptions{
		Message: t.continued,
		Stream:  forwarder.forward(stream.Chunks()),
		OnError: forwarder.recordMessageError,
	})

	for message, err := range messages {
		if err == nil {
			response = &message
		}
	}

	if forwarder.canceled {
		return
	}

	send(ctx, events, turnFinishedMsg{
		responseMessage: response,
		err:             forwarder.err(),
	})
}

// chunkForwarder passes the chunks of a remote stream to the UI while
// ReadMessageStream assembles them, and keeps the error that ended the
// stream. Error chunks are shown by the UI as they arrive.
type chunkForwarder struct {
	ctx    context.Context
	events chan<- tea.Msg

	canceled          bool
	lastChunkWasError bool
	streamErr         error
	messageErr        error
}

func (f *chunkForwarder) forward(chunks iter.Seq2[ui.Chunk, error]) iter.Seq2[ui.Chunk, error] {
	return func(yield func(ui.Chunk, error) bool) {
		for chunk, err := range chunks {
			if err != nil {
				f.streamErr = err
				yield(nil, err)

				return
			}

			_, f.lastChunkWasError = chunk.(ui.ErrorChunk)

			if !send(f.ctx, f.events, chunkMsg{
				chunk: chunk,
			}) {
				f.canceled = true

				return
			}

			if !yield(chunk, nil) {
				return
			}
		}
	}
}

// recordMessageError keeps chunks that do not fit the message. Stream
// errors and error chunks are recorded or shown elsewhere.
func (f *chunkForwarder) recordMessageError(err error) {
	if f.lastChunkWasError || f.streamErr != nil || f.messageErr != nil {
		return
	}

	f.messageErr = err
}

func (f *chunkForwarder) err() error {
	if f.streamErr != nil {
		return f.streamErr
	}

	return f.messageErr
}
