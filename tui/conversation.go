package tui

import (
	"context"
	"slices"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// approvalRequest is a tool call that waits for the user's decision.
type approvalRequest struct {
	approvalID       string
	toolCallID       string
	toolName         string
	title            string
	providerExecuted bool
}

// displayName is the tool's title, or its name without one.
func (r approvalRequest) displayName() string {
	if r.title != "" {
		return r.title
	}

	return r.toolName
}

// approvalDecision is the user's decision on an approval request.
type approvalDecision struct {
	request  approvalRequest
	approved bool
	reason   string
}

// conversation is the history of a session and the way its turns run: a
// local agent over model messages, or a remote agent over UI messages. The
// UI goroutine owns it; turns stream from a snapshot.
type conversation interface {
	// addPrompt adds a user prompt.
	addPrompt(prompt string)

	// addApprovalDecisions records the decisions on the approvals of the
	// last turn, which the next turn continues.
	addApprovalDecisions(decisions []approvalDecision)

	// nextTurn returns the next turn over a snapshot of the history.
	nextTurn() turnFunc

	// finishTurn adds the outcome of a turn and returns the approvals it
	// requested that the stream did not report.
	finishTurn(finished turnFinishedMsg) []approvalRequest
}

// localConversation runs an ai.Agent in this process.
type localConversation struct {
	agent    ai.Agent
	messages []model.Message
}

func newLocalConversation(agent ai.Agent) *localConversation {
	return &localConversation{
		agent: agent,
	}
}

func (c *localConversation) addPrompt(prompt string) {
	c.messages = append(c.messages, model.UserMessage{
		Content: []model.UserPart{
			model.TextPart{
				Text: prompt,
			},
		},
	})
}

// addApprovalDecisions adds a tool message with the approval responses.
func (c *localConversation) addApprovalDecisions(decisions []approvalDecision) {
	responses := make([]model.ToolPart, 0, len(decisions))

	for _, decision := range decisions {
		responses = append(responses, model.ToolApprovalResponsePart{
			ApprovalID:       decision.request.approvalID,
			Approved:         decision.approved,
			Reason:           decision.reason,
			ProviderExecuted: decision.request.providerExecuted,
		})
	}

	c.messages = append(c.messages, model.ToolMessage{
		Content: responses,
	})
}

func (c *localConversation) nextTurn() turnFunc {
	snapshot := slices.Clone(c.messages)

	return func(ctx context.Context, events chan<- tea.Msg) {
		streamLocalTurn(ctx, c.agent, snapshot, events)
	}
}

// finishTurn adds the generated messages. Local approvals arrive as stream
// parts, so none are returned.
func (c *localConversation) finishTurn(finished turnFinishedMsg) []approvalRequest {
	c.messages = append(c.messages, finished.responseMessages...)

	return nil
}

func streamLocalTurn(ctx context.Context, agent ai.Agent, messages []model.Message, events chan<- tea.Msg) {
	result, err := agent.Stream(ctx, ai.AgentStreamOptions{
		Messages: messages,
	})
	if err != nil {
		send(ctx, events, turnFinishedMsg{
			err: err,
		})

		return
	}

	for part := range result.FullStream() {
		delivered := send(ctx, events, streamPartMsg{
			part: part,
		})
		if !delivered {
			return
		}
	}

	finished := turnFinishedMsg{}

	final, err := result.Result()
	if final != nil {
		finished.responseMessages = final.ResponseMessages
	}
	finished.err = err

	send(ctx, events, finished)
}

// localApprovalRequest describes a stream's approval request.
func localApprovalRequest(request ai.ToolApprovalRequest) approvalRequest {
	return approvalRequest{
		approvalID:       request.ApprovalID,
		toolCallID:       request.ToolCall.ToolCallID,
		toolName:         request.ToolCall.ToolName,
		title:            request.ToolCall.Title,
		providerExecuted: request.ToolCall.ProviderExecuted,
	}
}
