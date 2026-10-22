package ai

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// ToolApprovalStatusType is the approval state of a tool call.
type ToolApprovalStatusType string

// Tool approval states. The empty type means ToolApprovalNotApplicable.
const (
	// ToolApprovalNotApplicable means the call runs without approval.
	ToolApprovalNotApplicable ToolApprovalStatusType = "not-applicable"

	// ToolApprovalApproved approves the call automatically.
	ToolApprovalApproved ToolApprovalStatusType = "approved"

	// ToolApprovalDenied denies the call automatically.
	ToolApprovalDenied ToolApprovalStatusType = "denied"

	// ToolApprovalUserApproval asks the user to approve the call.
	ToolApprovalUserApproval ToolApprovalStatusType = "user-approval"
)

// ToolApprovalStatus is an approval decision with an optional reason. For
// approved and denied calls the reason is reported on the approval response;
// for calls awaiting user approval it is reported on the approval request.
type ToolApprovalStatus struct {
	Type   ToolApprovalStatusType
	Reason string
}

// ToolApprovalConfiguration decides which tool calls need approval before
// they run, the Go form of upstream's ToolApprovalConfiguration union. Use
// a ToolApprovalFunc to decide every call, or ToolApprovals to configure
// tools by name. A decision of the configuration replaces the NeedsApproval
// function of the tool.
type ToolApprovalConfiguration interface {
	// DecideToolApproval returns the approval state of a call. It reports
	// false when the configuration has no decision for the call's tool, so
	// that the tool's NeedsApproval function decides.
	DecideToolApproval(ctx context.Context, event ToolApprovalEvent) (ToolApprovalStatus, bool, error)
}

// ToolApprovalFunc decides whether a tool call needs approval. It decides
// every call, so the NeedsApproval functions of the tools are not called.
type ToolApprovalFunc func(ctx context.Context, event ToolApprovalEvent) (ToolApprovalStatus, error)

// DecideToolApproval implements ToolApprovalConfiguration. A nil function
// has no decision.
func (f ToolApprovalFunc) DecideToolApproval(ctx context.Context, event ToolApprovalEvent) (ToolApprovalStatus, bool, error) {
	if f == nil {
		return ToolApprovalStatus{}, false, nil
	}

	status, err := f(ctx, event)
	if err != nil {
		return ToolApprovalStatus{}, true, err
	}

	return status, true, nil
}

// ToolApprovals configures the approval of tools by name. Calls of tools
// without an entry are decided by the tool's NeedsApproval function.
type ToolApprovals map[string]ToolApprovalRule

// ToolApprovalRule is the approval configuration of one tool. Decide, when
// set, decides each call; otherwise Status applies to every call. The empty
// status type means ToolApprovalNotApplicable.
type ToolApprovalRule struct {
	Status ToolApprovalStatus
	Decide SingleToolApprovalFunc
}

// SingleToolApprovalFunc decides whether a call of one tool needs approval
// from the call's validated JSON input.
type SingleToolApprovalFunc func(ctx context.Context, input json.RawMessage, options ToolNeedsApprovalOptions) (ToolApprovalStatus, error)

// DecideToolApproval implements ToolApprovalConfiguration. It has no
// decision for tools without an entry.
func (a ToolApprovals) DecideToolApproval(ctx context.Context, event ToolApprovalEvent) (ToolApprovalStatus, bool, error) {
	rule, found := a[event.ToolCall.ToolName]
	if !found {
		return ToolApprovalStatus{}, false, nil
	}

	if rule.Decide == nil {
		return rule.Status, true, nil
	}

	status, err := rule.Decide(ctx, event.ToolCall.Input, ToolNeedsApprovalOptions{
		ToolCallID: event.ToolCall.ToolCallID,
		Messages:   event.Messages,
	})
	if err != nil {
		return ToolApprovalStatus{}, true, err
	}

	return status, true, nil
}

// ToolApprovalEvent describes a tool call awaiting an approval decision.
type ToolApprovalEvent struct {
	ToolCall ToolCall

	// Tools are the tools available to the model.
	Tools []Tool

	// Messages are the messages sent to the model in the step that produced
	// the call. They exclude the system prompt and the assistant response
	// that contains the call.
	Messages []model.Message
}

// toolApprovalRequest asks for the approval state of one valid tool call.
type toolApprovalRequest struct {
	tools         []Tool
	configuration ToolApprovalConfiguration
	call          ToolCall
	messages      []model.Message
}

// resolveToolApproval returns the approval state of a call: the configured
// approval decides when it has a decision, and otherwise the tool's
// NeedsApproval function. Configured decisions are normalized here, once,
// whichever configuration made them.
func resolveToolApproval(ctx context.Context, request toolApprovalRequest) (ToolApprovalStatus, error) {
	if request.configuration != nil {
		status, decided, err := request.configuration.DecideToolApproval(ctx, ToolApprovalEvent{
			ToolCall: request.call,
			Tools:    request.tools,
			Messages: request.messages,
		})
		if err != nil {
			return ToolApprovalStatus{}, err
		}

		if decided {
			return normalizeToolApprovalStatus(request.call, status)
		}
	}

	tool, found := findTool(request.tools, request.call.ToolName)
	if !found || tool.NeedsApproval == nil {
		return notApplicableApproval(), nil
	}

	needsApproval, err := tool.NeedsApproval(ctx, request.call.Input, ToolNeedsApprovalOptions{
		ToolCallID: request.call.ToolCallID,
		Messages:   request.messages,
	})
	if err != nil {
		return ToolApprovalStatus{}, err
	}

	if !needsApproval {
		return notApplicableApproval(), nil
	}

	return ToolApprovalStatus{
		Type: ToolApprovalUserApproval,
	}, nil
}

// normalizeToolApprovalStatus maps the empty type to
// ToolApprovalNotApplicable and rejects types that are not approval
// states.
func normalizeToolApprovalStatus(call ToolCall, status ToolApprovalStatus) (ToolApprovalStatus, error) {
	switch status.Type {
	case "", ToolApprovalNotApplicable:
		return notApplicableApproval(), nil

	case ToolApprovalApproved, ToolApprovalDenied, ToolApprovalUserApproval:
		return status, nil

	default:
		return ToolApprovalStatus{}, &InvalidArgumentError{
			Parameter: "toolApproval." + call.ToolName,
			Value:     status.Type,
			Message:   "approval status type must be not-applicable, approved, denied or user-approval",
		}
	}
}

func notApplicableApproval() ToolApprovalStatus {
	return ToolApprovalStatus{
		Type: ToolApprovalNotApplicable,
	}
}

// collectedToolApproval is an approval response from the prompt with the
// request and the tool call it answers.
type collectedToolApproval struct {
	request  model.ToolApprovalRequestPart
	response model.ToolApprovalResponsePart
	call     ToolCall

	// existingResult is the result already recorded for the call, if any.
	existingResult *model.ToolResultPart
}

// collectedToolApprovals splits the approval responses of the prompt.
type collectedToolApprovals struct {
	approved []collectedToolApproval
	denied   []collectedToolApproval
}

// collectToolApprovals gathers the approval responses of the last message
// when it is a tool message. Responses that already have a result are
// skipped, except denials whose result is not an execution denial. Approved
// calls are returned in the order the model made them.
func collectToolApprovals(messages []model.Message) (collectedToolApprovals, error) {
	if len(messages) == 0 {
		return collectedToolApprovals{}, nil
	}

	lastMessage, isTool := messages[len(messages)-1].(model.ToolMessage)
	if !isTool {
		return collectedToolApprovals{}, nil
	}

	history := indexApprovalHistory(messages)
	results := toolResultsByCallID(lastMessage)

	var collected collectedToolApprovals

	for _, part := range lastMessage.Content {
		response, isResponse := part.(model.ToolApprovalResponsePart)
		if !isResponse {
			continue
		}

		request, found := history.requests[response.ApprovalID]
		if !found {
			return collectedToolApprovals{}, &InvalidToolApprovalError{
				ApprovalID: response.ApprovalID,
			}
		}

		existingResult := results[request.ToolCallID]
		if isApprovalResolved(response, existingResult) {
			continue
		}

		call, found := history.calls[request.ToolCallID]
		if !found {
			return collectedToolApprovals{}, &ToolCallNotFoundForApprovalError{
				ToolCallID: request.ToolCallID,
				ApprovalID: request.ApprovalID,
			}
		}

		approval := collectedToolApproval{
			request:        request,
			response:       response,
			call:           call,
			existingResult: existingResult,
		}

		if response.Approved {
			collected.approved = append(collected.approved, approval)
		} else {
			collected.denied = append(collected.denied, approval)
		}
	}

	slices.SortStableFunc(collected.approved, func(first, second collectedToolApproval) int {
		return history.callOrder[first.call.ToolCallID] - history.callOrder[second.call.ToolCallID]
	})

	return collected, nil
}

// approvalHistory indexes the tool calls and approval requests of the
// assistant messages.
type approvalHistory struct {
	calls     map[string]ToolCall
	callOrder map[string]int
	requests  map[string]model.ToolApprovalRequestPart
}

func indexApprovalHistory(messages []model.Message) approvalHistory {
	history := approvalHistory{
		calls:     make(map[string]ToolCall),
		callOrder: make(map[string]int),
		requests:  make(map[string]model.ToolApprovalRequestPart),
	}

	for _, message := range messages {
		assistant, ok := message.(model.AssistantMessage)
		if !ok {
			continue
		}

		for _, part := range assistant.Content {
			switch typed := part.(type) {
			case model.ToolCallPart:
				history.calls[typed.ToolCallID] = ToolCall{
					ToolCallID:       typed.ToolCallID,
					ToolName:         typed.ToolName,
					Input:            typed.Input,
					ProviderExecuted: typed.ProviderExecuted,
				}
				history.callOrder[typed.ToolCallID] = len(history.callOrder)

			case model.ToolApprovalRequestPart:
				history.requests[typed.ApprovalID] = typed
			}
		}
	}

	return history
}

func toolResultsByCallID(message model.ToolMessage) map[string]*model.ToolResultPart {
	results := make(map[string]*model.ToolResultPart)

	for _, part := range message.Content {
		if result, ok := part.(model.ToolResultPart); ok {
			results[result.ToolCallID] = &result
		}
	}

	return results
}

// isApprovalResolved reports whether an approval response already has its
// result. A denial is unresolved while its result is not an execution
// denial.
func isApprovalResolved(response model.ToolApprovalResponsePart, existingResult *model.ToolResultPart) bool {
	if existingResult == nil {
		return false
	}

	if response.Approved {
		return true
	}

	return !isExecutionDenied(existingResult.Output)
}

func isExecutionDenied(output model.ToolResultOutput) bool {
	_, denied := output.(model.ExecutionDeniedOutput)

	return denied
}

// validatedToolApprovals splits approved calls after revalidation.
type validatedToolApprovals struct {
	approved []collectedToolApproval

	// denied holds approvals that the approval policy now denies.
	denied []collectedToolApproval

	// invalid holds approvals whose input no longer validates.
	invalid []invalidToolApproval
}

// invalidToolApproval is an approved call whose input failed revalidation.
type invalidToolApproval struct {
	collectedToolApproval

	err *InvalidToolInputError
}

// approvalValidation describes the revalidation of approved calls that come
// from client-supplied message history.
type approvalValidation struct {
	approvals     []collectedToolApproval
	tools         []Tool
	configuration ToolApprovalConfiguration
	messages      []model.Message
	secret        []byte
	refine        map[string]ToolInputRefinement
}

// validateApprovedToolApprovals revalidates approved calls before they run:
// it verifies their signature when a secret is configured, checks their
// input against the tool's schema without changing it, and applies the
// approval policy again.
func validateApprovedToolApprovals(ctx context.Context, validation approvalValidation) (validatedToolApprovals, error) {
	var validated validatedToolApprovals

	for _, approval := range validation.approvals {
		if validation.secret != nil {
			if err := verifyApprovalSignature(validation.secret, approval); err != nil {
				return validatedToolApprovals{}, err
			}
		}

		if err := revalidateApprovedInput(ctx, validation, approval); err != nil {
			validated.invalid = append(validated.invalid, invalidToolApproval{
				collectedToolApproval: approval,
				err: &InvalidToolInputError{
					ToolName:  approval.call.ToolName,
					ToolInput: string(approval.call.Input),
					Cause:     err,
				},
			})

			continue
		}

		status, err := resolveToolApproval(ctx, toolApprovalRequest{
			tools:         validation.tools,
			configuration: validation.configuration,
			call:          approval.call,
			messages:      validation.messages,
		})
		if err != nil {
			return validatedToolApprovals{}, err
		}

		if status.Type != ToolApprovalDenied {
			validated.approved = append(validated.approved, approval)

			continue
		}

		approval.response.Approved = false
		if status.Reason != "" {
			approval.response.Reason = status.Reason
		}

		validated.denied = append(validated.denied, approval)
	}

	return validated, nil
}

func verifyApprovalSignature(secret []byte, approval collectedToolApproval) error {
	if approval.request.Signature == "" {
		return &InvalidToolApprovalSignatureError{
			ApprovalID: approval.request.ApprovalID,
			ToolCallID: approval.call.ToolCallID,
			Reason:     "missing signature",
		}
	}

	valid, err := verifyToolApprovalSignature(toolApprovalSignature{
		secret:     secret,
		approvalID: approval.request.ApprovalID,
		toolCallID: approval.call.ToolCallID,
		toolName:   approval.call.ToolName,
		input:      approval.call.Input,
	}, approval.request.Signature)
	if err != nil || !valid {
		return &InvalidToolApprovalSignatureError{
			ApprovalID: approval.request.ApprovalID,
			ToolCallID: approval.call.ToolCallID,
			Reason:     "invalid signature",
		}
	}

	return nil
}

// revalidateApprovedInput checks the approved input of an executable tool
// against its schema and verifies that refining it again does not change the
// operation that was approved. When the request recorded the input before
// refinement, that input is the one validated and refined again.
func revalidateApprovedInput(ctx context.Context, validation approvalValidation, approval collectedToolApproval) error {
	call := approval.call

	tool, found := findTool(validation.tools, call.ToolName)
	if !found || tool.Execute == nil {
		return nil
	}

	schemaInput := call.Input
	if approval.request.InputSchemaInput != nil {
		schemaInput = approval.request.InputSchemaInput
	}

	input, err := validateToolInput(tool, string(schemaInput))
	if err != nil {
		return err
	}

	revalidated := call
	revalidated.Input = input

	refined, err := applyToolInputRefinement(ctx, validation.refine, revalidated)
	if err != nil {
		return err
	}

	if !equalJSON(refined.Input, call.Input) {
		return errors.New("approved tool input does not match the validated schema output")
	}

	return nil
}

// equalJSON reports whether two JSON documents encode the same value.
func equalJSON(first, second json.RawMessage) bool {
	var firstValue, secondValue any

	if err := json.Unmarshal(first, &firstValue); err != nil {
		return false
	}

	if err := json.Unmarshal(second, &secondValue); err != nil {
		return false
	}

	return reflect.DeepEqual(firstValue, secondValue)
}
