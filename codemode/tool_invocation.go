package codemode

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/aarock1234/ai/internal/schema"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// hostTools calls the application's tools on behalf of the program.
type hostTools struct {
	tools           map[string]ai.Tool
	names           []string
	validators      map[string]func() (*schema.Validator, error)
	options         Options
	limits          executionLimits
	messages        []model.Message
	outerToolCallID string
}

// hostCall is one host tool call made by the program.
type hostCall struct {
	request int

	// bindingName is the host function the program called, and
	// argumentsJSON its run-js-v1 argument list.
	bindingName   string
	argumentsJSON string

	// resumed is the resolution of an interrupt the call requested in an
	// earlier execution.
	resumed *resumedCall
}

// hostOutcome is the result of a host tool call: an output, an interrupt
// request or a failure, with the tool and input the call resolved to.
type hostOutcome struct {
	toolName string
	input    json.RawMessage

	// output is the JSON encoding of the tool's output; empty means
	// undefined.
	output    string
	interrupt *InterruptPayload
	err       error
}

// indexTools maps the host tools by name and returns their names in
// JavaScript's default sort order, which numbers their host functions.
func indexTools(tools []ai.Tool) (map[string]ai.Tool, []string, error) {
	byName := make(map[string]ai.Tool, len(tools))

	for _, tool := range tools {
		if _, duplicate := byName[tool.Name]; duplicate {
			return nil, nil, fmt.Errorf("code mode: duplicate host tool %q", tool.Name)
		}

		byName[tool.Name] = tool
	}

	names := slices.AppendSeq(make([]string, 0, len(byName)), maps.Keys(byName))
	slices.SortFunc(names, compareUTF16)

	return byName, names, nil
}

func compileValidators(tools map[string]ai.Tool) map[string]func() (*schema.Validator, error) {
	validators := make(map[string]func() (*schema.Validator, error), len(tools))

	for name, tool := range tools {
		validators[name] = sync.OnceValues(func() (*schema.Validator, error) {
			return schema.Compile(tool.InputSchema)
		})
	}

	return validators
}

// nestedToolCallID is the ID of a host tool call: the outer code mode tool
// call ID and the call's 1-based request number.
func nestedToolCallID(outerToolCallID string, request int) string {
	return fmt.Sprintf("%s:tool-%d", outerToolCallID, request)
}

// invoke resolves, validates, approves and executes one host tool call.
func (h *hostTools) invoke(ctx context.Context, call hostCall) hostOutcome {
	toolCallID := nestedToolCallID(h.outerToolCallID, call.request)

	toolName, input, err := h.resolveCall(call)
	if err != nil {
		return failedOutcome(err)
	}

	outcome := h.invokeTool(ctx, call, toolName, input, toolCallID)
	outcome.toolName = toolName
	outcome.input = input

	return outcome
}

func (h *hostTools) invokeTool(ctx context.Context, call hostCall, toolName string, input json.RawMessage, toolCallID string) hostOutcome {
	skipApproval := false

	if call.resumed != nil {
		if isApprovalPayload(call.resumed.payload) {
			decision, err := decodeApprovalResolution(call.resumed.resolution)
			if err != nil {
				return failedOutcome(err)
			}

			if !decision.Approved {
				return failedOutcome(newApprovalDeniedError(PendingInterruption{
					ToolName:   toolName,
					ToolCallID: toolCallID,
					Input:      input,
				}, decision.Reason))
			}

			skipApproval = true
		} else {
			ctx = withResumedInterrupt(ctx, ResumedInterrupt{
				InterruptID: toolCallID + ":interrupt",
				Payload:     call.resumed.payload,
				Resolution:  call.resumed.resolution,
			})
		}
	}

	if err := ctx.Err(); err != nil {
		return failedOutcome(newAbortedError(context.Cause(ctx)))
	}

	tool, err := h.prepare(toolName, input, toolCallID)
	if err != nil {
		return failedOutcome(err)
	}

	if !skipApproval {
		outcome, approved := h.approve(ctx, tool, input, toolCallID)
		if !approved {
			return outcome
		}
	}

	return h.execute(ctx, tool, input, toolCallID)
}

// resolveCall returns the tool a host function call targets and the JSON
// of its input, as JSON.stringify writes the program's argument. An
// omitted input is an empty object.
func (h *hostTools) resolveCall(call hostCall) (string, json.RawMessage, error) {
	arguments, err := decodeRunArguments(call.argumentsJSON)
	if err != nil {
		return "", nil, newProtocolError("code mode host tool arguments are malformed: " + err.Error())
	}

	toolName, input, err := h.bindingTarget(call.bindingName, arguments)
	if err != nil {
		return "", nil, err
	}

	if input == nil {
		input = json.RawMessage("{}")
	}

	label := fmt.Sprintf("tool %q input", toolName)
	if err := checkPayloadSize(string(input), h.limits.maxToolInputBytes, label); err != nil {
		return "", nil, err
	}

	return toolName, input, nil
}

// bindingTarget maps a host function and its arguments to a tool name and
// input: __codeMode.toolN(input) calls the Nth sorted tool, and
// __codeMode.missing(name, input) a tool that is not available.
func (h *hostTools) bindingTarget(bindingName string, arguments []json.RawMessage) (string, json.RawMessage, error) {
	argument := func(position int) json.RawMessage {
		if position < len(arguments) {
			return arguments[position]
		}

		return nil
	}

	if bindingName == missingToolBinding {
		return missingToolName(argument(0)), argument(1), nil
	}

	digits, found := strings.CutPrefix(bindingName, toolBindingPrefix)

	index, err := strconv.Atoi(digits)
	if !found || err != nil || index < 0 || index >= len(h.names) || toolBindingName(index) != bindingName {
		return "", nil, newProtocolError(fmt.Sprintf("unknown code mode host function %q", bindingName))
	}

	return h.names[index], argument(0), nil
}

// missingToolName reads the name the program used for a tool that is not
// available.
func missingToolName(document json.RawMessage) string {
	var name string
	if err := json.Unmarshal(document, &name); err != nil {
		return string(document)
	}

	return name
}

// prepare finds the tool and validates the input against its schema.
func (h *hostTools) prepare(toolName string, input json.RawMessage, toolCallID string) (ai.Tool, error) {
	tool, found := h.tools[toolName]
	if !found {
		return ai.Tool{}, &Error{
			Code:       ErrorCodeTool,
			Message:    fmt.Sprintf("unknown tool: %s; available tools: %v", toolName, h.names),
			ToolName:   toolName,
			ToolCallID: toolCallID,
			Input:      input,
		}
	}

	if tool.Execute == nil {
		return ai.Tool{}, &Error{
			Code:       ErrorCodeTool,
			Message:    fmt.Sprintf("tool %q does not have an execute function", toolName),
			ToolName:   toolName,
			ToolCallID: toolCallID,
			Input:      input,
		}
	}

	if err := h.validate(toolName, input); err != nil {
		return ai.Tool{}, &Error{
			Code:       ErrorCodeTool,
			Message:    fmt.Sprintf("invalid input for tool %q: %v", toolName, err),
			ToolName:   toolName,
			ToolCallID: toolCallID,
			Input:      input,
			Cause:      err,
		}
	}

	return tool, nil
}

func (h *hostTools) validate(toolName string, input json.RawMessage) error {
	validator, err := h.validators[toolName]()
	if err != nil {
		return err
	}

	return validator.Validate(input)
}

// approve decides whether a call may run. It returns false with the
// outcome to report when the call must not run now.
func (h *hostTools) approve(ctx context.Context, tool ai.Tool, input json.RawMessage, toolCallID string) (hostOutcome, bool) {
	if tool.NeedsApproval == nil {
		return hostOutcome{}, true
	}

	needsApproval, err := tool.NeedsApproval(ctx, input, ai.ToolNeedsApprovalOptions{
		ToolCallID: toolCallID,
		Messages:   h.messages,
	})
	if err != nil {
		return failedOutcome(hostToolFailure(tool.Name, toolCallID, err)), false
	}

	if !needsApproval {
		return hostOutcome{}, true
	}

	if h.options.Approval.Mode == ApprovalModeInterrupt {
		return hostOutcome{
			interrupt: &InterruptPayload{
				Kind: ApprovalInterruptKind,
			},
		}, false
	}

	request := ApprovalRequest{
		ToolName:   tool.Name,
		Input:      input,
		ToolCallID: toolCallID,
	}

	if h.options.Approval.OnApprovalRequired == nil {
		return failedOutcome(&Error{
			Code:       ErrorCodeToolApprovalRequired,
			Message:    fmt.Sprintf("tool %q requires approval before execution", tool.Name),
			ToolName:   tool.Name,
			ToolCallID: toolCallID,
			Input:      input,
		}), false
	}

	decision, err := h.options.Approval.OnApprovalRequired(ctx, request)
	if err != nil {
		return failedOutcome(hostToolFailure(tool.Name, toolCallID, err)), false
	}

	if !decision.Approved {
		return failedOutcome(newApprovalDeniedError(PendingInterruption{
			ToolName:   tool.Name,
			ToolCallID: toolCallID,
			Input:      input,
		}, decision.Reason)), false
	}

	return hostOutcome{}, true
}

// execute runs the tool and encodes its output for the program.
func (h *hostTools) execute(ctx context.Context, tool ai.Tool, input json.RawMessage, toolCallID string) hostOutcome {
	output, err := tool.Execute(ctx, input, ai.ToolExecutionOptions{
		ToolCallID: toolCallID,
		Messages:   h.messages,
	})
	if err != nil {
		return h.executionFailure(tool.Name, toolCallID, err)
	}

	encoded, err := encodePayload(output, h.limits.maxToolOutputBytes, fmt.Sprintf("tool %q output", tool.Name))
	if err != nil {
		return failedOutcome(err)
	}

	return hostOutcome{
		output: encoded,
	}
}

// executionFailure classifies an error returned by a tool: an interrupt
// request, a code mode error from a nested invocation, or a failure that
// the program only sees as a generic host tool failure.
func (h *hostTools) executionFailure(toolName, toolCallID string, err error) hostOutcome {
	if request, ok := errors.AsType[*InterruptRequest](err); ok {
		if request.Payload.Kind == "" {
			return failedOutcome(newProtocolError("code mode interrupt payload must include a non-empty kind"))
		}

		payload := request.Payload.clone()

		return hostOutcome{
			interrupt: &payload,
		}
	}

	if codeModeErr, ok := errors.AsType[*Error](err); ok {
		return failedOutcome(codeModeErr)
	}

	return failedOutcome(hostToolFailure(toolName, toolCallID, err))
}

func hostToolFailure(toolName, toolCallID string, cause error) *Error {
	return &Error{
		Code:       ErrorCodeHostTool,
		Message:    "host tool failed",
		ToolName:   toolName,
		ToolCallID: toolCallID,
		Cause:      cause,
	}
}

func failedOutcome(err error) hostOutcome {
	return hostOutcome{
		err: err,
	}
}
