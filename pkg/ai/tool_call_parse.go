package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/schema"
	"github.com/aarock1234/ai/pkg/model"
)

// ToolCallRepairFunc attempts to repair a tool call that names an unknown
// tool or has invalid input. It returns the repaired call, or nil to keep
// the call invalid.
type ToolCallRepairFunc func(ctx context.Context, request ToolCallRepairRequest) (*model.ToolCall, error)

// ToolCallRepairRequest describes a tool call that failed to parse.
type ToolCallRepairRequest struct {
	// System is the system prompt of the step.
	System string

	// Messages are the messages of the step.
	Messages []model.Message

	// ToolCall is the call as the model produced it.
	ToolCall model.ToolCall

	// Tools are the tools available in the step.
	Tools []Tool

	// Error is the *NoSuchToolError or *InvalidToolInputError that the call
	// caused.
	Error error
}

// InputSchema returns the input schema of the named tool.
func (r ToolCallRepairRequest) InputSchema(toolName string) (json.RawMessage, bool) {
	tool, found := findTool(r.Tools, toolName)
	if !found {
		return nil, false
	}

	return tool.InputSchema, true
}

// ToolInputRefinement refines a validated tool input before the tool runs.
// The refined input must have the same shape as the original; it is used for
// execution, callbacks, stream parts and results.
type ToolInputRefinement func(ctx context.Context, input json.RawMessage) (json.RawMessage, error)

// toolCallParseRequest describes the parsing of one tool call.
type toolCallParseRequest struct {
	call   model.ToolCall
	tools  []Tool
	repair ToolCallRepairFunc
	refine map[string]ToolInputRefinement

	// system and messages describe the step for the repair function.
	system   string
	messages []model.Message
}

// parseToolCall validates a tool call against the tool's input schema,
// repairing and refining it when configured. Calls that cannot be parsed
// become invalid tool calls; the only error is the cancellation of ctx.
func parseToolCall(ctx context.Context, request toolCallParseRequest) (ToolCall, error) {
	parsed, err := parseOrRepairToolCall(ctx, request)
	if err == nil {
		return parsed, nil
	}

	if ctx.Err() != nil {
		return ToolCall{}, context.Cause(ctx)
	}

	return invalidToolCall(request.call, request.tools, err), nil
}

func parseOrRepairToolCall(ctx context.Context, request toolCallParseRequest) (ToolCall, error) {
	if len(request.tools) == 0 {
		if isProviderExecutedDynamicCall(request.call) {
			return refineToolCallInput(ctx, request.refine, parseProviderExecutedDynamicCall)(request.call)
		}

		return ToolCall{}, &NoSuchToolError{
			ToolName: request.call.ToolName,
		}
	}

	parse := refineToolCallInput(ctx, request.refine, func(call model.ToolCall) (ToolCall, error) {
		return parseDeclaredToolCall(call, request.tools)
	})

	parsed, err := parse(request.call)
	if err == nil || request.repair == nil || !isRepairableToolCallError(err) {
		return parsed, err
	}

	repaired, repairErr := request.repair(ctx, ToolCallRepairRequest{
		System:   request.system,
		Messages: request.messages,
		ToolCall: request.call,
		Tools:    request.tools,
		Error:    err,
	})
	if repairErr != nil {
		if ctx.Err() != nil {
			return ToolCall{}, context.Cause(ctx)
		}

		return ToolCall{}, &ToolCallRepairError{
			OriginalError: err,
			Cause:         repairErr,
		}
	}

	if repaired == nil {
		return ToolCall{}, err
	}

	return parse(*repaired)
}

func isRepairableToolCallError(err error) bool {
	var noSuchTool *NoSuchToolError
	var invalidInput *InvalidToolInputError

	return errors.As(err, &noSuchTool) || errors.As(err, &invalidInput)
}

func isProviderExecutedDynamicCall(call model.ToolCall) bool {
	return call.ProviderExecuted && call.Dynamic
}

// refineToolCallInput wraps a parser so that the configured refinement is
// applied to the parsed input.
func refineToolCallInput(
	ctx context.Context,
	refine map[string]ToolInputRefinement,
	parse func(call model.ToolCall) (ToolCall, error),
) func(call model.ToolCall) (ToolCall, error) {
	return func(call model.ToolCall) (ToolCall, error) {
		parsed, err := parse(call)
		if err != nil {
			return ToolCall{}, err
		}

		return applyToolInputRefinement(ctx, refine, parsed)
	}
}

// applyToolInputRefinement replaces the input of a parsed call with its
// refinement, if one is configured for the tool.
func applyToolInputRefinement(ctx context.Context, refine map[string]ToolInputRefinement, call ToolCall) (ToolCall, error) {
	refinement := refine[call.ToolName]
	if refinement == nil {
		return call, nil
	}

	refined, err := refinement(ctx, call.Input)
	if err != nil {
		return ToolCall{}, fmt.Errorf("refine input of tool %q: %w", call.ToolName, err)
	}

	call.schemaInput = call.Input
	call.Input = refined

	return call, nil
}

// parseDeclaredToolCall validates a call of one of the declared tools.
// Provider-executed dynamic calls of undeclared tools are accepted as is.
func parseDeclaredToolCall(call model.ToolCall, tools []Tool) (ToolCall, error) {
	tool, found := findTool(tools, call.ToolName)
	if !found {
		if isProviderExecutedDynamicCall(call) {
			return parseProviderExecutedDynamicCall(call)
		}

		return ToolCall{}, &NoSuchToolError{
			ToolName:       call.ToolName,
			AvailableTools: toolNames(tools),
		}
	}

	input, err := validateToolInput(tool, call.Input)
	if err != nil {
		return ToolCall{}, &InvalidToolInputError{
			ToolName:  call.ToolName,
			ToolInput: call.Input,
			Cause:     err,
		}
	}

	return ToolCall{
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		Input:            input,
		ProviderExecuted: call.ProviderExecuted,
		Dynamic:          tool.kind() == ToolKindDynamic,
		Title:            tool.Title,
		ProviderMetadata: call.ProviderMetadata,
		ToolMetadata:     tool.Metadata,
	}, nil
}

// validateToolInput checks the input against the tool's schema. Empty input,
// which many models produce for tools without parameters, is validated and
// returned as an empty object.
func validateToolInput(tool Tool, input string) (json.RawMessage, error) {
	raw := json.RawMessage(strings.TrimSpace(input))
	if len(raw) == 0 {
		raw = json.RawMessage(`{}`)
	}

	if !json.Valid(raw) {
		return nil, errors.New("input is not valid JSON")
	}

	validator, err := schema.Compile(tool.InputSchema)
	if err != nil {
		return nil, err
	}

	if err := validator.Validate(raw); err != nil {
		return nil, err
	}

	return raw, nil
}

// parseProviderExecutedDynamicCall accepts any JSON input for a dynamic tool
// that the provider runs itself.
func parseProviderExecutedDynamicCall(call model.ToolCall) (ToolCall, error) {
	raw := json.RawMessage(strings.TrimSpace(call.Input))
	if len(raw) == 0 {
		raw = json.RawMessage(`{}`)
	}

	if !json.Valid(raw) {
		return ToolCall{}, &InvalidToolInputError{
			ToolName:  call.ToolName,
			ToolInput: call.Input,
			Cause:     errors.New("input is not valid JSON"),
		}
	}

	return ToolCall{
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		Input:            raw,
		ProviderExecuted: true,
		Dynamic:          true,
		ProviderMetadata: call.ProviderMetadata,
	}, nil
}

// invalidToolCall records a call that could not be parsed. Its input is the
// parsed JSON when possible and otherwise the raw text as a JSON string.
func invalidToolCall(call model.ToolCall, tools []Tool, cause error) ToolCall {
	invalid := ToolCall{
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		Input:            invalidToolInput(call.Input),
		ProviderExecuted: call.ProviderExecuted,
		Dynamic:          true,
		Invalid:          true,
		Error:            cause,
		ProviderMetadata: call.ProviderMetadata,
	}

	if tool, found := findTool(tools, call.ToolName); found {
		invalid.Title = tool.Title
		invalid.ToolMetadata = tool.Metadata
	}

	return invalid
}

func invalidToolInput(input string) json.RawMessage {
	trimmed := bytes.TrimSpace([]byte(input))
	if len(trimmed) > 0 && json.Valid(trimmed) {
		return trimmed
	}

	encoded, err := json.Marshal(input)
	if err != nil {
		return json.RawMessage(`null`)
	}

	return encoded
}

func toolNames(tools []Tool) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}

	return names
}
