package harness

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"sync"

	"github.com/aarock1234/ai/internal/schema"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// invalidToolInputMessage is reported to the runtime for a call whose input
// failed validation.
const invalidToolInputMessage = "Tool input validation failed."

// errInvalidToolInput reports a host tool call whose input failed
// validation.
var errInvalidToolInput = errors.New("tool input validation failed")

// toolCallParser validates tool calls against tool input schemas, caching
// the compiled schemas. It is safe for concurrent use.
type toolCallParser struct {
	mu         sync.Mutex
	validators map[string]*schema.Validator
}

// parse validates a runtime tool call, as the ai package validates model
// tool calls. Calls that name an unknown tool or have invalid input become
// invalid dynamic calls. With skipSchemas, provider-executed dynamic calls
// are accepted without schema validation.
func (p *toolCallParser) parse(call model.ToolCall, tools []ai.Tool, skipSchemas bool) ai.ToolCall {
	if skipSchemas || len(tools) == 0 {
		if isProviderExecutedDynamicCall(call) {
			return p.parseUndeclared(call, tools)
		}

		return invalidToolCall(call, tools, &ai.NoSuchToolError{
			ToolName: call.ToolName,
		})
	}

	tool, found := findTool(tools, call.ToolName)
	if !found {
		if isProviderExecutedDynamicCall(call) {
			return p.parseUndeclared(call, tools)
		}

		return invalidToolCall(call, tools, &ai.NoSuchToolError{
			ToolName:       call.ToolName,
			AvailableTools: toolNames(tools),
		})
	}

	input, err := p.validateInput(*tool, call.Input)
	if err != nil {
		return invalidToolCall(call, tools, &ai.InvalidToolInputError{
			ToolName:  call.ToolName,
			ToolInput: call.Input,
			Cause:     err,
		})
	}

	return ai.ToolCall{
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		Input:            input,
		ProviderExecuted: call.ProviderExecuted,
		Dynamic:          tool.Kind == ai.ToolKindDynamic,
		Title:            tool.Title,
		ProviderMetadata: call.ProviderMetadata,
		ToolMetadata:     tool.Metadata,
	}
}

// parseUndeclared accepts any JSON input of a provider-executed dynamic
// call, such as a tool of an MCP server the runtime connected to.
func (p *toolCallParser) parseUndeclared(call model.ToolCall, tools []ai.Tool) ai.ToolCall {
	input := normalizedToolInput(call.Input)
	if !json.Valid(input) {
		return invalidToolCall(call, tools, &ai.InvalidToolInputError{
			ToolName:  call.ToolName,
			ToolInput: call.Input,
			Cause:     errors.New("input is not valid JSON"),
		})
	}

	return ai.ToolCall{
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		Input:            input,
		ProviderExecuted: true,
		Dynamic:          true,
		ProviderMetadata: call.ProviderMetadata,
	}
}

// validateInput checks input against the tool's schema. Empty input is an
// empty object.
func (p *toolCallParser) validateInput(tool ai.Tool, input string) (json.RawMessage, error) {
	raw := normalizedToolInput(input)
	if !json.Valid(raw) {
		return nil, errors.New("input is not valid JSON")
	}

	validator, err := p.validator(tool)
	if err != nil {
		return nil, err
	}

	if err := validator.Validate(raw); err != nil {
		return nil, err
	}

	return raw, nil
}

// validator returns the compiled input schema of a tool.
func (p *toolCallParser) validator(tool ai.Tool) (*schema.Validator, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if cached, ok := p.validators[tool.Name]; ok {
		return cached, nil
	}

	compiled, err := schema.Compile(tool.InputSchema)
	if err != nil {
		return nil, err
	}

	if p.validators == nil {
		p.validators = make(map[string]*schema.Validator)
	}

	p.validators[tool.Name] = compiled

	return compiled, nil
}

// invalidToolCall records a call that could not be parsed. Its input is the
// parsed JSON when possible and the raw text as a JSON string otherwise.
func invalidToolCall(call model.ToolCall, tools []ai.Tool, cause error) ai.ToolCall {
	invalid := ai.ToolCall{
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

// invalidToolInput returns the JSON of an invalid call's input.
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

// normalizedToolInput trims input and replaces empty input with an empty
// object.
func normalizedToolInput(input string) json.RawMessage {
	raw := json.RawMessage(strings.TrimSpace(input))
	if len(raw) == 0 {
		return json.RawMessage(`{}`)
	}

	return raw
}

// isProviderExecutedDynamicCall reports a dynamic call the runtime runs.
func isProviderExecutedDynamicCall(call model.ToolCall) bool {
	return call.ProviderExecuted && call.Dynamic
}

// findTool returns the named tool.
func findTool(tools []ai.Tool, name string) (*ai.Tool, bool) {
	for index := range tools {
		if tools[index].Name == name {
			return &tools[index], true
		}
	}

	return nil, false
}

// hasTool reports whether tools declares the named tool.
func hasTool(tools []ai.Tool, name string) bool {
	_, found := findTool(tools, name)

	return found
}

// toolNames returns the names of tools in order.
func toolNames(tools []ai.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}

	return names
}
