package openai

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// assistantToolCall converts a tool call from an earlier turn.
func (c *responsesInputConverter) assistantToolCall(part model.ToolCallPart) error {
	metadata, err := responsesProviderValue[ResponsesToolCallMetadata](part.ProviderOptions, c.options.namespace)
	if err != nil {
		return err
	}

	if c.parallelCall(metadata.ParallelToolCall) {
		return nil
	}

	id := metadata.ItemID

	if metadata.Caller != nil && metadata.Caller.Type == ResponsesCallerProgram {
		c.programmaticToolCallIDs[part.ToolCallID] = true
	}

	if c.options.hasConversation && id != "" {
		return nil
	}

	providerName := c.options.toolNames.ToProvider(part.ToolName)

	if c.isToolSearch(part.ToolName) {
		return c.toolSearchCall(part, id)
	}

	if providerName == "programmatic_tool_calling" {
		return c.programCall(part, id)
	}

	if part.ProviderExecuted {
		if c.options.store && id != "" {
			c.addReference(id)
		}

		// Without stored responses, shell calls are rebuilt together with
		// their shell call outputs.
		rebuildsShellCall := !c.options.store && c.options.hasShellTool && providerName == "shell"
		if !rebuildsShellCall {
			return nil
		}
	}

	isProviderDefined := c.isProviderDefinedCall(providerName)

	// Items of the previous response chain must not be sent again.
	if c.options.hasPreviousResponseID && c.options.store && id != "" && isProviderDefined {
		return nil
	}

	// Stored provider-defined calls are sent as references. Plain function
	// calls are not: their outputs refer to them by call ID, which the API
	// cannot reconcile with a referenced item ID.
	if c.options.store && id != "" && isProviderDefined {
		c.addReference(id)

		return nil
	}

	return c.clientToolCall(part, providerName, id, metadata)
}

// isProviderDefinedCall reports whether a tool call belongs to a
// provider-defined tool that the application executes.
func (c *responsesInputConverter) isProviderDefinedCall(providerName string) bool {
	switch {
	case c.options.hasLocalShellTool && providerName == "local_shell":
		return true
	case c.options.hasShellTool && providerName == "shell":
		return true
	case c.options.hasApplyPatchTool && providerName == "apply_patch":
		return true
	case c.options.hasComputerTool && providerName == "computer":
		return true
	default:
		return c.options.customToolNames[providerName]
	}
}

// clientToolCall sends a client-executed tool call with its content.
func (c *responsesInputConverter) clientToolCall(
	part model.ToolCallPart,
	providerName string,
	id string,
	metadata ResponsesToolCallMetadata,
) error {
	switch {
	case c.options.hasLocalShellTool && providerName == "local_shell":
		return c.localShellCall(part, id)
	case c.options.hasShellTool && providerName == "shell":
		return c.shellCall(part, id)
	case c.options.hasApplyPatchTool && providerName == "apply_patch":
		return c.applyPatchCall(part, id)
	case c.options.hasComputerTool && providerName == "computer":
		return c.computerCall(part, id)
	case c.options.customToolNames[providerName]:
		c.add(responsesCustomToolCallInput{
			Type:   responsesCustomToolCall,
			ID:     id,
			CallID: part.ToolCallID,
			Name:   providerName,
			Input:  responsesCustomToolInput(part.Input),
			Async:  metadata.Async,
		})

		return nil
	default:
		c.add(responsesFunctionCallInput{
			Type:      responsesFunctionCall,
			CallID:    part.ToolCallID,
			Name:      providerName,
			Arguments: responsesToolArguments(part.Input),
			Async:     metadata.Async,
			Namespace: metadata.Namespace,
			Caller:    responsesCallerOf(metadata.Caller),
		})

		return nil
	}
}

func (c *responsesInputConverter) toolSearchCall(part model.ToolCallPart, id string) error {
	if c.options.store && id != "" {
		c.addReference(id)

		return nil
	}

	input, err := responsesDecodeToolInput[ToolSearchInput](part.Input)
	if err != nil {
		return err
	}

	execution := "server"
	if input.CallID != nil {
		execution = "client"
	}

	c.add(responsesToolSearchCallInput{
		Type:      responsesToolSearchCall,
		ID:        responsesFirstNonEmpty(id, part.ToolCallID),
		Execution: execution,
		CallID:    input.CallID,
		Status:    "completed",
		Arguments: input.Arguments,
	})

	return nil
}

func (c *responsesInputConverter) programCall(part model.ToolCallPart, id string) error {
	if c.options.store && id != "" {
		c.addReference(id)

		return nil
	}

	input, err := responsesDecodeToolInput[ProgrammaticToolCallingInput](part.Input)
	if err != nil {
		return err
	}

	c.add(responsesProgramInput{
		Type:        responsesProgram,
		ID:          responsesFirstNonEmpty(id, part.ToolCallID),
		CallID:      part.ToolCallID,
		Code:        input.Code,
		Fingerprint: input.Fingerprint,
	})

	return nil
}

func (c *responsesInputConverter) localShellCall(part model.ToolCallPart, id string) error {
	input, err := responsesDecodeToolInput[LocalShellInput](part.Input)
	if err != nil {
		return err
	}

	c.add(responsesLocalShellCallInput{
		Type:   responsesLocalShellCall,
		ID:     id,
		CallID: part.ToolCallID,
		Action: responsesLocalShellAction{
			Type:             "exec",
			Command:          input.Action.Command,
			TimeoutMs:        input.Action.TimeoutMs,
			User:             input.Action.User,
			WorkingDirectory: input.Action.WorkingDirectory,
			Env:              input.Action.Env,
		},
	})

	return nil
}

func (c *responsesInputConverter) shellCall(part model.ToolCallPart, id string) error {
	input, err := responsesDecodeToolInput[ShellInput](part.Input)
	if err != nil {
		return err
	}

	c.add(responsesShellCallInput{
		Type:   responsesShellCall,
		ID:     id,
		CallID: part.ToolCallID,
		Status: "completed",
		Action: responsesShellAction(input.Action),
	})

	return nil
}

func (c *responsesInputConverter) applyPatchCall(part model.ToolCallPart, id string) error {
	input, err := responsesDecodeToolInput[ApplyPatchInput](part.Input)
	if err != nil {
		return err
	}

	c.add(responsesApplyPatchCallInput{
		Type:      responsesApplyPatchCall,
		ID:        id,
		CallID:    input.CallID,
		Status:    "completed",
		Operation: input.Operation,
	})

	return nil
}

func (c *responsesInputConverter) computerCall(part model.ToolCallPart, id string) error {
	input, err := responsesDecodeToolInput[ComputerInput](part.Input)
	if err != nil {
		return err
	}

	actions := make([]responsesComputerAction, 0, len(input.Actions))
	for _, action := range input.Actions {
		actions = append(actions, responsesComputerAction(action))
	}

	safetyChecks := input.PendingSafetyChecks
	if safetyChecks == nil {
		safetyChecks = []ComputerSafetyCheck{}
	}

	c.add(responsesComputerCallInput{
		Type:                responsesComputerCall,
		ID:                  id,
		CallID:              part.ToolCallID,
		Status:              input.Status,
		Actions:             actions,
		PendingSafetyChecks: safetyChecks,
	})

	return nil
}

// assistantToolResult converts the result of a provider-executed tool.
func (c *responsesInputConverter) assistantToolResult(part model.ToolResultPart) error {
	// Denied approvals produce synthetic results that OpenAI never stored.
	if responsesIsDeniedOutput(part.Output) {
		return nil
	}

	if c.options.hasConversation {
		return nil
	}

	options, err := responsesProviderValue[ResponsesToolResultOptions](part.ProviderOptions, c.options.namespace)
	if err != nil {
		return err
	}

	itemID := responsesFirstNonEmpty(options.ItemID, part.ToolCallID)
	providerName := c.options.toolNames.ToProvider(part.ToolName)

	if providerName == "web_search" || providerName == "web_search_preview" {
		if c.options.store {
			c.addReference(itemID)

			return nil
		}

		if call, ok := responsesWebSearchCallInputOf(part.Output, itemID); ok {
			c.add(call)

			return nil
		}
	}

	switch {
	case c.isToolSearch(part.ToolName):
		return c.hostedToolSearchOutput(part, itemID)
	case providerName == "programmatic_tool_calling":
		return c.programOutput(part, itemID)
	case c.options.hasShellTool && providerName == "shell":
		// Shell outputs have their own item IDs, which the call's metadata
		// does not carry, so they are always rebuilt.
		return c.shellOutput(part)
	}

	if c.options.store {
		c.addReference(itemID)

		return nil
	}

	c.warnOther(fmt.Sprintf("Results for OpenAI tool %s are not sent to the API when store is false", part.ToolName))

	return nil
}

func (c *responsesInputConverter) hostedToolSearchOutput(part model.ToolResultPart, itemID string) error {
	if c.options.store {
		c.addReference(itemID)

		return nil
	}

	output, ok := part.Output.(model.JSONOutput)
	if !ok {
		return nil
	}

	decoded, err := responsesDecodeToolOutput[ToolSearchOutput](output)
	if err != nil {
		return err
	}

	c.add(responsesToolSearchOutputInput{
		Type:      responsesToolSearchOutput,
		ID:        itemID,
		Execution: "server",
		Status:    "completed",
		Tools:     responsesNonNilTools(decoded.Tools),
	})

	return nil
}

func (c *responsesInputConverter) programOutput(part model.ToolResultPart, itemID string) error {
	if c.options.store {
		c.addReference(itemID)

		return nil
	}

	output, ok := part.Output.(model.JSONOutput)
	if !ok {
		return nil
	}

	decoded, err := responsesDecodeToolOutput[ProgrammaticToolCallingResult](output)
	if err != nil {
		return err
	}

	if err := responsesCheckEnum("program status", decoded.Status, "completed", "incomplete"); err != nil {
		return err
	}

	c.add(responsesProgramOutputInput{
		Type:   responsesProgramOutput,
		ID:     itemID,
		CallID: part.ToolCallID,
		Result: decoded.Result,
		Status: decoded.Status,
	})

	return nil
}

func (c *responsesInputConverter) shellOutput(part model.ToolResultPart) error {
	output, ok := part.Output.(model.JSONOutput)
	if !ok {
		return nil
	}

	decoded, err := responsesDecodeToolOutput[ShellOutput](output)
	if err != nil {
		return err
	}

	c.add(responsesShellCallOutputInput{
		Type:            responsesShellCallOutput,
		CallID:          part.ToolCallID,
		MaxOutputLength: decoded.MaxOutputLength,
		Output:          responsesShellEntriesOf(decoded.Output),
	})

	return nil
}

// responsesIsDeniedOutput reports whether an output records a denied
// execution, directly or as a JSON object of type "execution-denied".
func responsesIsDeniedOutput(output model.ToolResultOutput) bool {
	switch output := output.(type) {
	case model.ExecutionDeniedOutput:
		return true
	case model.JSONOutput:
		var header struct {
			Type string `json:"type"`
		}

		return json.Unmarshal(output.Value, &header) == nil && header.Type == "execution-denied"
	default:
		return false
	}
}

// responsesWebSearchCallInputOf rebuilds a web search call from its result.
// It reports false when the result has no recognizable action.
func responsesWebSearchCallInputOf(output model.ToolResultOutput, id string) (responsesWebSearchCallInput, bool) {
	jsonOutput, ok := output.(model.JSONOutput)
	if !ok {
		return responsesWebSearchCallInput{}, false
	}

	var result WebSearchResult
	if err := json.Unmarshal(jsonOutput.Value, &result); err != nil || result.Action == nil {
		return responsesWebSearchCallInput{}, false
	}

	var action responsesWebSearchAction

	switch result.Action.Type {
	case WebSearchActionSearch:
		action = responsesWebSearchAction{
			Type:    "search",
			Query:   responsesOptionalString(result.Action.Query),
			Queries: result.Action.Queries,
			Sources: result.Sources,
		}
	case WebSearchActionOpenPage:
		action = responsesWebSearchAction{
			Type: "open_page",
			URL:  responsesOptionalString(result.Action.URL),
		}
	case WebSearchActionFindInPage:
		action = responsesWebSearchAction{
			Type:    "find_in_page",
			URL:     responsesOptionalString(result.Action.URL),
			Pattern: responsesOptionalString(result.Action.Pattern),
		}
	default:
		return responsesWebSearchCallInput{}, false
	}

	return responsesWebSearchCallInput{
		Type:   responsesWebSearchCall,
		ID:     id,
		Status: "completed",
		Action: action,
	}, true
}

// responsesToolArguments returns the JSON arguments of a function call, or
// an empty object when there are none.
func responsesToolArguments(input json.RawMessage) string {
	if len(bytes.TrimSpace(input)) == 0 {
		return "{}"
	}

	return string(input)
}

// responsesCustomToolInput returns the text input of a custom tool call,
// which is stored as a JSON string.
func responsesCustomToolInput(input json.RawMessage) string {
	var text string
	if err := json.Unmarshal(input, &text); err == nil {
		return text
	}

	return string(input)
}

// responsesDecodeToolInput decodes a tool call input. Inputs stored as a
// JSON string containing JSON are unwrapped first.
func responsesDecodeToolInput[T any](input json.RawMessage) (T, error) {
	var value T

	var text string
	if err := json.Unmarshal(input, &text); err == nil {
		input = json.RawMessage(text)
	}

	if err := json.Unmarshal(input, &value); err != nil {
		return value, fmt.Errorf("%w: decode %T tool input: %w", model.ErrInvalidPrompt, value, err)
	}

	return value, nil
}

func responsesCallerOf(caller *ResponsesToolCaller) *responsesCaller {
	if caller == nil {
		return nil
	}

	return &responsesCaller{
		Type:     caller.Type,
		CallerID: caller.CallerID,
	}
}

// responsesShellEntriesOf converts shell command outputs to the API format.
func responsesShellEntriesOf(outputs []ShellCommandOutput) []responsesShellOutputEntry {
	entries := make([]responsesShellOutputEntry, 0, len(outputs))

	for _, output := range outputs {
		outcome := responsesShellOutcome{
			Type: "timeout",
		}

		if output.Outcome.Type != "timeout" {
			outcome = responsesShellOutcome{
				Type:     "exit",
				ExitCode: output.Outcome.ExitCode,
			}
		}

		entries = append(entries, responsesShellOutputEntry{
			Stdout:  output.Stdout,
			Stderr:  output.Stderr,
			Outcome: outcome,
		})
	}

	return entries
}

func responsesNonNilTools(tools []json.RawMessage) []json.RawMessage {
	if tools == nil {
		return []json.RawMessage{}
	}

	return tools
}
