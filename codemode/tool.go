package codemode

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/ai"
)

// ToolDiscovery selects how the host tools are presented to the model.
type ToolDiscovery string

// Tool discovery modes. The empty mode means ToolDiscoveryDescription.
const (
	// ToolDiscoveryDescription declares the host tools in the code mode
	// tool's description.
	ToolDiscoveryDescription ToolDiscovery = "description"

	// ToolDiscoveryConversation keeps the code mode tool's declaration
	// stable and announces the current host tools in a user message.
	ToolDiscoveryConversation ToolDiscovery = "conversation"
)

// CallerToolOptions configure NewCallerTool.
type CallerToolOptions struct {
	Options

	ToolDiscovery ToolDiscovery
}

// toolInput is the input of the code mode tool.
type toolInput struct {
	JS string `json:"js"`
}

// toolInputSchema is the JSON Schema of toolInput.
type toolInputSchema struct {
	Type                 string                        `json:"type"`
	Properties           map[string]toolPropertySchema `json:"properties"`
	Required             []string                      `json:"required"`
	AdditionalProperties bool                          `json:"additionalProperties"`
}

type toolPropertySchema struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}

// NewTool returns a tool that runs the TypeScript the model writes in a
// sandbox where tools are available as tools.name(input). The tool's
// description declares them. Its output is the value the program returns,
// or an *Interrupt when a host tool interrupts the invocation.
func NewTool(name string, tools []ai.Tool, options Options) ai.Tool {
	return newCodeModeTool(name, tools, options, ToolDiscoveryDescription)
}

// NewCallerTool returns a code mode tool whose host tools are bound by the
// generation: list its name in ai.GenerateTextOptions.ToolCallers for the
// tools it may call. With ToolDiscoveryConversation, the bound tools are
// announced in a user message, which keeps the tool's declaration stable.
func NewCallerTool(name string, options CallerToolOptions) ai.Tool {
	discovery := options.ToolDiscovery
	if discovery == "" {
		discovery = ToolDiscoveryDescription
	}

	caller := ai.LocalToolCaller{
		Bind: func(tools []ai.Tool) ai.Tool {
			return newCodeModeTool(name, tools, options.Options, discovery)
		},
	}

	if discovery == ToolDiscoveryConversation {
		caller.PrepareModelMessage = buildCatalogMessage
	}

	tool := newCodeModeTool(name, nil, options.Options, discovery)
	tool.Caller = caller

	return tool
}

func newCodeModeTool(name string, tools []ai.Tool, options Options, discovery ToolDiscovery) ai.Tool {
	return ai.Tool{
		Name:        name,
		Description: buildToolDescription(tools, discovery),
		InputSchema: codeModeInputSchema(),
		Execute: func(ctx context.Context, input json.RawMessage, execution ai.ToolExecutionOptions) (any, error) {
			return executeCodeMode(ctx, codeModeExecution{
				input:     input,
				tools:     tools,
				options:   options,
				execution: execution,
			})
		},
	}
}

// codeModeExecution is one call of the code mode tool.
type codeModeExecution struct {
	input     json.RawMessage
	tools     []ai.Tool
	options   Options
	execution ai.ToolExecutionOptions
}

func executeCodeMode(ctx context.Context, call codeModeExecution) (any, error) {
	var input toolInput
	if err := json.Unmarshal(call.input, &input); err != nil {
		return nil, fmt.Errorf("code mode: decode input: %w", err)
	}

	result, err := Run(ctx, RunOptions{
		JS:         input.JS,
		Tools:      call.tools,
		ToolCallID: call.execution.ToolCallID,
		Messages:   call.execution.Messages,
		Options:    call.options,
	})
	if err != nil {
		return nil, err
	}

	if result.Interrupt != nil {
		return result.Interrupt, nil
	}

	if result.Output == nil {
		return nil, nil
	}

	return result.Output, nil
}

func codeModeInputSchema() json.RawMessage {
	inputSchema := toolInputSchema{
		Type: "object",
		Properties: map[string]toolPropertySchema{
			"js": {
				Type:        "string",
				Description: jsInputDescription(),
			},
		},
		Required: []string{
			"js",
		},
		AdditionalProperties: false,
	}

	encoded, err := json.Marshal(inputSchema)
	if err != nil {
		panic(fmt.Sprintf("code mode: encode input schema: %v", err))
	}

	return encoded
}
