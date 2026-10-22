package ai

import (
	"context"
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// ToolKind distinguishes how a tool is declared to the model.
type ToolKind string

// Tool kinds.
const (
	// ToolKindFunction is an application tool whose input and output types
	// are known at development time. The empty kind means the same.
	ToolKindFunction ToolKind = "function"

	// ToolKindDynamic is an application tool whose types are only known at
	// runtime, such as a tool discovered from an MCP server.
	ToolKindDynamic ToolKind = "dynamic"

	// ToolKindProvider is a tool defined by the provider, such as web search
	// or a provider-specified shell tool.
	ToolKindProvider ToolKind = "provider"
)

// Tool is a tool the model may call. Construct function tools with NewTool or
// as a literal, dynamic tools with DynamicTool, and provider tools with
// ProviderExecutedTool or ProviderDefinedTool.
type Tool struct {
	// Name is the name the model uses to call the tool. It must be unique
	// within a call.
	Name string

	// Description tells the model what the tool does and when to use it.
	Description string

	// Title is an optional display title.
	Title string

	// InputSchema is the JSON Schema of the input object. The model uses it
	// to produce the input, and calls are validated against it before
	// execution. Provider tools may leave it empty.
	InputSchema json.RawMessage

	// OutputSchema is the optional JSON Schema of the tool's output. It is
	// not sent to the model; ui.ValidateMessages checks stored outputs
	// against it, and tools that present other tools, such as code mode,
	// use it to describe their results.
	OutputSchema json.RawMessage

	// Strict requests strict schema adherence from providers that support
	// it. Nil leaves the provider default.
	Strict *bool

	// InputExamples are example input objects for providers that accept
	// them.
	InputExamples []json.RawMessage

	// Metadata describes the tool itself, for example its source. It is not
	// sent to the model; it is copied onto the tool's calls and results.
	Metadata map[string]any

	// DeferLoading hides the tool from the model until a ToolSearch tool of
	// the same generation discovers it; it is declared from the following
	// step on. It is not sent to providers: provider-side deferral, such as
	// Anthropic's, is configured through ProviderOptions.
	DeferLoading bool

	// Caller lets this tool call other tools programmatically. The tools it
	// may call are configured with the generation's ToolCallers option.
	Caller ToolCaller

	// ProviderOptions are passed to the provider with the tool declaration.
	ProviderOptions model.ProviderOptions

	// Kind selects how the tool is declared. The empty kind is
	// ToolKindFunction.
	Kind ToolKind

	// Provider identifies the provider tool and its configuration. It is
	// required when Kind is ToolKindProvider and ignored otherwise.
	Provider *model.ProviderTool

	// ProviderExecuted marks provider tools that the provider runs itself.
	// Their results arrive in the model response and Execute is not called.
	ProviderExecuted bool

	// SupportsDeferredResults marks provider-executed tools whose results
	// may arrive in a later response than their calls.
	SupportsDeferredResults bool

	// NeedsApproval reports whether a call must be approved before it runs.
	// Nil means no approval is needed.
	NeedsApproval ToolNeedsApprovalFunc

	// Execute runs the tool. Nil means the application handles calls itself
	// and the generation stops at them.
	Execute ToolExecuteFunc

	// ToModelOutput converts the tool's output into the result sent to the
	// model. Nil sends strings as text and other values as JSON.
	ToModelOutput ToolModelOutputFunc

	// OnInputStart is called when the model starts producing the input. Outside
	// streaming it is called immediately before OnInputAvailable.
	OnInputStart ToolInputStartFunc

	// OnInputDelta is called for each streamed chunk of input text.
	OnInputDelta ToolInputDeltaFunc

	// OnInputAvailable is called when the complete input is available, even
	// when the tool has no Execute function.
	OnInputAvailable ToolInputAvailableFunc

	// err records a construction failure, such as a schema that could not be
	// generated. Generation reports it before calling the model.
	err error

	// search marks a tool created by ToolSearch.
	search *toolSearchConfig
}

// ToolExecuteFunc runs a tool with the JSON input from the model and returns
// its output. The output must encode to JSON unless the tool converts it with
// ToModelOutput.
type ToolExecuteFunc func(ctx context.Context, input json.RawMessage, options ToolExecutionOptions) (any, error)

// ToolNeedsApprovalFunc reports whether a tool call requires approval.
type ToolNeedsApprovalFunc func(ctx context.Context, input json.RawMessage, options ToolNeedsApprovalOptions) (bool, error)

// ToolModelOutputFunc converts a tool output into the result sent to the
// model.
type ToolModelOutputFunc func(ctx context.Context, event ToolOutputEvent[json.RawMessage, any]) (model.ToolResultOutput, error)

// ToolInputStartFunc observes the start of a tool call's input.
type ToolInputStartFunc func(ctx context.Context, options ToolExecutionOptions)

// ToolInputDeltaFunc observes a streamed chunk of a tool call's input.
type ToolInputDeltaFunc func(ctx context.Context, event ToolInputDeltaEvent)

// ToolInputAvailableFunc observes a tool call's complete input.
type ToolInputAvailableFunc func(ctx context.Context, event ToolInputEvent[json.RawMessage])

// ToolExecutionOptions describe the tool call being executed.
type ToolExecutionOptions struct {
	ToolCallID string

	// Messages are the messages sent to the model in the step that produced
	// the call. They exclude the system prompt and the assistant response
	// that contains the call.
	Messages []model.Message

	// preliminary receives preliminary outputs. Nil discards them.
	preliminary func(output any)
}

// Preliminary reports an intermediate output while the tool is still
// running, such as progress for a long task. Streaming calls emit it as a
// preliminary tool result, and ExecuteTool passes it to OnPreliminary; the
// value returned by Execute remains the final output. It is safe to call
// from the goroutine running Execute.
func (o ToolExecutionOptions) Preliminary(output any) {
	if o.preliminary != nil {
		o.preliminary(output)
	}
}

// ToolNeedsApprovalOptions describe the tool call awaiting an approval
// decision.
type ToolNeedsApprovalOptions struct {
	ToolCallID string

	// Messages are the messages sent to the model in the step that produced
	// the call.
	Messages []model.Message
}

// ToolInputDeltaEvent is a streamed chunk of a tool call's input.
type ToolInputDeltaEvent struct {
	ToolExecutionOptions

	InputTextDelta string
}

// ToolInputEvent is a tool call's complete input.
type ToolInputEvent[Input any] struct {
	ToolExecutionOptions

	Input Input
}

// ToolOutputEvent is a tool call's output, to be converted for the model.
type ToolOutputEvent[Input, Output any] struct {
	ToolCallID string
	Input      Input
	Output     Output
}

// DynamicTool marks tool as a dynamic tool, whose input and output types are
// only known at runtime.
func DynamicTool(tool Tool) Tool {
	tool.Kind = ToolKindDynamic

	return tool
}

// ProviderExecutedTool declares a provider tool that the provider runs
// itself, such as web search or code execution.
func ProviderExecutedTool(definition model.ProviderTool) Tool {
	return Tool{
		Name:             definition.Name,
		Kind:             ToolKindProvider,
		Provider:         &definition,
		ProviderExecuted: true,
	}
}

// ProviderDefinedTool declares a provider tool whose input format the
// provider defines but that the application runs, such as a local shell
// tool. Set Execute on the result to run it automatically.
func ProviderDefinedTool(definition model.ProviderTool) Tool {
	return Tool{
		Name:     definition.Name,
		Kind:     ToolKindProvider,
		Provider: &definition,
	}
}

// kind returns the tool kind, treating the empty kind as a function tool.
func (t Tool) kind() ToolKind {
	if t.Kind == "" {
		return ToolKindFunction
	}

	return t.Kind
}
