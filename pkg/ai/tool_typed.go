package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/schema"
	"github.com/aarock1234/ai/pkg/model"
)

// ToolDefinition describes a function tool with Go input and output types,
// for NewTool. The fields match those of Tool, with typed callbacks.
type ToolDefinition[Input, Output any] struct {
	Name        string
	Description string
	Title       string

	// Strict requests strict schema adherence. Nil leaves the provider
	// default.
	Strict *bool

	// InputExamples are example inputs for providers that accept them.
	InputExamples []Input

	// Metadata describes the tool itself; it is not sent to the model.
	Metadata map[string]any

	DeferLoading    bool
	ProviderOptions model.ProviderOptions

	// NeedsApproval reports whether a call must be approved before it runs.
	NeedsApproval func(ctx context.Context, input Input, options ToolNeedsApprovalOptions) (bool, error)

	// Execute runs the tool. Nil means the application handles calls itself.
	Execute func(ctx context.Context, input Input, options ToolExecutionOptions) (Output, error)

	// ToModelOutput converts the output into the result sent to the model.
	ToModelOutput func(ctx context.Context, event ToolOutputEvent[Input, Output]) (model.ToolResultOutput, error)

	OnInputStart     ToolInputStartFunc
	OnInputDelta     ToolInputDeltaFunc
	OnInputAvailable func(ctx context.Context, event ToolInputEvent[Input])
}

// NewTool returns a function tool whose input schema is generated from
// Input and whose output schema is generated from Output. Inputs from the
// model are decoded into Input before the typed callbacks run. An input
// schema or example that cannot be encoded is recorded in the tool and
// reported by the generation call that uses it. The output schema is
// optional: it is left empty when Output is an interface type, such as any,
// or when its JSON form cannot be inferred.
func NewTool[Input, Output any](definition ToolDefinition[Input, Output]) Tool {
	tool := Tool{
		Name:             definition.Name,
		Description:      definition.Description,
		Title:            definition.Title,
		Strict:           definition.Strict,
		Metadata:         definition.Metadata,
		DeferLoading:     definition.DeferLoading,
		ProviderOptions:  definition.ProviderOptions,
		Kind:             ToolKindFunction,
		NeedsApproval:    typedNeedsApproval(definition.NeedsApproval),
		Execute:          typedExecute(definition.Execute),
		ToModelOutput:    typedModelOutput(definition.ToModelOutput),
		OnInputStart:     definition.OnInputStart,
		OnInputDelta:     definition.OnInputDelta,
		OnInputAvailable: typedInputAvailable(definition.OnInputAvailable),
	}

	inputSchema, err := schema.Generate[Input]()
	if err != nil {
		tool.err = fmt.Errorf("generate input schema for tool %q: %w", definition.Name, err)

		return tool
	}

	tool.InputSchema = inputSchema
	tool.OutputSchema = generateOutputSchema[Output]()

	examples, err := encodeInputExamples(definition.InputExamples)
	if err != nil {
		tool.err = fmt.Errorf("encode input examples for tool %q: %w", definition.Name, err)

		return tool
	}

	tool.InputExamples = examples

	return tool
}

func typedNeedsApproval[Input any](
	needsApproval func(ctx context.Context, input Input, options ToolNeedsApprovalOptions) (bool, error),
) ToolNeedsApprovalFunc {
	if needsApproval == nil {
		return nil
	}

	return func(ctx context.Context, rawInput json.RawMessage, options ToolNeedsApprovalOptions) (bool, error) {
		input, err := decodeToolValue[Input](rawInput)
		if err != nil {
			return false, err
		}

		return needsApproval(ctx, input, options)
	}
}

func typedExecute[Input, Output any](
	execute func(ctx context.Context, input Input, options ToolExecutionOptions) (Output, error),
) ToolExecuteFunc {
	if execute == nil {
		return nil
	}

	return func(ctx context.Context, rawInput json.RawMessage, options ToolExecutionOptions) (any, error) {
		input, err := decodeToolValue[Input](rawInput)
		if err != nil {
			return nil, err
		}

		return execute(ctx, input, options)
	}
}

func typedModelOutput[Input, Output any](
	toModelOutput func(ctx context.Context, event ToolOutputEvent[Input, Output]) (model.ToolResultOutput, error),
) ToolModelOutputFunc {
	if toModelOutput == nil {
		return nil
	}

	return func(ctx context.Context, event ToolOutputEvent[json.RawMessage, any]) (model.ToolResultOutput, error) {
		input, err := decodeToolValue[Input](event.Input)
		if err != nil {
			return nil, err
		}

		output, err := convertToolOutput[Output](event.Output)
		if err != nil {
			return nil, err
		}

		return toModelOutput(ctx, ToolOutputEvent[Input, Output]{
			ToolCallID: event.ToolCallID,
			Input:      input,
			Output:     output,
		})
	}
}

func typedInputAvailable[Input any](
	onInputAvailable func(ctx context.Context, event ToolInputEvent[Input]),
) ToolInputAvailableFunc {
	if onInputAvailable == nil {
		return nil
	}

	return func(ctx context.Context, event ToolInputEvent[json.RawMessage]) {
		// Execute reports the same decoding failure as a tool error.
		input, err := decodeToolValue[Input](event.Input)
		if err != nil {
			return
		}

		onInputAvailable(ctx, ToolInputEvent[Input]{
			ToolExecutionOptions: event.ToolExecutionOptions,
			Input:                input,
		})
	}
}

// decodeToolValue decodes a tool input. Empty input, which some models send
// for tools without parameters, decodes to the zero value.
func decodeToolValue[Value any](raw json.RawMessage) (Value, error) {
	var value Value
	if len(bytes.TrimSpace(raw)) == 0 {
		return value, nil
	}

	if err := json.Unmarshal(raw, &value); err != nil {
		return value, fmt.Errorf("decode tool input: %w", err)
	}

	return value, nil
}

// convertToolOutput returns output as an Output. Outputs produced by Execute
// already have that type; outputs restored from serialized history are
// converted through JSON.
func convertToolOutput[Output any](output any) (Output, error) {
	if typed, ok := output.(Output); ok {
		return typed, nil
	}

	var converted Output

	encoded, err := json.Marshal(output)
	if err != nil {
		return converted, fmt.Errorf("encode tool output: %w", err)
	}

	if err := json.Unmarshal(encoded, &converted); err != nil {
		return converted, fmt.Errorf("decode tool output: %w", err)
	}

	return converted, nil
}

func encodeInputExamples[Input any](examples []Input) ([]json.RawMessage, error) {
	if len(examples) == 0 {
		return nil, nil
	}

	encoded := make([]json.RawMessage, len(examples))
	for index, example := range examples {
		value, err := json.Marshal(example)
		if err != nil {
			return nil, fmt.Errorf("example %d: %w", index, err)
		}

		encoded[index] = value
	}

	return encoded, nil
}
