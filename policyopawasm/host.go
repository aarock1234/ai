package policyopawasm

import (
	"context"
	"fmt"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

// hostFunction is a function that OPA policies import from env.
type hostFunction struct {
	name           string
	parameters     uint32
	returnsAddress bool
	call           api.GoModuleFunc
}

// hostFunctions are the env functions of the OPA WebAssembly ABI. The
// built-in functions receive the built-in identifier, a reserved context
// address and their argument addresses.
var hostFunctions = []hostFunction{
	{
		name:       "opa_abort",
		parameters: 1,
		call:       abort,
	},
	{
		name:       "opa_println",
		parameters: 1,
		call:       printMessage,
	},
	{
		name:           "opa_builtin0",
		parameters:     2,
		returnsAddress: true,
		call:           callBuiltin,
	},
	{
		name:           "opa_builtin1",
		parameters:     3,
		returnsAddress: true,
		call:           callBuiltin,
	},
	{
		name:           "opa_builtin2",
		parameters:     4,
		returnsAddress: true,
		call:           callBuiltin,
	},
	{
		name:           "opa_builtin3",
		parameters:     5,
		returnsAddress: true,
		call:           callBuiltin,
	},
	{
		name:           "opa_builtin4",
		parameters:     6,
		returnsAddress: true,
		call:           callBuiltin,
	},
}

// instanceKey is the context key of the policy instance that a host
// function serves.
type instanceKey struct{}

// withInstance returns a context whose host function calls are served by
// instance.
func withInstance(ctx context.Context, instance *policyInstance) context.Context {
	return context.WithValue(ctx, instanceKey{}, instance)
}

// instanceFromContext returns the policy instance a host function serves.
func instanceFromContext(ctx context.Context) *policyInstance {
	return ctx.Value(instanceKey{}).(*policyInstance)
}

// instantiateHostModule instantiates the opa_host module that every env
// module imports its functions from.
func instantiateHostModule(ctx context.Context, runtime wazero.Runtime) error {
	builder := runtime.NewHostModuleBuilder(hostModuleName)

	for _, function := range hostFunctions {
		builder.NewFunctionBuilder().
			WithGoModuleFunction(function.call, function.parameterTypes(), function.resultTypes()).
			Export(function.name)
	}

	if _, err := builder.Instantiate(ctx); err != nil {
		return fmt.Errorf("instantiate opa host module: %w", err)
	}

	return nil
}

// parameterTypes returns the WebAssembly parameter types of the function.
func (f hostFunction) parameterTypes() []api.ValueType {
	types := make([]api.ValueType, f.parameters)
	for index := range types {
		types[index] = api.ValueTypeI32
	}

	return types
}

// resultTypes returns the WebAssembly result types of the function.
func (f hostFunction) resultTypes() []api.ValueType {
	if !f.returnsAddress {
		return nil
	}

	return []api.ValueType{
		api.ValueTypeI32,
	}
}

// abort stops the evaluation with the message at the given address.
func abort(ctx context.Context, _ api.Module, stack []uint64) {
	instance := instanceFromContext(ctx)

	message, err := instance.readString(api.DecodeU32(stack[0]))
	if err != nil {
		instance.trap(fmt.Errorf("%w: read message: %w", ErrAborted, err))

		return
	}

	instance.trap(fmt.Errorf("%w: %s", ErrAborted, message))
}

// printMessage passes the message at the given address to Options.OnPrint.
func printMessage(ctx context.Context, _ api.Module, stack []uint64) {
	instance := instanceFromContext(ctx)

	message, err := instance.readString(api.DecodeU32(stack[0]))
	if err != nil {
		instance.trap(fmt.Errorf("read opa print message: %w", err))

		return
	}

	instance.print(ctx, message)
}

// callBuiltin calls the built-in function with the identifier in the first
// stack slot, with the argument addresses after the reserved context slot,
// and returns the address of its result.
func callBuiltin(ctx context.Context, _ api.Module, stack []uint64) {
	instance := instanceFromContext(ctx)

	id := api.DecodeU32(stack[0])
	argumentAddresses := make([]uint32, 0, len(stack)-2)
	for _, slot := range stack[2:] {
		argumentAddresses = append(argumentAddresses, api.DecodeU32(slot))
	}

	resultAddress, err := instance.callBuiltin(ctx, id, argumentAddresses)
	if err != nil {
		instance.trap(err)

		return
	}

	stack[0] = api.EncodeU32(resultAddress)
}
