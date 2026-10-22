package policyopawasm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/experimental"
)

// supportedABIVersion is the major version of the OPA WebAssembly ABI that
// this package implements.
const supportedABIVersion = 1

// fastEvaluationMinorVersion is the first ABI minor version with the
// single-call opa_eval export.
const fastEvaluationMinorVersion = 2

// wasmPageSize is the size of a WebAssembly memory page in bytes.
const wasmPageSize = 64 * 1024

// emptyDataDocument is the data document of a policy without data.
var emptyDataDocument = json.RawMessage(`{}`)

// policyInstance is one instantiated policy with its own memory. It is not
// safe for concurrent use; Client hands each instance to one evaluation at
// a time.
type policyInstance struct {
	settings *instanceSettings

	env    api.Module
	module api.Module
	memory api.Memory

	functions map[string]api.Function

	abiMinorVersion uint64
	builtinNames    map[uint32]string
	entrypoints     map[string]uint32

	dataAddress     uint32
	baseHeapPointer uint32
	dataHeapPointer uint32

	// hostErr is the error that made a host function trap during the
	// current call, reported instead of the wrapped wazero error.
	hostErr error
}

// instanceSettings are the options that every instance of a Client shares.
type instanceSettings struct {
	data     json.RawMessage
	builtins map[string]BuiltinFunc
	onPrint  func(ctx context.Context, message string)
}

// newPolicyInstance instantiates the policy with a new env module, checks
// its ABI version and loads the data document.
func newPolicyInstance(ctx context.Context, runtime wazero.Runtime, modules compiledModules, settings *instanceSettings) (*policyInstance, error) {
	instance := &policyInstance{
		settings:  settings,
		functions: make(map[string]api.Function),
	}

	ctx = withInstance(ctx, instance)

	if err := instance.instantiate(ctx, runtime, modules); err != nil {
		return nil, err
	}

	if err := instance.prepare(ctx); err != nil {
		instance.close(ctx)

		return nil, err
	}

	return instance, nil
}

// instantiate instantiates an anonymous env module and the policy module
// that imports from it.
func (i *policyInstance) instantiate(ctx context.Context, runtime wazero.Runtime, modules compiledModules) error {
	anonymous := wazero.NewModuleConfig().WithName("")

	env, err := runtime.InstantiateModule(ctx, modules.env, anonymous)
	if err != nil {
		return fmt.Errorf("instantiate opa env module: %w", err)
	}

	resolveEnv := func(name string) api.Module {
		if name == envModuleName {
			return env
		}

		return nil
	}

	module, err := runtime.InstantiateModule(experimental.WithImportResolver(ctx, resolveEnv), modules.policy, anonymous)
	if err != nil {
		closeModule(ctx, env)

		return fmt.Errorf("instantiate opa policy module: %w", err)
	}

	i.env = env
	i.module = module
	i.memory = env.ExportedMemory(envMemoryExport)

	return nil
}

// prepare checks the ABI version, reads the built-in functions and
// entrypoints of the policy, and loads its data document, as
// @open-policy-agent/opa-wasm does when it loads a policy.
func (i *policyInstance) prepare(ctx context.Context) error {
	if err := i.readABIVersion(); err != nil {
		return err
	}

	if err := i.readBuiltins(ctx); err != nil {
		return err
	}

	dataAddress, err := i.loadJSON(ctx, emptyDataDocument)
	if err != nil {
		return fmt.Errorf("load empty opa data document: %w", err)
	}

	baseHeapPointer, err := i.call(ctx, "opa_heap_ptr_get")
	if err != nil {
		return err
	}

	i.dataAddress = dataAddress
	i.baseHeapPointer = baseHeapPointer
	i.dataHeapPointer = baseHeapPointer

	if err := i.readEntrypoints(ctx); err != nil {
		return err
	}

	if i.settings.data == nil {
		return nil
	}

	return i.setData(ctx, i.settings.data)
}

// readABIVersion rejects policies built for another major ABI version and
// records the minor version. Policies without version globals are treated
// as ABI 1.0.
func (i *policyInstance) readABIVersion() error {
	if version := i.module.ExportedGlobal("opa_wasm_abi_version"); version != nil {
		if version.Get() != supportedABIVersion {
			return fmt.Errorf("%w: %d", ErrUnsupportedABI, version.Get())
		}
	}

	if minorVersion := i.module.ExportedGlobal("opa_wasm_abi_minor_version"); minorVersion != nil {
		i.abiMinorVersion = minorVersion.Get()
	}

	return nil
}

// readBuiltins reads the identifiers of the built-in functions the policy
// calls on the host.
func (i *policyInstance) readBuiltins(ctx context.Context) error {
	builtinIDs := make(map[string]uint32)
	if err := i.readExportedDocument(ctx, "builtins", &builtinIDs); err != nil {
		return err
	}

	i.builtinNames = make(map[uint32]string, len(builtinIDs))
	for name, id := range builtinIDs {
		i.builtinNames[id] = name
	}

	return nil
}

// readEntrypoints reads the identifiers of the policy entrypoints.
func (i *policyInstance) readEntrypoints(ctx context.Context) error {
	i.entrypoints = make(map[string]uint32)

	return i.readExportedDocument(ctx, "entrypoints", &i.entrypoints)
}

// readExportedDocument decodes the JSON document that the named export
// returns the address of.
func (i *policyInstance) readExportedDocument(ctx context.Context, export string, target any) error {
	address, err := i.call(ctx, export)
	if err != nil {
		return err
	}

	document, err := i.dumpJSON(ctx, address)
	if err != nil {
		return fmt.Errorf("read opa %s: %w", export, err)
	}

	if err := json.Unmarshal(document, target); err != nil {
		return fmt.Errorf("decode opa %s: %w", export, err)
	}

	return nil
}

// setData replaces the data document that evaluations read.
func (i *policyInstance) setData(ctx context.Context, data json.RawMessage) error {
	if _, err := i.call(ctx, "opa_heap_ptr_set", i.baseHeapPointer); err != nil {
		return err
	}

	dataAddress, err := i.loadJSON(ctx, data)
	if err != nil {
		return fmt.Errorf("load opa data document: %w", err)
	}

	dataHeapPointer, err := i.call(ctx, "opa_heap_ptr_get")
	if err != nil {
		return err
	}

	i.dataAddress = dataAddress
	i.dataHeapPointer = dataHeapPointer

	return nil
}

// evaluate evaluates the entrypoint with the JSON input and returns the
// JSON result set. A nil input leaves the input document undefined.
func (i *policyInstance) evaluate(ctx context.Context, entrypoint uint32, input json.RawMessage) (json.RawMessage, error) {
	i.hostErr = nil
	ctx = withInstance(ctx, i)

	if i.abiMinorVersion >= fastEvaluationMinorVersion {
		return i.evaluateFast(ctx, entrypoint, input)
	}

	return i.evaluateLegacy(ctx, entrypoint, input)
}

// evaluateFast evaluates with the opa_eval export of ABI 1.2: the input is
// written to the heap after the data document and the policy parses it.
func (i *policyInstance) evaluateFast(ctx context.Context, entrypoint uint32, input json.RawMessage) (json.RawMessage, error) {
	var inputAddress, inputLength uint32

	if input != nil {
		if uint64(len(input)) > math.MaxUint32-uint64(i.dataHeapPointer) {
			return nil, ErrMemoryLimit
		}

		inputAddress = i.dataHeapPointer
		inputLength = uint32(len(input))

		if err := i.ensureMemory(uint64(inputAddress) + uint64(inputLength)); err != nil {
			return nil, err
		}

		if !i.memory.Write(inputAddress, input) {
			return nil, fmt.Errorf("write opa input: %w", ErrMemoryLimit)
		}
	}

	heapPointer := i.dataHeapPointer + inputLength

	resultAddress, err := i.call(ctx, "opa_eval", 0, entrypoint, i.dataAddress, inputAddress, inputLength, heapPointer, 0)
	if err != nil {
		return nil, err
	}

	return i.readCString(resultAddress)
}

// evaluateLegacy evaluates with an evaluation context, for policies built
// for ABI versions before 1.2.
func (i *policyInstance) evaluateLegacy(ctx context.Context, entrypoint uint32, input json.RawMessage) (json.RawMessage, error) {
	if _, err := i.call(ctx, "opa_heap_ptr_set", i.dataHeapPointer); err != nil {
		return nil, err
	}

	inputAddress, err := i.loadJSON(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("load opa input: %w", err)
	}

	contextAddress, err := i.call(ctx, "opa_eval_ctx_new")
	if err != nil {
		return nil, err
	}

	if _, err := i.call(ctx, "opa_eval_ctx_set_input", contextAddress, inputAddress); err != nil {
		return nil, err
	}

	if _, err := i.call(ctx, "opa_eval_ctx_set_data", contextAddress, i.dataAddress); err != nil {
		return nil, err
	}

	if _, err := i.call(ctx, "opa_eval_ctx_set_entrypoint", contextAddress, entrypoint); err != nil {
		return nil, err
	}

	if _, err := i.call(ctx, "eval", contextAddress); err != nil {
		return nil, err
	}

	resultAddress, err := i.call(ctx, "opa_eval_ctx_get_result", contextAddress)
	if err != nil {
		return nil, err
	}

	return i.dumpJSON(ctx, resultAddress)
}

// callBuiltin calls the built-in function with the given identifier on the
// JSON values at the argument addresses, and returns the address of the
// parsed result, or zero when the result is undefined.
func (i *policyInstance) callBuiltin(ctx context.Context, id uint32, argumentAddresses []uint32) (uint32, error) {
	name := i.builtinNames[id]

	builtin := i.settings.builtin(name)
	if builtin == nil {
		return 0, fmt.Errorf("%w: %d: %s", ErrBuiltinNotImplemented, id, name)
	}

	arguments := make([]json.RawMessage, 0, len(argumentAddresses))
	for _, address := range argumentAddresses {
		argument, err := i.dumpJSON(ctx, address)
		if err != nil {
			return 0, fmt.Errorf("read argument of built-in function %s: %w", name, err)
		}

		arguments = append(arguments, argument)
	}

	result, err := builtin(ctx, arguments)
	if err != nil {
		return 0, fmt.Errorf("call built-in function %s: %w", name, err)
	}

	resultAddress, err := i.loadJSON(ctx, result)
	if err != nil {
		return 0, fmt.Errorf("load result of built-in function %s: %w", name, err)
	}

	return resultAddress, nil
}

// builtin returns the implementation of the named built-in function: a
// custom one when Options.Builtins has it, otherwise a default one.
func (s *instanceSettings) builtin(name string) BuiltinFunc {
	if builtin, ok := s.builtins[name]; ok {
		return builtin
	}

	return defaultBuiltins[name]
}

// print passes a policy print message to Options.OnPrint.
func (i *policyInstance) print(ctx context.Context, message string) {
	if i.settings.onPrint == nil {
		return
	}

	i.settings.onPrint(ctx, message)
}

// trap records err as the cause of the current call and stops the
// WebAssembly execution. wazero turns a host function panic into an error
// returned by the exported function call.
func (i *policyInstance) trap(err error) {
	i.hostErr = err

	panic(err)
}

// loadJSON copies the JSON value into the heap and returns the address of
// the value the policy parsed from it. A nil value returns zero, which the
// ABI reads as undefined.
func (i *policyInstance) loadJSON(ctx context.Context, value json.RawMessage) (uint32, error) {
	if value == nil {
		return 0, nil
	}

	if uint64(len(value)) > math.MaxUint32 {
		return 0, ErrMemoryLimit
	}

	length := uint32(len(value))

	rawAddress, err := i.call(ctx, "opa_malloc", length)
	if err != nil {
		return 0, err
	}

	if !i.memory.Write(rawAddress, value) {
		return 0, fmt.Errorf("write json value: %w", ErrMemoryLimit)
	}

	parsedAddress, err := i.call(ctx, "opa_json_parse", rawAddress, length)
	if err != nil {
		return 0, err
	}

	if parsedAddress == 0 {
		return 0, errors.New("failed to parse json value")
	}

	return parsedAddress, nil
}

// dumpJSON serializes the value at address and returns its JSON text.
func (i *policyInstance) dumpJSON(ctx context.Context, address uint32) (json.RawMessage, error) {
	rawAddress, err := i.call(ctx, "opa_json_dump", address)
	if err != nil {
		return nil, err
	}

	return i.readCString(rawAddress)
}

// readString reads the null-terminated string at address.
func (i *policyInstance) readString(address uint32) (string, error) {
	text, err := i.readCString(address)
	if err != nil {
		return "", err
	}

	return string(text), nil
}

// readCString copies the bytes from address up to the next null byte.
func (i *policyInstance) readCString(address uint32) ([]byte, error) {
	size := i.memory.Size()
	if address >= size {
		return nil, fmt.Errorf("read string at %d: %w", address, ErrMemoryLimit)
	}

	view, ok := i.memory.Read(address, size-address)
	if !ok {
		return nil, fmt.Errorf("read string at %d: %w", address, ErrMemoryLimit)
	}

	end := bytes.IndexByte(view, 0)
	if end < 0 {
		return nil, fmt.Errorf("read string at %d: missing null terminator", address)
	}

	return bytes.Clone(view[:end]), nil
}

// ensureMemory grows the memory so that it holds at least size bytes.
func (i *policyInstance) ensureMemory(size uint64) error {
	current := uint64(i.memory.Size())
	if size <= current {
		return nil
	}

	pages := (size - current + wasmPageSize - 1) / wasmPageSize
	if pages > math.MaxUint32 {
		return ErrMemoryLimit
	}

	if _, ok := i.memory.Grow(uint32(pages)); !ok {
		return ErrMemoryLimit
	}

	return nil
}

// call calls the named export of the policy and returns its first result.
// A host function failure is reported through its own error, and a call
// stopped by the context through the context error.
func (i *policyInstance) call(ctx context.Context, export string, parameters ...uint32) (uint32, error) {
	function, err := i.exportedFunction(export)
	if err != nil {
		return 0, err
	}

	encoded := make([]uint64, 0, len(parameters))
	for _, parameter := range parameters {
		encoded = append(encoded, api.EncodeU32(parameter))
	}

	results, err := function.Call(ctx, encoded...)
	if err != nil {
		return 0, i.callError(ctx, export, err)
	}

	if len(results) == 0 {
		return 0, nil
	}

	return api.DecodeU32(results[0]), nil
}

// exportedFunction returns the named export of the policy, caching it for
// later calls.
func (i *policyInstance) exportedFunction(export string) (api.Function, error) {
	if function, ok := i.functions[export]; ok {
		return function, nil
	}

	function := i.module.ExportedFunction(export)
	if function == nil {
		return nil, fmt.Errorf("%w: %s", ErrMissingExport, export)
	}

	i.functions[export] = function

	return function, nil
}

// callError returns the most specific cause of a failed export call.
func (i *policyInstance) callError(ctx context.Context, export string, err error) error {
	if i.hostErr != nil {
		return i.hostErr
	}

	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}

	return fmt.Errorf("call opa export %s: %w", export, err)
}

// close closes the policy and env modules.
func (i *policyInstance) close(ctx context.Context) {
	ctx = context.WithoutCancel(ctx)

	closeModule(ctx, i.module)
	closeModule(ctx, i.env)
}

// closeModule closes a module and releases its memory. Closing only fails
// for modules that are already closed, so the error is not reported.
func closeModule(ctx context.Context, module api.Module) {
	_ = module.Close(ctx)
}
