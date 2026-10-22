// Package policyopawasm evaluates Open Policy Agent policies compiled to
// WebAssembly in process, a port of the wasmPolicyClient of the Vercel AI
// SDK @ai-sdk/policy-opa package and of the @open-policy-agent/opa-wasm
// runtime it uses.
//
// Build a bundle with "opa build -t wasm -e agent/call/decision", extract
// its policy.wasm and pass a Client wherever package policyopa accepts a
// policyopa.PolicyClient:
//
//	client, err := policyopawasm.New(ctx, policyopawasm.Options{
//		Wasm: policyWasm,
//	})
//	if err != nil {
//		return err
//	}
//	defer client.Close(ctx)
//
//	toolApproval := policyopa.OPAPolicy(policyopa.OPAPolicyOptions{
//		Client: client,
//		Path:   "agent/call/decision",
//	})
//
// The policy runs on the pure Go wazero runtime. It is a nested module so
// that the core module does not depend on wazero.
//
// Go adaptations: canceling the context stops a running evaluation, and
// Close releases the runtime. Each concurrent evaluation uses its own
// policy instance; idle instances are reused. Options.Entrypoint selects a
// named entrypoint, where upstream always evaluates the default one. The
// sprintf and regex.split built-in functions follow the Go semantics of
// OPA itself instead of JavaScript's. The YAML built-in functions of
// @open-policy-agent/opa-wasm are not provided, because they need a YAML
// library; supply them through Options.Builtins when a policy uses them.
// Policy print output goes to Options.OnPrint instead of the console.
package policyopawasm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/tetratelabs/wazero"

	"github.com/aarock1234/ai/pkg/policyopa"
)

// defaultInitialMemoryPages is the initial memory of a policy instance,
// 320 KiB as in @open-policy-agent/opa-wasm.
const defaultInitialMemoryPages = 5

var (
	// ErrNoResult is returned when the policy produces no result, usually
	// because the bundle was built with another entrypoint.
	ErrNoResult = errors.New("opa wasm policy produced no result; check that the bundle was built with the correct entrypoint (opa build -t wasm -e <path>)")

	// ErrClosed is returned by evaluations after Close.
	ErrClosed = errors.New("opa wasm policy client closed")

	// ErrAborted is returned when the policy aborts, for example on a
	// conflicting rule result.
	ErrAborted = errors.New("opa wasm policy aborted")

	// ErrBuiltinNotImplemented is returned when the policy calls a
	// built-in function that neither the defaults nor Options.Builtins
	// implement.
	ErrBuiltinNotImplemented = errors.New("not implemented: built-in function")

	// ErrUnsupportedABI is returned for policies built for another major
	// version of the OPA WebAssembly ABI.
	ErrUnsupportedABI = errors.New("unsupported opa wasm abi version")

	// ErrMissingExport is returned when the module lacks a function of the
	// OPA WebAssembly ABI.
	ErrMissingExport = errors.New("opa wasm module lacks export")

	// ErrMemoryLimit is returned when the policy memory cannot hold a value.
	ErrMemoryLimit = errors.New("opa wasm memory limit exceeded")

	// ErrUnknownEntrypoint is returned by New when Options.Entrypoint is not
	// an entrypoint of the policy.
	ErrUnknownEntrypoint = errors.New("unknown opa wasm entrypoint")
)

// Options configure a Client.
type Options struct {
	// Wasm is the policy.wasm module of a bundle built with
	// "opa build -t wasm".
	Wasm []byte

	// Data is the data document that the policy reads. Nil uses an empty
	// document.
	Data any

	// Entrypoint is the name of the entrypoint to evaluate, such as
	// "agent/call/decision". Empty evaluates the first entrypoint.
	Entrypoint string

	// Builtins implement built-in functions that the policy calls on the
	// host, by name. They take precedence over the defaults (json.is_valid,
	// regex.split and sprintf).
	Builtins map[string]BuiltinFunc

	// InitialMemoryPages is the initial size of the policy memory in 64 KiB
	// pages. Zero uses 5 pages, or more when the module requires it.
	InitialMemoryPages uint32

	// MaxMemoryPages limits the policy memory in 64 KiB pages. Zero leaves
	// it limited by the module and the 4 GiB WebAssembly maximum.
	MaxMemoryPages uint32

	// OnPrint receives the output of print calls in the policy. Nil discards
	// it.
	OnPrint func(ctx context.Context, message string)
}

// Client evaluates an OPA policy compiled to WebAssembly. It is safe for
// concurrent use.
type Client struct {
	runtime    wazero.Runtime
	modules    compiledModules
	settings   *instanceSettings
	entrypoint uint32

	mu     sync.Mutex
	idle   []*policyInstance
	closed bool
}

// compiledModules are the compiled policy module and the env module that
// each policy instance imports its memory and host functions from.
type compiledModules struct {
	policy wazero.CompiledModule
	env    wazero.CompiledModule
}

var _ policyopa.PolicyClient = (*Client)(nil)

// policyResult is one entry of the result set of an evaluation.
type policyResult struct {
	Result json.RawMessage `json:"result"`
}

// New compiles the policy and instantiates it once, so that an invalid
// module, ABI version, data document or entrypoint is reported here. Call
// Close to release the runtime.
func New(ctx context.Context, options Options) (*Client, error) {
	if len(options.Wasm) == 0 {
		return nil, errors.New("opa wasm module is required")
	}

	settings, err := newInstanceSettings(options)
	if err != nil {
		return nil, err
	}

	config := wazero.NewRuntimeConfig().WithCloseOnContextDone(true)

	client := &Client{
		runtime:  wazero.NewRuntimeWithConfig(ctx, config),
		settings: settings,
	}

	if err := client.start(ctx, options); err != nil {
		_ = client.runtime.Close(context.WithoutCancel(ctx))

		return nil, err
	}

	return client, nil
}

// newInstanceSettings encodes the data document and collects the options
// that every policy instance shares.
func newInstanceSettings(options Options) (*instanceSettings, error) {
	settings := &instanceSettings{
		builtins: options.Builtins,
		onPrint:  options.OnPrint,
	}

	if options.Data == nil {
		return settings, nil
	}

	data, err := json.Marshal(options.Data)
	if err != nil {
		return nil, fmt.Errorf("encode opa data document: %w", err)
	}

	settings.data = data

	return settings, nil
}

// start compiles the modules, instantiates the first policy instance and
// resolves the entrypoint.
func (c *Client) start(ctx context.Context, options Options) error {
	if err := instantiateHostModule(ctx, c.runtime); err != nil {
		return err
	}

	modules, err := compileModules(ctx, c.runtime, options)
	if err != nil {
		return err
	}

	c.modules = modules

	instance, err := newPolicyInstance(ctx, c.runtime, c.modules, c.settings)
	if err != nil {
		return err
	}

	entrypoint, err := resolveEntrypoint(instance, options.Entrypoint)
	if err != nil {
		instance.close(ctx)

		return err
	}

	c.entrypoint = entrypoint
	c.idle = append(c.idle, instance)

	return nil
}

// compileModules compiles the policy module and an env module whose memory
// satisfies the memory the policy imports.
func compileModules(ctx context.Context, runtime wazero.Runtime, options Options) (compiledModules, error) {
	policy, err := runtime.CompileModule(ctx, options.Wasm)
	if err != nil {
		return compiledModules{}, fmt.Errorf("compile opa wasm module: %w", err)
	}

	limits, err := envMemoryLimits(policy, options)
	if err != nil {
		return compiledModules{}, err
	}

	env, err := runtime.CompileModule(ctx, encodeEnvModule(limits))
	if err != nil {
		return compiledModules{}, fmt.Errorf("compile opa env module: %w", err)
	}

	return compiledModules{
		policy: policy,
		env:    env,
	}, nil
}

// envMemoryLimits returns the memory limits from the options, raised to the
// minimum and lowered to the maximum of the memory the policy imports.
func envMemoryLimits(policy wazero.CompiledModule, options Options) (memoryLimits, error) {
	limits := memoryLimits{
		initialPages: options.InitialMemoryPages,
		maxPages:     options.MaxMemoryPages,
		hasMax:       options.MaxMemoryPages > 0,
	}

	if limits.initialPages == 0 {
		limits.initialPages = defaultInitialMemoryPages
	}

	for _, memory := range policy.ImportedMemories() {
		moduleName, _, _ := memory.Import()
		if moduleName != envModuleName {
			continue
		}

		limits.initialPages = max(limits.initialPages, memory.Min())

		importedMax, hasImportedMax := memory.Max()
		if hasImportedMax && (!limits.hasMax || importedMax < limits.maxPages) {
			limits.maxPages = importedMax
			limits.hasMax = true
		}
	}

	if limits.hasMax && limits.initialPages > limits.maxPages {
		return memoryLimits{}, fmt.Errorf("%w: initial memory of %d pages exceeds the maximum of %d pages", ErrMemoryLimit, limits.initialPages, limits.maxPages)
	}

	return limits, nil
}

// resolveEntrypoint returns the identifier of the named entrypoint, or the
// first entrypoint when name is empty.
func resolveEntrypoint(instance *policyInstance, name string) (uint32, error) {
	if name == "" {
		return 0, nil
	}

	entrypoint, ok := instance.entrypoints[name]
	if !ok {
		return 0, fmt.Errorf("%w: %s", ErrUnknownEntrypoint, name)
	}

	return entrypoint, nil
}

// Evaluate evaluates the policy with input and returns the result of the
// first entry of the result set. The path is informational: the bundle is
// built around its entrypoints at "opa build" time, and Options.Entrypoint
// selects one. A policy without a result returns ErrNoResult, so that
// callers fail closed.
func (c *Client) Evaluate(ctx context.Context, path string, input any) (json.RawMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	encodedInput, err := encodeInput(input)
	if err != nil {
		return nil, fmt.Errorf("encode opa policy %q input: %w", path, err)
	}

	instance, err := c.acquire(ctx)
	if err != nil {
		return nil, err
	}

	resultSet, err := instance.evaluate(ctx, c.entrypoint, encodedInput)
	if err != nil {
		// A failed instance may be closed by its context or left in an
		// inconsistent state, so it is not reused.
		instance.close(ctx)

		return nil, fmt.Errorf("evaluate opa wasm policy %q: %w", path, err)
	}

	c.release(ctx, instance)

	return firstResult(resultSet)
}

// encodeInput encodes the input document, or returns nil when input is
// null so that the input stays undefined.
func encodeInput(input any) (json.RawMessage, error) {
	encoded, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}

	if string(encoded) == "null" {
		return nil, nil
	}

	return encoded, nil
}

// firstResult returns the result of the first entry of a result set.
func firstResult(resultSet json.RawMessage) (json.RawMessage, error) {
	var results []policyResult
	if err := json.Unmarshal(resultSet, &results); err != nil {
		return nil, fmt.Errorf("decode opa wasm result set: %w", err)
	}

	if len(results) == 0 {
		return nil, ErrNoResult
	}

	return results[0].Result, nil
}

// acquire returns an idle policy instance, or instantiates a new one when
// every instance is in use.
func (c *Client) acquire(ctx context.Context) (*policyInstance, error) {
	c.mu.Lock()

	if c.closed {
		c.mu.Unlock()

		return nil, ErrClosed
	}

	if count := len(c.idle); count > 0 {
		instance := c.idle[count-1]
		c.idle = c.idle[:count-1]
		c.mu.Unlock()

		return instance, nil
	}

	c.mu.Unlock()

	return newPolicyInstance(ctx, c.runtime, c.modules, c.settings)
}

// release returns an instance to the idle instances, or closes it when the
// client is closed.
func (c *Client) release(ctx context.Context, instance *policyInstance) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		instance.close(ctx)

		return
	}

	c.idle = append(c.idle, instance)
}

// Close releases the runtime and every policy instance. Evaluations that
// are still running fail.
func (c *Client) Close(ctx context.Context) error {
	c.mu.Lock()

	if c.closed {
		c.mu.Unlock()

		return nil
	}

	c.closed = true
	c.idle = nil
	c.mu.Unlock()

	if err := c.runtime.Close(ctx); err != nil {
		return fmt.Errorf("close opa wasm runtime: %w", err)
	}

	return nil
}
