package codemode

import (
	"cmp"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/dop251/goja"
)

var (
	//go:embed sandbox_bootstrap.js
	bootstrapSource string

	//go:embed sandbox_serde.js
	serdeSource string
)

const (
	bootstrapName = "code-mode-bootstrap.js"
	serdeName     = "code-mode-serde.js"
)

// sandboxInput is everything one execution of a program needs.
type sandboxInput struct {
	js     string
	host   *hostTools
	replay *replay
	limits executionLimits

	onConsole func(ctx context.Context, entry ConsoleEntry)
}

// sandboxOutcome is the result of an execution that completed or stopped at
// interruptions.
type sandboxOutcome struct {
	// output is the JSON encoding of the returned value; nil means
	// undefined.
	output json.RawMessage

	// interruptions are ordered by request; ledger records every host
	// tool call of the execution.
	interruptions []interruptedCall
	ledger        []ledgerEntry
	console       consoleRecorder
}

// interruptedCall is a host tool call that requested an interrupt.
type interruptedCall struct {
	request        int
	interruptionID string
	toolName       string
	input          json.RawMessage
	payload        InterruptPayload
}

// sandboxHelpers are JavaScript functions captured before the program
// runs, so that the program cannot replace them.
type sandboxHelpers struct {
	stringify     goja.Callable
	parse         goja.Callable
	createError   goja.Callable
	describeError goja.Callable
	detachedCall  goja.Callable
}

// sandbox runs one program in a fresh JavaScript runtime. The runtime is
// only used on the goroutine that calls run; host tool calls run on worker
// goroutines and report back through completions.
type sandbox struct {
	ctx     context.Context
	vm      *goja.Runtime
	helpers sandboxHelpers
	input   sandboxInput
	console consoleRecorder
	clock   runClock

	requests     map[int]*bridgeRequest
	requestCount int
	inFlight     int
	completions  chan bridgeCompletion
	workers      sync.WaitGroup

	// ledger starts as the resumed execution's ledger and records every
	// call as it settles or interrupts.
	ledger           []ledgerEntry
	recordedRequests int
	lastSettledOrder int

	// responses holds settled outcomes by settled order until every
	// earlier outcome has been delivered.
	responses    map[int]bridgeResponse
	nextResponse int

	interruptions []interruptedCall

	// hostErrors maps the JavaScript errors given to the program to the Go
	// errors they report, so that an uncaught one is returned unchanged.
	hostErrors map[*goja.Object]error

	// fatal is a failure that ends the execution, such as an exceeded
	// bridge limit.
	fatal error
}

// runSandbox executes a program within the invocation's timeout. It waits
// for every host tool call it started before returning.
func runSandbox(ctx context.Context, input sandboxInput) (sandboxOutcome, error) {
	timeout := input.limits.timeout
	ctx, cancel := context.WithTimeoutCause(ctx, timeout, newTimeoutError(timeout))

	recorded := input.replay.state.Ledger

	s := &sandbox{
		ctx:   ctx,
		vm:    goja.New(),
		input: input,
		console: consoleRecorder{
			maxBytes:  input.limits.maxConsoleOutputBytes,
			onConsole: input.onConsole,
		},
		clock: runClock{
			nowMs: input.replay.state.Determinism.DateNowMs,
		},
		requests:         make(map[int]*bridgeRequest),
		completions:      make(chan bridgeCompletion, input.limits.maxInFlightBridgeRequests),
		ledger:           slices.Clone(recorded),
		recordedRequests: len(recorded),
		lastSettledOrder: lastSettledOrder(recorded),
		responses:        make(map[int]bridgeResponse),
		nextResponse:     1,
		hostErrors:       make(map[*goja.Object]error),
	}

	stopWatching := s.interruptWhenDone()

	defer func() {
		stopWatching()
		cancel()
		s.workers.Wait()
	}()

	return s.run()
}

func (s *sandbox) run() (sandboxOutcome, error) {
	if err := s.install(); err != nil {
		return sandboxOutcome{}, err
	}

	stripped, err := stripTypes(s.input.js)
	if err != nil {
		return sandboxOutcome{}, err
	}

	program, err := goja.Compile(programName, stripped, false)
	if err != nil {
		return sandboxOutcome{}, syntaxFailure(err)
	}

	value, err := s.vm.RunProgram(program)
	if err != nil {
		return sandboxOutcome{}, s.failure(err)
	}

	promise, ok := value.Export().(*goja.Promise)
	if !ok {
		return sandboxOutcome{}, errors.New("code mode: program wrapper did not return a promise")
	}

	if err := s.drive(promise); err != nil {
		return sandboxOutcome{}, err
	}

	return s.settle(promise)
}

// install configures the runtime and installs the sandbox globals. The
// clock and Math.random follow upstream's runtime, so that a replayed
// program observes the values the original execution observed.
func (s *sandbox) install() error {
	random := newRunRandom(s.input.replay.state.Determinism.RandomSeed)

	s.vm.SetRandSource(random.next)
	s.vm.SetTimeSource(s.clock.now)
	s.vm.SetMaxCallStackSize(s.input.limits.maxCallStackDepth)

	serialize, err := s.vm.RunScript(serdeName, serdeSource)
	if err != nil {
		return fmt.Errorf("code mode: run sandbox serializer: %w", err)
	}

	value, err := s.vm.RunScript(bootstrapName, bootstrapSource)
	if err != nil {
		return fmt.Errorf("code mode: run sandbox bootstrap: %w", err)
	}

	bootstrap, ok := goja.AssertFunction(value)
	if !ok {
		return errors.New("code mode: sandbox bootstrap is not a function")
	}

	bridge, err := s.newBridge()
	if err != nil {
		return err
	}

	helpers, err := bootstrap(goja.Undefined(), bridge, serialize)
	if err != nil {
		return fmt.Errorf("code mode: install sandbox globals: %w", err)
	}

	return s.captureHelpers(helpers.ToObject(s.vm))
}

// newBridge returns the host functions and tool bindings the bootstrap
// installs.
func (s *sandbox) newBridge() (*goja.Object, error) {
	bindings := s.vm.NewObject()

	for index, toolName := range s.input.host.names {
		if err := bindings.Set(toolName, toolBindingName(index)); err != nil {
			return nil, fmt.Errorf("code mode: install tool binding: %w", err)
		}
	}

	members := map[string]any{
		"callTool":       s.callTool,
		"writeConsole":   s.writeConsole,
		"bindings":       bindings,
		"missingBinding": missingToolBinding,
	}

	bridge := s.vm.NewObject()

	for name, member := range members {
		if err := bridge.Set(name, member); err != nil {
			return nil, fmt.Errorf("code mode: install sandbox bridge: %w", err)
		}
	}

	return bridge, nil
}

func (s *sandbox) captureHelpers(helpers *goja.Object) error {
	targets := map[string]*goja.Callable{
		"stringify":     &s.helpers.stringify,
		"parse":         &s.helpers.parse,
		"createError":   &s.helpers.createError,
		"describeError": &s.helpers.describeError,
		"detachedCall":  &s.helpers.detachedCall,
	}

	for name, target := range targets {
		callable, ok := goja.AssertFunction(helpers.Get(name))
		if !ok {
			return fmt.Errorf("code mode: sandbox helper %s is not a function", name)
		}

		*target = callable
	}

	return nil
}

// interruptWhenDone interrupts the runtime when the invocation's context
// ends. The returned function stops watching and waits for the watcher.
func (s *sandbox) interruptWhenDone() func() {
	finished := make(chan struct{})

	var watcher sync.WaitGroup

	watcher.Go(func() {
		select {
		case <-s.ctx.Done():
			s.vm.Interrupt(contextFailure(s.ctx))

		case <-finished:
		}
	})

	return func() {
		close(finished)
		watcher.Wait()
	}
}

// settle turns the state of the program's promise into the outcome. Like
// upstream's runtime, an execution with pending interruptions stops at
// them once no call is running, however the program's promise settled.
func (s *sandbox) settle(promise *goja.Promise) (sandboxOutcome, error) {
	if len(s.interruptions) > 0 && s.inFlight == 0 {
		return s.outcome(nil), nil
	}

	switch promise.State() {
	case goja.PromiseStateFulfilled:
		return s.completed(promise.Result())

	case goja.PromiseStateRejected:
		return sandboxOutcome{}, s.rejection(promise.Result())

	default:
		if len(s.responses) > 0 || s.requestCount < s.recordedRequests {
			return sandboxOutcome{}, newProtocolError("code mode continuation replay diverged from the recorded host tool calls")
		}

		return sandboxOutcome{}, &Error{
			Code:    ErrorCodeGeneric,
			Message: "program did not finish: it awaits a promise that never settles",
		}
	}
}

// completed checks that a program that returned awaited every host tool
// call it made and replayed the whole continuation, and encodes its result.
func (s *sandbox) completed(result goja.Value) (sandboxOutcome, error) {
	if s.inFlight > 0 {
		return sandboxOutcome{}, newDetachedError("program returned while host tool calls it started were still running; await every tool call")
	}

	if err := s.checkDetachedCalls(); err != nil {
		return sandboxOutcome{}, err
	}

	if s.requestCount < s.recordedRequests {
		return sandboxOutcome{}, newProtocolError("code mode continuation completed before replaying every recorded host tool call")
	}

	output, err := s.encodeResult(result)
	if err != nil {
		return sandboxOutcome{}, err
	}

	return s.outcome(output), nil
}

// checkDetachedCalls fails a program that created a host tool call it never
// awaited, or that returned while one was still pending.
func (s *sandbox) checkDetachedCalls() error {
	detached, err := s.helpers.detachedCall(goja.Undefined())
	if err != nil {
		return s.failure(err)
	}

	if goja.IsNull(detached) {
		return nil
	}

	fields := detached.ToObject(s.vm)
	toolName := fields.Get("toolName").String()

	if !fields.Get("awaited").ToBoolean() {
		return newDetachedError(fmt.Sprintf("program created the host tool call %s without awaiting it; await every tool call", toolName))
	}

	return newDetachedError(fmt.Sprintf("program returned while the host tool call %s was still pending; await every tool call", toolName))
}

func newDetachedError(message string) *Error {
	return &Error{
		Code:    ErrorCodeDetachedBridgeRequest,
		Message: message,
	}
}

func (s *sandbox) outcome(output json.RawMessage) sandboxOutcome {
	interruptions := slices.SortedFunc(slices.Values(s.interruptions), func(left, right interruptedCall) int {
		return cmp.Compare(left.request, right.request)
	})

	return sandboxOutcome{
		output:        output,
		interruptions: interruptions,
		ledger:        s.ledger,
		console:       s.console,
	}
}

// encodeResult encodes the returned value with JSON.stringify semantics.
func (s *sandbox) encodeResult(value goja.Value) (json.RawMessage, error) {
	if goja.IsUndefined(value) {
		return nil, nil
	}

	encoded, err := s.helpers.stringify(goja.Undefined(), value)
	if err != nil {
		if exception, ok := errors.AsType[*goja.Exception](err); ok {
			return nil, &Error{
				Code:    ErrorCodeSerialization,
				Message: "code mode result is not JSON-serializable: " + s.describe(exception.Value()).Message,
			}
		}

		return nil, s.failure(err)
	}

	if goja.IsUndefined(encoded) {
		return nil, &Error{
			Code:    ErrorCodeSerialization,
			Message: "code mode result is not JSON-serializable",
		}
	}

	result := encoded.String()
	if err := checkPayloadSize(result, s.input.limits.maxResultBytes, "code mode result"); err != nil {
		return nil, err
	}

	return json.RawMessage(result), nil
}

// failure translates an error returned by the runtime.
func (s *sandbox) failure(err error) error {
	if interrupted, ok := errors.AsType[*goja.InterruptedError](err); ok {
		if cause, ok := interrupted.Value().(error); ok {
			return cause
		}

		return newAbortedError(err)
	}

	if _, ok := errors.AsType[*goja.StackOverflowError](err); ok {
		return &ScriptError{
			Name:    "RangeError",
			Message: "maximum call stack size exceeded",
		}
	}

	if exception, ok := errors.AsType[*goja.Exception](err); ok {
		return s.rejection(exception.Value())
	}

	return err
}

// rejection translates a value the program threw: a host tool error is
// returned unchanged, anything else as a ScriptError.
func (s *sandbox) rejection(reason goja.Value) error {
	if object, ok := reason.(*goja.Object); ok {
		if err, found := s.hostErrors[object]; found {
			return err
		}
	}

	return s.describe(reason)
}

func (s *sandbox) describe(reason goja.Value) *ScriptError {
	description, err := s.helpers.describeError(goja.Undefined(), reason)
	if err != nil {
		return &ScriptError{
			Message: "uncaught exception",
		}
	}

	fields := description.ToObject(s.vm)

	return &ScriptError{
		Name:    fields.Get("name").String(),
		Message: fields.Get("message").String(),
		Stack:   strings.TrimSpace(fields.Get("stack").String()),
	}
}

// contextFailure is the error reported when the invocation's context ends:
// the timeout error for the invocation's own timeout, and an aborted error
// for the caller's cancellation or deadline.
func contextFailure(ctx context.Context) error {
	cause := context.Cause(ctx)

	if codeModeErr, ok := errors.AsType[*Error](cause); ok && codeModeErr.Code == ErrorCodeTimeout {
		return codeModeErr
	}

	return newAbortedError(cause)
}

func (s *sandbox) writeConsole(call goja.FunctionCall) goja.Value {
	s.console.write(s.ctx, ConsoleEntry{
		Level:   ConsoleLevel(call.Argument(0).String()),
		Message: call.Argument(1).String(),
	})

	return goja.Undefined()
}

// consoleRecorder keeps console output within a byte budget.
type consoleRecorder struct {
	entries   []ConsoleEntry
	bytes     int
	maxBytes  int
	truncated bool

	onConsole func(ctx context.Context, entry ConsoleEntry)
}

func (c *consoleRecorder) write(ctx context.Context, entry ConsoleEntry) {
	if c.truncated {
		return
	}

	remaining := c.maxBytes - c.bytes
	if len(entry.Message) > remaining {
		entry.Message = strings.ToValidUTF8(entry.Message[:remaining], "")
		c.truncated = true
	}

	c.bytes += len(entry.Message)
	c.entries = append(c.entries, entry)

	if c.onConsole != nil {
		c.onConsole(ctx, entry)
	}
}
