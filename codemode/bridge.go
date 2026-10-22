package codemode

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/dop251/goja"
)

// runErrorName is the name of the JavaScript error a failed host tool call
// gives the program, as upstream's runtime names it.
const runErrorName = "RunError"

// bridgeRequest is a host tool call that the program is awaiting.
type bridgeRequest struct {
	index         int
	bindingName   string
	argumentsJSON string
	resolve       func(value any) error
	reject        func(reason any) error
}

// bridgeCompletion is the outcome of a live host tool call and the host
// clock when it settled.
type bridgeCompletion struct {
	request     int
	outcome     hostOutcome
	settledAtMs int64
}

// bridgeResponse is a settled host tool call waiting for its turn: the
// program receives outcomes in the order the calls settled.
type bridgeResponse struct {
	request   int
	dateNowMs int64

	// output is the JSON of the call's output; nil is undefined.
	output json.RawMessage

	// failure is the error the program receives, and cause the Go error
	// it reports when the program does not catch it.
	failure *ledgerError
	cause   error
}

// callTool implements the host side of tools.name(input). The program
// calls it with the host function's binding name and its serialized
// argument list when it first awaits the call, and receives a promise that
// settles with the call's outcome.
func (s *sandbox) callTool(call goja.FunctionCall) goja.Value {
	promise, resolve, reject := s.vm.NewPromise()

	request := &bridgeRequest{
		bindingName:   call.Argument(0).String(),
		argumentsJSON: call.Argument(1).String(),
		resolve:       resolve,
		reject:        reject,
	}

	if err := s.open(request); err != nil {
		// Errors from the runtime itself, such as an interrupt, are
		// uncatchable and must unwind it, so they are raised as panics,
		// which goja recovers.
		reason, reasonErr := s.programError(recordError(err), err)
		if reasonErr != nil {
			panic(reasonErr)
		}

		if rejectErr := reject(reason); rejectErr != nil {
			panic(rejectErr)
		}
	}

	return s.vm.ToValue(promise)
}

// open numbers a host tool call and starts it, or replays the outcome an
// earlier execution recorded for it.
func (s *sandbox) open(request *bridgeRequest) error {
	if s.fatal != nil {
		return s.fatal
	}

	s.requestCount++
	request.index = s.requestCount

	maxRequests := s.input.limits.maxBridgeRequests
	if request.index > maxRequests {
		return s.fail(newBridgeLimitError(
			fmt.Sprintf("program exceeded the limit of %d host tool calls", maxRequests),
			maxRequests,
		))
	}

	s.requests[request.index] = request

	call := hostCall{
		request:       request.index,
		bindingName:   request.bindingName,
		argumentsJSON: request.argumentsJSON,
	}

	entry, recorded := recordedEntry(s.ledger, request.index)
	if !recorded {
		return s.start(call)
	}

	if entry.BindingName != request.bindingName || entry.InputJSON != request.argumentsJSON {
		return s.fail(newProtocolError("code mode continuation replay diverged from the recorded host tool calls"))
	}

	if entry.Status != ledgerInterrupted {
		return s.queueRecorded(request.index, entry)
	}

	resumed, err := s.input.replay.resumeInterruptedEntry(entry)
	if err != nil {
		return s.fail(err)
	}

	call.resumed = resumed

	return s.start(call)
}

// queueRecorded queues the outcome an earlier execution recorded for a
// request at its original turn.
func (s *sandbox) queueRecorded(request int, entry ledgerEntry) error {
	response := bridgeResponse{
		request:   request,
		dateNowMs: entry.DateNowMs,
	}

	if entry.Status == ledgerRejected {
		response.failure = entry.Error
		response.cause = &Error{
			Code:    ErrorCode(entry.Error.Code),
			Message: entry.Error.Message,
		}
	} else {
		output, err := decodeRunValue(entry.ValueJSON)
		if err != nil {
			return s.fail(newProtocolError("code mode continuation has a malformed recorded output: " + err.Error()))
		}

		response.output = output
	}

	return s.queueResponse(entry.SettledOrder, response)
}

func (s *sandbox) queueResponse(order int, response bridgeResponse) error {
	if _, queued := s.responses[order]; queued || order < s.nextResponse {
		return s.fail(newProtocolError("code mode continuation has a duplicate settled order"))
	}

	s.responses[order] = response

	return nil
}

// start runs a live host tool call on a worker goroutine.
func (s *sandbox) start(call hostCall) error {
	maxInFlight := s.input.limits.maxInFlightBridgeRequests
	if s.inFlight >= maxInFlight {
		return s.fail(newBridgeLimitError(
			fmt.Sprintf("program exceeded the limit of %d concurrent host tool calls", maxInFlight),
			maxInFlight,
		))
	}

	s.inFlight++

	s.workers.Go(func() {
		outcome := s.input.host.invoke(s.ctx, call)

		s.completions <- bridgeCompletion{
			request:     call.request,
			outcome:     outcome,
			settledAtMs: time.Now().UnixMilli(),
		}
	})

	return nil
}

// fail ends the execution with err: the runtime is interrupted as soon as
// the program runs again.
func (s *sandbox) fail(err error) error {
	s.fatal = err
	s.vm.Interrupt(err)

	return err
}

// drive delivers host tool outcomes to the program until its promise
// settles or it cannot make progress: no call is running and no outcome
// can be delivered.
func (s *sandbox) drive(promise *goja.Promise) error {
	for {
		if err := s.deliverReady(); err != nil {
			return err
		}

		if promise.State() != goja.PromiseStatePending || s.inFlight == 0 {
			return nil
		}

		completion, err := s.awaitCompletion()
		if err != nil {
			return err
		}

		if err := s.settleLive(completion); err != nil {
			return err
		}
	}
}

// deliverReady delivers the queued outcomes whose turn has come.
func (s *sandbox) deliverReady() error {
	for s.fatal == nil {
		response, ready := s.responses[s.nextResponse]
		if !ready {
			return nil
		}

		delete(s.responses, s.nextResponse)
		s.nextResponse++

		if err := s.deliver(response); err != nil {
			return err
		}
	}

	return s.fatal
}

func (s *sandbox) awaitCompletion() (bridgeCompletion, error) {
	select {
	case completion := <-s.completions:
		s.inFlight--

		return completion, nil

	case <-s.ctx.Done():
		return bridgeCompletion{}, contextFailure(s.ctx)
	}
}

// settleLive records the outcome of a live call in the ledger and queues it
// for the program, or records the interrupt it requested.
func (s *sandbox) settleLive(completion bridgeCompletion) error {
	request := s.requests[completion.request]
	outcome := completion.outcome

	if outcome.interrupt != nil {
		return s.recordInterruption(request, outcome)
	}

	s.lastSettledOrder++

	entry := ledgerEntry{
		BindingName:  request.bindingName,
		InputJSON:    request.argumentsJSON,
		SettledOrder: s.lastSettledOrder,
		DateNowMs:    completion.settledAtMs,
	}

	response := bridgeResponse{
		request:   request.index,
		dateNowMs: completion.settledAtMs,
	}

	if outcome.err != nil {
		failure := recordError(outcome.err)

		entry.Status = ledgerRejected
		entry.Error = failure
		response.failure = failure
		response.cause = outcome.err
	} else {
		valueJSON, err := encodeRunValue(json.RawMessage(outcome.output))
		if err != nil {
			return s.fail(fmt.Errorf("code mode: encode host tool output: %w", err))
		}

		entry.Status = ledgerFulfilled
		entry.ValueJSON = valueJSON
		response.output = json.RawMessage(outcome.output)
	}

	s.setLedgerEntry(request.index, entry)

	return s.queueResponse(entry.SettledOrder, response)
}

// recordInterruption records a call that requested an interrupt. A call
// that interrupts again after it was resumed keeps its interruption ID.
func (s *sandbox) recordInterruption(request *bridgeRequest, outcome hostOutcome) error {
	interruptionID := runInterruptionID(request.index)

	if previous, recorded := recordedEntry(s.ledger, request.index); recorded && previous.Status == ledgerInterrupted {
		interruptionID = previous.InterruptionID
	}

	payload, err := json.Marshal(outcome.interrupt)
	if err != nil {
		return s.fail(fmt.Errorf("code mode: encode interrupt payload: %w", err))
	}

	payloadJSON, err := encodeRunValue(payload)
	if err != nil {
		return s.fail(fmt.Errorf("code mode: encode interrupt payload: %w", err))
	}

	s.setLedgerEntry(request.index, ledgerEntry{
		BindingName:    request.bindingName,
		InputJSON:      request.argumentsJSON,
		Status:         ledgerInterrupted,
		InterruptionID: interruptionID,
		PayloadJSON:    payloadJSON,
	})

	s.interruptions = append(s.interruptions, interruptedCall{
		request:        request.index,
		interruptionID: interruptionID,
		toolName:       outcome.toolName,
		input:          outcome.input,
		payload:        *outcome.interrupt,
	})

	return nil
}

func (s *sandbox) setLedgerEntry(request int, entry ledgerEntry) {
	for len(s.ledger) < request {
		s.ledger = append(s.ledger, ledgerEntry{})
	}

	s.ledger[request-1] = entry
}

// deliver settles a request with its outcome after resetting the program's
// clock to the host clock when the call settled.
func (s *sandbox) deliver(response bridgeResponse) error {
	s.clock.reset(response.dateNowMs)

	request := s.requests[response.request]

	if response.failure != nil {
		reason, err := s.programError(response.failure, response.cause)
		if err != nil {
			return s.failure(err)
		}

		if err := request.reject(reason); err != nil {
			return s.failure(err)
		}

		return nil
	}

	value := goja.Undefined()

	if response.output != nil {
		parsed, err := s.helpers.parse(goja.Undefined(), s.vm.ToValue(string(response.output)))
		if err != nil {
			return s.failure(err)
		}

		value = parsed
	}

	if err := request.resolve(value); err != nil {
		return s.failure(err)
	}

	return nil
}

// programError creates the JavaScript error that the program receives for
// a failed host tool call, with the failure's name, message and code. The
// program's error maps back to cause, so that Run returns cause when the
// program does not catch it.
func (s *sandbox) programError(failure *ledgerError, cause error) (goja.Value, error) {
	code := goja.Undefined()
	if failure.Code != "" {
		code = s.vm.ToValue(failure.Code)
	}

	reason, err := s.helpers.createError(
		goja.Undefined(),
		s.vm.ToValue(failure.Message),
		s.vm.ToValue(failure.Name),
		code,
	)
	if err != nil {
		return nil, err
	}

	if object, ok := reason.(*goja.Object); ok {
		s.hostErrors[object] = cause
	}

	return reason, nil
}

// recordError returns the error a failed host tool call gives the program:
// the message and code of a code mode error, and a generic host tool
// failure otherwise.
func recordError(err error) *ledgerError {
	if codeModeErr, ok := errors.AsType[*Error](err); ok {
		return &ledgerError{
			Code:    string(codeModeErr.Code),
			Message: codeModeErr.Message,
			Name:    runErrorName,
		}
	}

	return &ledgerError{
		Code:    string(ErrorCodeHostTool),
		Message: "host tool failed",
		Name:    runErrorName,
	}
}
