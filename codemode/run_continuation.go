package codemode

import (
	"encoding/json"
	"fmt"
	"slices"
)

// continuationInput is what prepareContinuation needs to resume an
// invocation.
type continuationInput struct {
	options   RunOptions
	toolNames []string
	limits    executionLimits
	security  continuationSecurity
}

// resumption is how an invocation starts: fresh, by resuming the token of
// an interrupted execution with the resolutions of its interruptions, or
// by exposing the next interruption of a batch without executing anything.
type resumption struct {
	token         string
	resolutions   map[string]json.RawMessage
	nextInterrupt *Interrupt
}

// prepareContinuation validates a continuation and records the resolution
// of its next interruption. Interruptions of one batch are exposed one at a
// time; the program only resumes when the whole batch is resolved.
func prepareContinuation(input continuationInput) (resumption, error) {
	continuation := input.options.Continuation
	resolution := input.options.InterruptResolution

	if continuation == nil {
		if resolution != nil {
			return resumption{}, newProtocolError("a code mode interrupt resolution was provided without continuation state")
		}

		return resumption{}, nil
	}

	if err := verifyResumedContinuation(input); err != nil {
		return resumption{}, err
	}

	pending := continuation.PendingInterruptions
	next := len(continuation.Resolutions)

	if next >= len(pending) || pending[next].InterruptID != resolution.InterruptID {
		return resumption{}, newProtocolError(fmt.Sprintf(
			"interrupt resolution %s does not match the next pending code mode interruption",
			resolution.InterruptID,
		))
	}

	value, err := encodeResolution(pending[next], resolution.Resolution, input.limits.maxToolOutputBytes)
	if err != nil {
		return resumption{}, err
	}

	resolutions := append(slices.Clone(continuation.Resolutions), PendingResolution{
		RunInterruptionID: pending[next].RunInterruptionID,
		Value:             value,
	})

	if len(resolutions) < len(pending) {
		return nextInterruptOfBatch(*continuation, resolutions, input.security)
	}

	if err := checkApprovals(pending, resolutions); err != nil {
		return resumption{}, err
	}

	byInterruption := make(map[string]json.RawMessage, len(resolutions))
	for _, collected := range resolutions {
		byInterruption[collected.RunInterruptionID] = collected.Value
	}

	return resumption{
		token:       continuation.Token,
		resolutions: byInterruption,
	}, nil
}

// verifyResumedContinuation checks that a continuation is authentic and
// belongs to the program and tools being resumed.
func verifyResumedContinuation(input continuationInput) error {
	continuation := input.options.Continuation

	if err := verifyContinuation(*continuation, input.security); err != nil {
		return err
	}

	if continuation.JS != input.options.JS {
		return newProtocolError("code mode continuation source does not match the resumed source")
	}

	if !slices.Equal(continuation.ToolNames, input.toolNames) {
		return newProtocolError("code mode continuation tool names do not match the resumed tools")
	}

	if input.options.InterruptResolution == nil {
		return newProtocolError("a code mode continuation requires an interrupt resolution")
	}

	return nil
}

// nextInterruptOfBatch signs the collected resolutions and exposes the next
// interruption of the batch.
func nextInterruptOfBatch(continuation Continuation, resolutions []PendingResolution, security continuationSecurity) (resumption, error) {
	continuation.Resolutions = resolutions

	signed, err := signContinuation(continuation, security)
	if err != nil {
		return resumption{}, err
	}

	interrupt, err := interruptAt(signed, len(resolutions))
	if err != nil {
		return resumption{}, err
	}

	return resumption{
		nextInterrupt: interrupt,
	}, nil
}

// encodeResolution normalizes a resolution: approval interrupts take an
// approval decision, other interrupts any JSON value within the tool output
// limit. A nil resolution is undefined.
func encodeResolution(pending PendingInterruption, resolution any, maxBytes int) (json.RawMessage, error) {
	if isApprovalPayload(pending.Payload) {
		return encodeApprovalResolution(resolution)
	}

	encoded, err := encodePayload(resolution, maxBytes, fmt.Sprintf("resolution %q", pending.InterruptID))
	if err != nil {
		return nil, err
	}

	if encoded == "" {
		return nil, nil
	}

	return json.RawMessage(encoded), nil
}

// checkApprovals fails the invocation when any approval of the batch was
// denied, before any call of the batch runs.
func checkApprovals(pending []PendingInterruption, resolutions []PendingResolution) error {
	for index, interruption := range pending {
		if !isApprovalPayload(interruption.Payload) {
			continue
		}

		decision, err := decodeApprovalResolution(resolutions[index].Value)
		if err != nil {
			return err
		}

		if !decision.Approved {
			return newApprovalDeniedError(interruption, decision.Reason)
		}
	}

	return nil
}

// replayInput is what newReplay needs to resume a continuation token.
type replayInput struct {
	resumption resumption
	scope      continuationScope
	limits     executionLimits
	security   continuationSecurity
}

// newReplay returns a fresh replay state, or decodes and validates the
// resumed continuation token against the invocation's scope and checks
// that the resolutions resolve exactly its pending interruptions.
func newReplay(input replayInput) (*replay, error) {
	if input.resumption.token == "" {
		return &replay{
			state: freshRunState(),
		}, nil
	}

	document, err := decodeRunToken(input.resumption.token, input.security)
	if err != nil {
		return nil, err
	}

	state, err := decodeRunState(document, input.scope, newStateLimits(input.limits))
	if err != nil {
		return nil, err
	}

	if err := checkResolutions(state.Ledger, input.resumption.resolutions); err != nil {
		return nil, err
	}

	return &replay{
		state:       state,
		resolutions: input.resumption.resolutions,
	}, nil
}

// checkResolutions requires exactly one resolution for every interrupted
// call of the ledger.
func checkResolutions(ledger []ledgerEntry, resolutions map[string]json.RawMessage) error {
	pending := 0

	for _, entry := range ledger {
		if entry.Status != ledgerInterrupted {
			continue
		}

		if _, resolved := resolutions[entry.InterruptionID]; !resolved {
			return newProtocolError("a code mode continuation requires a resolution for every pending interruption")
		}

		pending++
	}

	if pending == 0 || pending != len(resolutions) {
		return newProtocolError("code mode resolutions do not match the pending interruptions of the continuation")
	}

	return nil
}
