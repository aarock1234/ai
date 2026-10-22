package codemode

import (
	"encoding/json"
)

// replay is how an execution starts: the replay state of the interrupted
// execution it resumes, or a fresh state, and the resolutions of the
// resumed interruptions by run interruption ID. A nil resolution is
// undefined.
type replay struct {
	state       runState
	resolutions map[string]json.RawMessage
}

// resumedCall is an interrupted host tool call and its resolution.
type resumedCall struct {
	payload    InterruptPayload
	resolution json.RawMessage
}

// recordedEntry returns the ledger entry an earlier execution recorded for
// a request.
func recordedEntry(ledger []ledgerEntry, request int) (ledgerEntry, bool) {
	if request > len(ledger) || ledger[request-1].Status == "" {
		return ledgerEntry{}, false
	}

	return ledger[request-1], true
}

// lastSettledOrder returns the highest settled order of a ledger.
func lastSettledOrder(ledger []ledgerEntry) int {
	last := 0

	for _, entry := range ledger {
		if entry.Status != ledgerInterrupted {
			last = max(last, entry.SettledOrder)
		}
	}

	return last
}

// resumeInterruptedEntry returns the payload of an interrupted entry and
// the resolution that resumes it.
func (r *replay) resumeInterruptedEntry(entry ledgerEntry) (*resumedCall, error) {
	resolution, found := r.resolutions[entry.InterruptionID]
	if !found {
		return nil, newProtocolError("code mode continuation has an unresolved interruption " + entry.InterruptionID)
	}

	document, err := decodeRunValue(entry.PayloadJSON)
	if err != nil {
		return nil, newProtocolError("code mode interruption payload is malformed")
	}

	var payload InterruptPayload
	if err := json.Unmarshal(document, &payload); err != nil {
		return nil, newProtocolError("code mode interruption payload is malformed")
	}

	return &resumedCall{
		payload:    payload,
		resolution: resolution,
	}, nil
}
