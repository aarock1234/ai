// Package partialjson parses JSON that is still being streamed, by repairing a
// prefix into the most complete valid document it can represent.
package partialjson

import "encoding/json"

// State reports how Parse produced its result.
type State string

// Parse states.
const (
	// Undefined means the input was empty.
	Undefined State = "undefined-input"

	// Successful means the input was already valid JSON.
	Successful State = "successful-parse"

	// Repaired means the input became valid after Fix.
	Repaired State = "repaired-parse"

	// Failed means the input was not a valid JSON prefix.
	Failed State = "failed-parse"
)

// Parse returns text as a valid JSON document, repairing it when it is an
// incomplete prefix. The result is nil unless the state is Successful or
// Repaired.
func Parse(text string) (json.RawMessage, State) {
	if text == "" {
		return nil, Undefined
	}

	if json.Valid([]byte(text)) {
		return json.RawMessage(text), Successful
	}

	repaired := Fix(text)
	if json.Valid([]byte(repaired)) {
		return json.RawMessage(repaired), Repaired
	}

	return nil, Failed
}
