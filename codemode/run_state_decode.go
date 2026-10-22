package codemode

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strconv"
)

var (
	lowerHex32Pattern  = regexp.MustCompile(`^[0-9a-f]{32}$`)
	lowerHex64Pattern  = regexp.MustCompile(`^[0-9a-f]{64}$`)
	bindingNamePattern = regexp.MustCompile(`^[A-Za-z_$][\w$]*\.[^.]+$`)
)

// stateLimits are the limits a continuation's state is validated against,
// as upstream's runtime derives them from the execution policy.
type stateLimits struct {
	maxBridgeRequests int
	maxInputBytes     int
	maxOutputBytes    int
}

func newStateLimits(limits executionLimits) stateLimits {
	return stateLimits{
		maxBridgeRequests: limits.maxBridgeRequests,
		maxInputBytes:     expandedSerializationLimit(limits.maxToolInputBytes),
		maxOutputBytes:    expandedSerializationLimit(limits.maxToolOutputBytes),
	}
}

// expandedSerializationLimit is the limit upstream gives its runtime for a
// serialized payload, whose run-js-v1 encoding is larger than its JSON.
func expandedSerializationLimit(limit int) int {
	return min(maxRunLimit, max(4096, limit*2+1024))
}

// decodeRunState validates the state of a decoded continuation token
// against the scope it resumes, as upstream's runtime does.
func decodeRunState(document json.RawMessage, scope continuationScope, limits stateLimits) (runState, error) {
	invalid := newProtocolError("code mode continuation does not match the supplied source, scope, or runtime version")

	fields, ok := exactObject(document, "determinism", "ledger", "logicalRunId", "runtime", "scopeHash", "serde", "source", "version")
	if !ok {
		return runState{}, invalid
	}

	header := struct {
		LogicalRunID string `json:"logicalRunId"`
		Runtime      string `json:"runtime"`
		ScopeHash    string `json:"scopeHash"`
		Serde        string `json:"serde"`
		Source       string `json:"source"`
		Version      int    `json:"version"`
	}{}

	if err := json.Unmarshal(document, &header); err != nil {
		return runState{}, invalid
	}

	matches := header.Version == runStateVersion &&
		header.Runtime == runStateRuntime &&
		header.Serde == runStateSerde &&
		header.Source == scope.source &&
		lowerHex32Pattern.MatchString(header.LogicalRunID) &&
		lowerHex64Pattern.MatchString(header.ScopeHash) &&
		header.ScopeHash == scope.scopeHash

	if !matches {
		return runState{}, invalid
	}

	determinism, err := decodeDeterminism(fields["determinism"])
	if err != nil {
		return runState{}, err
	}

	ledger, err := decodeLedger(fields["ledger"], limits)
	if err != nil {
		return runState{}, err
	}

	return runState{
		Determinism:  determinism,
		Ledger:       ledger,
		LogicalRunID: header.LogicalRunID,
		Runtime:      header.Runtime,
		ScopeHash:    header.ScopeHash,
		Serde:        header.Serde,
		Source:       header.Source,
		Version:      header.Version,
	}, nil
}

func decodeDeterminism(document json.RawMessage) (runDeterminism, error) {
	invalid := newProtocolError("code mode continuation determinism state is invalid")

	fields, ok := exactObject(document, "dateNowMs", "randomSeed")
	if !ok {
		return runDeterminism{}, invalid
	}

	dateNowMs, ok := safeInteger(fields["dateNowMs"])
	if !ok || dateNowMs < 0 {
		return runDeterminism{}, invalid
	}

	var seed string
	if err := json.Unmarshal(fields["randomSeed"], &seed); err != nil || !lowerHex32Pattern.MatchString(seed) {
		return runDeterminism{}, invalid
	}

	return runDeterminism{
		DateNowMs:  dateNowMs,
		RandomSeed: seed,
	}, nil
}

func decodeLedger(document json.RawMessage, limits stateLimits) ([]ledgerEntry, error) {
	var items []json.RawMessage
	if err := json.Unmarshal(document, &items); err != nil || len(items) == 0 || len(items) > limits.maxBridgeRequests {
		return nil, newProtocolError("code mode continuation ledger length is invalid")
	}

	ledger := make([]ledgerEntry, len(items))
	settledOrders := make(map[int]bool)

	for index, item := range items {
		entry, err := decodeLedgerEntry(item, index, limits)
		if err != nil {
			return nil, err
		}

		if entry.Status != ledgerInterrupted {
			if settledOrders[entry.SettledOrder] {
				return nil, newProtocolError("code mode continuation settled order is duplicated")
			}

			settledOrders[entry.SettledOrder] = true
		}

		ledger[index] = entry
	}

	for order := 1; order <= len(settledOrders); order++ {
		if !settledOrders[order] {
			return nil, newProtocolError("code mode continuation settled order is not unique and contiguous")
		}
	}

	return ledger, nil
}

// ledgerEntryHeader holds the members every ledger entry has.
type ledgerEntryHeader struct {
	BindingName string       `json:"bindingName"`
	InputJSON   string       `json:"inputJson"`
	Status      ledgerStatus `json:"status"`
}

func decodeLedgerEntry(document json.RawMessage, index int, limits stateLimits) (ledgerEntry, error) {
	malformed := newProtocolError(fmt.Sprintf("code mode continuation ledger entry %d is malformed", index))

	var header ledgerEntryHeader
	if err := json.Unmarshal(document, &header); err != nil {
		return ledgerEntry{}, malformed
	}

	validHeader := bindingNamePattern.MatchString(header.BindingName) &&
		len(header.BindingName) <= 1024 &&
		validRunPayload(header.InputJSON, limits.maxInputBytes) &&
		isRunArgumentList(header.InputJSON)

	if !validHeader {
		return ledgerEntry{}, malformed
	}

	entry := ledgerEntry{
		BindingName: header.BindingName,
		InputJSON:   header.InputJSON,
		Status:      header.Status,
	}

	switch header.Status {
	case ledgerFulfilled:
		return decodeFulfilledEntry(document, entry, limits, malformed)

	case ledgerRejected:
		return decodeRejectedEntry(document, entry, limits, malformed)

	case ledgerInterrupted:
		return decodeInterruptedEntry(document, entry, index, limits, malformed)

	default:
		return ledgerEntry{}, malformed
	}
}

func decodeFulfilledEntry(document json.RawMessage, entry ledgerEntry, limits stateLimits, malformed error) (ledgerEntry, error) {
	fields, ok := exactObject(document, "bindingName", "dateNowMs", "inputJson", "settledOrder", "status", "valueJson")
	if !ok {
		return ledgerEntry{}, malformed
	}

	if err := decodeSettlement(fields, &entry); err != nil {
		return ledgerEntry{}, malformed
	}

	if err := json.Unmarshal(fields["valueJson"], &entry.ValueJSON); err != nil || !validRunPayload(entry.ValueJSON, limits.maxOutputBytes) {
		return ledgerEntry{}, malformed
	}

	return entry, nil
}

func decodeRejectedEntry(document json.RawMessage, entry ledgerEntry, limits stateLimits, malformed error) (ledgerEntry, error) {
	fields, ok := exactObject(document, "bindingName", "dateNowMs", "error", "inputJson", "settledOrder", "status")
	if !ok {
		return ledgerEntry{}, malformed
	}

	if err := decodeSettlement(fields, &entry); err != nil {
		return ledgerEntry{}, malformed
	}

	failure, err := decodeLedgerError(fields["error"])
	if err != nil || len(fields["error"]) > limits.maxOutputBytes {
		return ledgerEntry{}, malformed
	}

	entry.Error = failure

	return entry, nil
}

func decodeInterruptedEntry(document json.RawMessage, entry ledgerEntry, index int, limits stateLimits, malformed error) (ledgerEntry, error) {
	fields, ok := exactObject(document, "bindingName", "inputJson", "interruptionId", "payloadJson", "status")
	if !ok {
		return ledgerEntry{}, malformed
	}

	if err := json.Unmarshal(fields["interruptionId"], &entry.InterruptionID); err != nil || entry.InterruptionID != runInterruptionID(index+1) {
		return ledgerEntry{}, malformed
	}

	if err := json.Unmarshal(fields["payloadJson"], &entry.PayloadJSON); err != nil || !validRunPayload(entry.PayloadJSON, limits.maxOutputBytes) {
		return ledgerEntry{}, malformed
	}

	return entry, nil
}

// decodeSettlement reads the settled order and clock of a settled entry.
func decodeSettlement(fields map[string]json.RawMessage, entry *ledgerEntry) error {
	order, orderOK := safeInteger(fields["settledOrder"])
	dateNowMs, dateOK := safeInteger(fields["dateNowMs"])

	if !orderOK || !dateOK || order <= 0 || order > maxRunLimit || dateNowMs < 0 {
		return errors.New("settlement is malformed")
	}

	entry.SettledOrder = int(order)
	entry.DateNowMs = dateNowMs

	return nil
}

// decodeLedgerError reads a serialized error: a name and a message, with an
// optional code, details and stack, and no other members.
func decodeLedgerError(document json.RawMessage) (*ledgerError, error) {
	var members map[string]json.RawMessage
	if err := json.Unmarshal(document, &members); err != nil {
		return nil, err
	}

	for key := range members {
		if !slices.Contains([]string{"code", "details", "message", "name", "stack"}, key) {
			return nil, fmt.Errorf("unexpected error member %q", key)
		}
	}

	var failure ledgerError
	if err := json.Unmarshal(document, &failure); err != nil {
		return nil, err
	}

	if _, ok := members["name"]; !ok {
		return nil, errors.New("error has no name")
	}

	if _, ok := members["message"]; !ok {
		return nil, errors.New("error has no message")
	}

	return &failure, nil
}

func validRunPayload(serialized string, maxBytes int) bool {
	if len(serialized) > maxBytes {
		return false
	}

	_, _, err := newRunDecoder(serialized)

	return err == nil
}

func isRunArgumentList(serialized string) bool {
	decoder, root, err := newRunDecoder(serialized)
	if err != nil {
		return false
	}

	list, ok := decoder.slot(root)

	return ok && list.kind == jsArray && !isTypedEntry(list) && !isSparseEntry(list)
}

// exactObject decodes a JSON object that has exactly the given members.
func exactObject(document json.RawMessage, keys ...string) (map[string]json.RawMessage, bool) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(document, &fields); err != nil || fields == nil {
		return nil, false
	}

	present := slices.Sorted(maps.Keys(fields))

	return fields, slices.Equal(present, slices.Sorted(slices.Values(keys)))
}

// safeInteger reads a JSON number that is an integer JavaScript represents
// exactly.
func safeInteger(document json.RawMessage) (int64, bool) {
	decoder := json.NewDecoder(bytes.NewReader(document))
	decoder.UseNumber()

	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		return 0, false
	}

	number, ok := decoded.(json.Number)
	if !ok {
		return 0, false
	}

	value, err := strconv.ParseFloat(number.String(), 64)
	if err != nil || value != float64(int64(value)) || value > 1<<53-1 || value < -(1<<53-1) {
		return 0, false
	}

	return int64(value), true
}
