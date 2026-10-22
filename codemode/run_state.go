package codemode

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"
)

// The continuation token is the replay state of upstream's run runtime,
// version run-replay-v2, so that continuations created by upstream resume
// here and the other way around.
const (
	runStateVersion      = 2
	runStateRuntime      = "run-replay-v2"
	runStateSerde        = "run-js-v1"
	continuationAudience = "@ai-sdk/code-mode/v2"

	toolBindingNamespace = "__codeMode"
	missingToolBinding   = toolBindingNamespace + ".missing"
	toolBindingPrefix    = toolBindingNamespace + ".tool"

	runInterruptionPrefix = "interrupt-"

	// maxRunLimit is the largest limit upstream's runtime accepts.
	maxRunLimit = 2_147_483_647
)

// runState is the replay state of an interrupted invocation: what it
// needs to replay the program deterministically and the outcome of every
// host function call it made.
type runState struct {
	Determinism  runDeterminism `json:"determinism"`
	Ledger       []ledgerEntry  `json:"ledger"`
	LogicalRunID string         `json:"logicalRunId"`
	Runtime      string         `json:"runtime"`
	ScopeHash    string         `json:"scopeHash"`
	Serde        string         `json:"serde"`
	Source       string         `json:"source"`
	Version      int            `json:"version"`
}

// runDeterminism seeds the program's clock and Math.random, so that a
// replayed program observes the same values.
type runDeterminism struct {
	DateNowMs  int64  `json:"dateNowMs"`
	RandomSeed string `json:"randomSeed"`
}

// ledgerStatus is the outcome of a recorded host function call.
type ledgerStatus string

const (
	ledgerFulfilled   ledgerStatus = "fulfilled"
	ledgerRejected    ledgerStatus = "rejected"
	ledgerInterrupted ledgerStatus = "interrupted"
)

// ledgerEntry is one host function call, at the position of its 1-based
// request number. Values are run-js-v1 payloads.
type ledgerEntry struct {
	BindingName string
	InputJSON   string
	Status      ledgerStatus

	// SettledOrder and DateNowMs are set for settled calls: the order in
	// which calls settled, which is the order the program receives their
	// outcomes in, and the host clock when the call settled.
	SettledOrder int
	DateNowMs    int64

	ValueJSON string
	Error     *ledgerError

	InterruptionID string
	PayloadJSON    string
}

// ledgerError is the error a rejected host function call gave the program.
type ledgerError struct {
	Code    string          `json:"code,omitempty"`
	Details json.RawMessage `json:"details,omitempty"`
	Message string          `json:"message"`
	Name    string          `json:"name"`
	Stack   string          `json:"stack,omitempty"`
}

// fulfilledEntryJSON, rejectedEntryJSON and interruptedEntryJSON are the
// exact wire shapes of the ledger entries.
type fulfilledEntryJSON struct {
	BindingName  string       `json:"bindingName"`
	DateNowMs    int64        `json:"dateNowMs"`
	InputJSON    string       `json:"inputJson"`
	SettledOrder int          `json:"settledOrder"`
	Status       ledgerStatus `json:"status"`
	ValueJSON    string       `json:"valueJson"`
}

type rejectedEntryJSON struct {
	BindingName  string       `json:"bindingName"`
	DateNowMs    int64        `json:"dateNowMs"`
	Error        *ledgerError `json:"error"`
	InputJSON    string       `json:"inputJson"`
	SettledOrder int          `json:"settledOrder"`
	Status       ledgerStatus `json:"status"`
}

type interruptedEntryJSON struct {
	BindingName    string       `json:"bindingName"`
	InputJSON      string       `json:"inputJson"`
	InterruptionID string       `json:"interruptionId"`
	PayloadJSON    string       `json:"payloadJson"`
	Status         ledgerStatus `json:"status"`
}

// MarshalJSON implements json.Marshaler with the shape of the entry's
// status.
func (e ledgerEntry) MarshalJSON() ([]byte, error) {
	switch e.Status {
	case ledgerFulfilled:
		return json.Marshal(fulfilledEntryJSON{
			BindingName:  e.BindingName,
			DateNowMs:    e.DateNowMs,
			InputJSON:    e.InputJSON,
			SettledOrder: e.SettledOrder,
			Status:       e.Status,
			ValueJSON:    e.ValueJSON,
		})

	case ledgerRejected:
		return json.Marshal(rejectedEntryJSON{
			BindingName:  e.BindingName,
			DateNowMs:    e.DateNowMs,
			Error:        e.Error,
			InputJSON:    e.InputJSON,
			SettledOrder: e.SettledOrder,
			Status:       e.Status,
		})

	case ledgerInterrupted:
		return json.Marshal(interruptedEntryJSON{
			BindingName:    e.BindingName,
			InputJSON:      e.InputJSON,
			InterruptionID: e.InterruptionID,
			PayloadJSON:    e.PayloadJSON,
			Status:         e.Status,
		})

	default:
		return nil, errors.New("code mode continuation ledger has a missing entry")
	}
}

// freshRunState returns the state of an invocation that starts without a
// continuation.
func freshRunState() runState {
	return runState{
		Determinism: runDeterminism{
			DateNowMs:  time.Now().UnixMilli(),
			RandomSeed: randomHex(16),
		},
		LogicalRunID: randomHex(16),
		Runtime:      runStateRuntime,
		Serde:        runStateSerde,
		Version:      runStateVersion,
	}
}

func randomHex(size int) string {
	data := make([]byte, size)
	_, _ = rand.Read(data)

	return hex.EncodeToString(data)
}

// continuationScope binds a continuation to the program, the outer tool
// call and the host tools, as upstream's runner scope does.
type continuationScope struct {
	source    string
	scopeHash string
}

// scopeDocument is the runner scope upstream hashes, with members in its
// order.
type scopeDocument struct {
	Audience              string              `json:"audience"`
	BindingManifest       []string            `json:"bindingManifest"`
	ContinuationContext   continuationContext `json:"continuationContext"`
	TransformedSourceHash string              `json:"transformedSourceHash"`
}

// continuationContext is the context upstream's code mode binds its
// continuations to.
type continuationContext struct {
	OuterToolCallID string   `json:"outerToolCallId"`
	ToolNames       []string `json:"toolNames"`
}

// newContinuationScope returns the source upstream runs for a program and
// the hash of its runner scope.
func newContinuationScope(js, outerToolCallID string, toolNames []string) (continuationScope, error) {
	source := codeModeSource(js, toolNames)

	scope := scopeDocument{
		Audience:        continuationAudience,
		BindingManifest: bindingManifest(len(toolNames)),
		ContinuationContext: continuationContext{
			OuterToolCallID: outerToolCallID,
			ToolNames:       toolNames,
		},
		TransformedSourceHash: sha256Hex(source),
	}

	document, err := json.Marshal(scope)
	if err != nil {
		return continuationScope{}, fmt.Errorf("code mode: encode continuation scope: %w", err)
	}

	encoded, err := encodeRunValue(document)
	if err != nil {
		return continuationScope{}, fmt.Errorf("code mode: encode continuation scope: %w", err)
	}

	return continuationScope{
		source:    source,
		scopeHash: sha256Hex(encoded),
	}, nil
}

// codeModeSource is the source upstream gives its runtime for a program:
// the program wrapped with the tools proxy that binds each tool to a host
// function. The continuation stores it and its hash binds the scope, so it
// must match upstream byte for byte.
func codeModeSource(js string, toolNames []string) string {
	type binding struct {
		name      string
		reference string
	}

	bindings := make([]binding, len(toolNames))
	for index, toolName := range toolNames {
		bindings[index] = binding{
			name:      toolName,
			reference: toolBindingName(index),
		}
	}

	bindings = inPropertyOrder(bindings, func(entry binding) string {
		return entry.name
	})

	entries := make([]string, len(bindings))
	for index, entry := range bindings {
		var name strings.Builder
		writeJSONString(&name, entry.name)

		entries[index] = name.String() + ":" + entry.reference
	}

	return "const __codeModeBindings={" + strings.Join(entries, ",") + "};" +
		`const tools=new Proxy(Object.create(null),{get(_target,name){const binding=__codeModeBindings[name];return typeof binding==="function"?(input)=>binding(input):(input)=>__codeMode.missing(String(name),input);}});` +
		"const __codeModeResult=await(async()=>{\n" + js + "\n})();" +
		"if(__codeModeResult===undefined)return undefined;return JSON.parse(JSON.stringify(__codeModeResult));"
}

// bindingManifest lists the host functions of the scope, sorted as
// upstream sorts them.
func bindingManifest(toolCount int) []string {
	names := []string{
		"missing",
	}

	for index := range toolCount {
		names = append(names, "tool"+strconv.Itoa(index))
	}

	slices.SortFunc(names, compareUTF16)

	manifest := make([]string, len(names))
	for index, name := range names {
		manifest[index] = toolBindingNamespace + "." + name
	}

	return manifest
}

// toolBindingName is the host function of the tool at index in the sorted
// tool names.
func toolBindingName(index int) string {
	return toolBindingPrefix + strconv.Itoa(index)
}

func sha256Hex(text string) string {
	sum := sha256.Sum256([]byte(text))

	return hex.EncodeToString(sum[:])
}

func runInterruptionID(request int) string {
	return runInterruptionPrefix + strconv.Itoa(request)
}
