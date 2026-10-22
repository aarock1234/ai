// Package toolcalls correlates streamed tool call deltas from OpenAI-compatible
// chat streams. It ports the upstream streaming tool call tracker and argument
// state shared by the OpenAI, OpenAI-compatible and Mistral providers.
package toolcalls

import (
	"cmp"
	"fmt"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

// Delta is one streamed fragment of a tool call in an OpenAI-compatible chat
// stream. Nil fields were absent.
type Delta struct {
	Index     *int
	ID        *string
	Type      *string
	Name      *string
	Arguments *string

	// ProviderMetadata is attached to the finished tool call when this delta
	// starts it.
	ProviderMetadata model.ProviderMetadata
}

// TypeValidation selects how the type of a delta that starts a call is
// checked.
type TypeValidation int

const (
	// ValidateTypeNone accepts any type.
	ValidateTypeNone TypeValidation = iota

	// ValidateTypeIfPresent rejects a present type other than "function".
	ValidateTypeIfPresent

	// ValidateTypeRequired rejects a type other than "function", including
	// an absent one.
	ValidateTypeRequired
)

// Options configures a Tracker.
type Options struct {
	// GenerateID names calls without a usable wire ID. Blank or repeated
	// results are made unique. Nil uses random IDs.
	GenerateID func() string

	// TypeValidation checks the type of deltas that start calls.
	TypeValidation TypeValidation
}

// Tracker correlates streamed tool call deltas, accumulates their arguments
// and produces tool input and tool call stream parts. Calls are finished only
// by Flush, because parsable arguments can still be the prefix of longer
// arguments. Use NewTracker to create it.
type Tracker struct {
	generateID     func() string
	typeValidation TypeValidation

	calls        []*trackedToolCall
	callsByID    map[string][]*trackedToolCall
	callsByIndex map[int][]*trackedToolCall
	usedIDs      map[string]bool
	nextSuffixes map[string]int
}

type trackedToolCall struct {
	id    string
	index *int

	name      string
	arguments strings.Builder
	state     argumentState

	finished bool
	metadata model.ProviderMetadata
}

// resolution is the outcome of matching a delta to the tracked calls.
type resolution int

const (
	resolvedExisting resolution = iota
	resolvedNew
	resolvedAmbiguous
)

// NewTracker returns a tracker configured by options.
func NewTracker(options Options) *Tracker {
	generateID := options.GenerateID
	if generateID == nil {
		generateID = func() string {
			return id.New("")
		}
	}

	return &Tracker{
		generateID:     generateID,
		typeValidation: options.TypeValidation,
		callsByID:      make(map[string][]*trackedToolCall),
		callsByIndex:   make(map[int][]*trackedToolCall),
		usedIDs:        make(map[string]bool),
		nextSuffixes:   make(map[string]int),
	}
}

// ProcessDelta consumes one delta and returns the stream parts it produces.
// A delta that starts a call without a function name is invalid.
func (t *Tracker) ProcessDelta(delta Delta) ([]model.StreamPart, error) {
	hasBlankName := delta.Name != nil && strings.TrimSpace(*delta.Name) == ""
	wireID := nonBlank(delta.ID)
	name := nonBlank(delta.Name)

	call, outcome := t.resolve(resolveInput{
		wireID:               wireID,
		index:                delta.Index,
		name:                 name,
		hasExplicitCallStart: name != "" && startsWithStructuredValue(delta.Arguments),
		hasEmptyArguments:    delta.Arguments == nil || strings.TrimSpace(*delta.Arguments) == "",
	})

	var parts []model.StreamPart

	switch outcome {
	case resolvedAmbiguous:
		// Upstream drops a delta that matches several calls equally well:
		// appending it to a guessed call would corrupt that call's arguments,
		// and failing would end a stream whose other calls are intact.
		return nil, nil

	case resolvedNew:
		// Blank names cannot start a usable call. Continuations that repeat
		// a blank name were matched above.
		if hasBlankName {
			return nil, nil
		}

		if err := t.validateType(delta.Type); err != nil {
			return nil, err
		}

		if name == "" {
			return nil, fmt.Errorf("%w: expected 'function.name' to be a string", model.ErrInvalidResponse)
		}

		call, parts = t.start(delta, wireID, name)

	case resolvedExisting:
		if wireID != "" {
			t.associateID(call, wireID)
		}

		parts = t.continueCall(call, delta)
	}

	if delta.Index != nil {
		t.associateIndex(call, *delta.Index)
	}

	return parts, nil
}

// validateType checks the type of a delta that starts a call.
func (t *Tracker) validateType(callType *string) error {
	isFunction := callType != nil && *callType == "function"

	switch t.typeValidation {
	case ValidateTypeRequired:
		if !isFunction {
			return fmt.Errorf("%w: expected 'function' type", model.ErrInvalidResponse)
		}

	case ValidateTypeIfPresent:
		if callType != nil && !isFunction {
			return fmt.Errorf("%w: expected 'function' type", model.ErrInvalidResponse)
		}
	}

	return nil
}

// Flush finishes every unfinished call and returns its tool input end and
// tool call parts. Calls are ordered by index when every call has one, and
// by arrival otherwise. The stable sort keeps arrival order among calls
// that share an index.
func (t *Tracker) Flush() []model.StreamPart {
	calls := t.calls
	if t.allIndexed() {
		calls = slices.Clone(t.calls)
		slices.SortStableFunc(calls, func(a, b *trackedToolCall) int {
			return cmp.Compare(*a.index, *b.index)
		})
	}

	var parts []model.StreamPart
	for _, call := range calls {
		if !call.finished {
			parts = append(parts, t.finish(call)...)
		}
	}

	return parts
}

func (t *Tracker) allIndexed() bool {
	for _, call := range t.calls {
		if call.index == nil {
			return false
		}
	}

	return true
}

// resolveInput is the correlation evidence of one delta.
type resolveInput struct {
	wireID               string
	index                *int
	name                 string
	hasExplicitCallStart bool
	hasEmptyArguments    bool
}

// resolve matches a delta to a tracked call, a new call, or neither.
//
// | ID evidence | index/name evidence | start evidence        | resolution                              |
// | ----------- | ------------------- | --------------------- | --------------------------------------- |
// | known       | matching            | any                   | matching call, new call, or ambiguity   |
// | known       | conflicting         | named                 | new call                                |
// | unseen      | matching            | structured start      | new call                                |
// | unseen      | matching            | named empty arguments | incomplete call, new call, or ambiguity |
// | unseen      | matching            | continuation          | matching call or ambiguity              |
// | absent      | matching            | any                   | matching call, new call, or ambiguity   |
// | absent      | absent              | named                 | new call                                |
// | absent      | absent              | unnamed               | sole unfinished call, new call, or ambiguity |
func (t *Tracker) resolve(input resolveInput) (*trackedToolCall, resolution) {
	var indexed []*trackedToolCall
	if input.index != nil {
		indexed = t.callsByIndex[*input.index]
	}

	hasIndexedCalls := len(indexed) > 0
	matchingIndexed := filterByName(indexed, input.name)

	if input.wireID != "" {
		return t.resolveWithID(input, hasIndexedCalls, matchingIndexed)
	}

	if hasIndexedCalls {
		// Repeated names are valid on continuations. A different name at the
		// same index starts a call from a provider that reuses indices.
		return resolveMatching(matchingIndexed, input.hasExplicitCallStart)
	}

	if input.name != "" {
		return nil, resolvedNew
	}

	var unfinished []*trackedToolCall
	for _, call := range t.calls {
		if !call.finished {
			unfinished = append(unfinished, call)
		}
	}

	return resolveMatching(unfinished, false)
}

// resolveWithID matches a delta that carries a wire ID.
func (t *Tracker) resolveWithID(
	input resolveInput,
	hasIndexedCalls bool,
	matchingIndexed []*trackedToolCall,
) (*trackedToolCall, resolution) {
	withID, known := t.callsByID[input.wireID]
	if !known {
		if len(matchingIndexed) == 0 {
			return nil, resolvedNew
		}

		// A named structured start with an unseen ID is a new call. Empty
		// arguments can also come from a continuation that changed its ID,
		// so an incomplete matching call is kept in that case.
		if input.hasExplicitCallStart {
			return nil, resolvedNew
		}

		return resolveMatching(matchingIndexed, input.name != "" && input.hasEmptyArguments)
	}

	if input.index == nil {
		if input.name == "" {
			return resolveMatching(withID, false)
		}

		return resolveMatching(filterByName(withID, input.name), input.hasExplicitCallStart)
	}

	var matching []*trackedToolCall
	for _, call := range matchingIndexed {
		if slices.Contains(withID, call) {
			matching = append(matching, call)
		}
	}

	call, outcome := resolveMatching(matching, input.hasExplicitCallStart)
	if outcome != resolvedNew {
		return call, outcome
	}

	// A named delta with a distinct index starts a call even when its ID and
	// name repeat, because providers may reuse IDs across parallel calls.
	if input.name != "" {
		return nil, resolvedNew
	}

	// Conflicting labels on a continuation cannot be resolved safely.
	if hasIndexedCalls {
		return nil, resolvedAmbiguous
	}

	return resolveMatching(withID, false)
}

// resolveMatching picks the continuation target among candidate calls. A
// fresh structured start is evidence of another call only once the candidate
// has completed its own structured arguments.
func resolveMatching(calls []*trackedToolCall, hasExplicitCallStart bool) (*trackedToolCall, resolution) {
	if len(calls) == 0 {
		return nil, resolvedNew
	}

	if !hasExplicitCallStart && len(calls) == 1 {
		return calls[0], resolvedExisting
	}

	var continuable []*trackedToolCall
	for _, call := range calls {
		if !call.state.hasCompleteStructuredValue() {
			continuable = append(continuable, call)
		}
	}

	if len(continuable) == 1 {
		return continuable[0], resolvedExisting
	}

	if len(continuable) > 1 || !hasExplicitCallStart {
		return nil, resolvedAmbiguous
	}

	return nil, resolvedNew
}

func filterByName(calls []*trackedToolCall, name string) []*trackedToolCall {
	if name == "" {
		return slices.Clone(calls)
	}

	var matching []*trackedToolCall
	for _, call := range calls {
		if call.name == name {
			matching = append(matching, call)
		}
	}

	return matching
}

// start tracks a new call and returns its tool input start and any initial
// arguments.
func (t *Tracker) start(delta Delta, wireID string, name string) (*trackedToolCall, []model.StreamPart) {
	call := &trackedToolCall{
		id:       t.uniqueID(wireID),
		index:    delta.Index,
		name:     name,
		metadata: delta.ProviderMetadata,
	}

	t.calls = append(t.calls, call)
	if wireID != "" {
		t.associateID(call, wireID)
	}

	parts := []model.StreamPart{
		model.ToolInputStart{
			ID:       call.id,
			ToolName: name,
		},
	}

	if delta.Arguments == nil || *delta.Arguments == "" {
		return call, parts
	}

	call.arguments.WriteString(*delta.Arguments)
	call.state.append(*delta.Arguments)

	parts = append(parts, model.ToolInputDelta{
		ID:    call.id,
		Delta: *delta.Arguments,
	})

	return call, parts
}

// continueCall appends a delta's arguments to an unfinished call.
func (t *Tracker) continueCall(call *trackedToolCall, delta Delta) []model.StreamPart {
	if call.finished || delta.Arguments == nil {
		return nil
	}

	call.arguments.WriteString(*delta.Arguments)
	call.state.append(*delta.Arguments)

	return []model.StreamPart{
		model.ToolInputDelta{
			ID:    call.id,
			Delta: *delta.Arguments,
		},
	}
}

func (t *Tracker) finish(call *trackedToolCall) []model.StreamPart {
	call.finished = true

	return []model.StreamPart{
		model.ToolInputEnd{
			ID: call.id,
		},
		model.ToolCall{
			ToolCallID:       call.id,
			ToolName:         call.name,
			Input:            call.arguments.String(),
			ProviderMetadata: call.metadata,
		},
	}
}

func (t *Tracker) associateID(call *trackedToolCall, wireID string) {
	if !slices.Contains(t.callsByID[wireID], call) {
		t.callsByID[wireID] = append(t.callsByID[wireID], call)
	}
}

func (t *Tracker) associateIndex(call *trackedToolCall, index int) {
	if !slices.Contains(t.callsByIndex[index], call) {
		t.callsByIndex[index] = append(t.callsByIndex[index], call)
	}
}

// uniqueID returns the wire ID when it is unused, and otherwise a generated
// ID made unique with a numeric suffix.
func (t *Tracker) uniqueID(wireID string) string {
	if wireID != "" && !t.usedIDs[wireID] {
		t.usedIDs[wireID] = true

		return wireID
	}

	generatedID := t.generateID()
	generated := nonBlank(&generatedID)
	if generated == "" {
		generated = "tool-call"
	}

	if !t.usedIDs[generated] {
		t.usedIDs[generated] = true

		return generated
	}

	suffix := max(t.nextSuffixes[generated], 1)
	for t.usedIDs[fmt.Sprintf("%s-%d", generated, suffix)] {
		suffix++
	}

	unique := fmt.Sprintf("%s-%d", generated, suffix)
	t.usedIDs[unique] = true
	t.nextSuffixes[generated] = suffix + 1

	return unique
}

// nonBlank returns the value, or an empty string when it is absent or blank.
func nonBlank(value *string) string {
	if value == nil || strings.TrimSpace(*value) == "" {
		return ""
	}

	return *value
}
