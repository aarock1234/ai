package codemode

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// interruptType is the type marker of an Interrupt.
const interruptType = "code-mode-interrupt"

// InterruptPayload describes why a host tool interrupted the invocation.
// It encodes as a JSON object with a "kind" member and the members of
// Fields.
type InterruptPayload struct {
	// Kind identifies the interruption, such as ApprovalInterruptKind.
	Kind string

	// Fields are the other members of the payload.
	Fields map[string]json.RawMessage
}

// MarshalJSON implements json.Marshaler.
func (p InterruptPayload) MarshalJSON() ([]byte, error) {
	members := make(map[string]json.RawMessage, len(p.Fields)+1)
	maps.Copy(members, p.Fields)

	kind, err := json.Marshal(p.Kind)
	if err != nil {
		return nil, err
	}

	members["kind"] = kind

	return json.Marshal(members)
}

// UnmarshalJSON implements json.Unmarshaler. It requires a string kind.
func (p *InterruptPayload) UnmarshalJSON(data []byte) error {
	var members map[string]json.RawMessage
	if err := json.Unmarshal(data, &members); err != nil {
		return fmt.Errorf("decode interrupt payload: %w", err)
	}

	var kind string
	if err := json.Unmarshal(members["kind"], &kind); err != nil {
		return errors.New("interrupt payload must include a string kind")
	}

	delete(members, "kind")

	p.Kind = kind
	p.Fields = nil

	if len(members) > 0 {
		p.Fields = members
	}

	return nil
}

func (p InterruptPayload) clone() InterruptPayload {
	cloned := InterruptPayload{
		Kind: p.Kind,
	}

	if p.Fields != nil {
		cloned.Fields = make(map[string]json.RawMessage, len(p.Fields))
		for key, value := range p.Fields {
			cloned.Fields[key] = slices.Clone(value)
		}
	}

	return cloned
}

// Interrupt is the result of an invocation that stopped because a host tool
// interrupted it, for example to ask for approval. Resume it with
// ContinueInterrupt or, for approvals, ContinueApproval.
type Interrupt struct {
	// Type is always "code-mode-interrupt".
	Type string `json:"type"`

	InterruptID     string           `json:"interruptId"`
	ToolName        string           `json:"toolName"`
	ToolCallID      string           `json:"toolCallId"`
	OuterToolCallID string           `json:"outerToolCallId"`
	Input           json.RawMessage  `json:"input,omitempty"`
	Payload         InterruptPayload `json:"payload"`
	Continuation    Continuation     `json:"continuation"`
}

// InterruptRequest is the error a host tool returns to interrupt the
// invocation. Create it with RequestInterrupt.
type InterruptRequest struct {
	Payload InterruptPayload
}

// Error implements error.
func (r *InterruptRequest) Error() string {
	return fmt.Sprintf("code mode interrupt requested: %s", r.Payload.Kind)
}

// RequestInterrupt returns the error that a host tool's Execute function
// returns to interrupt the invocation. The invocation stops with an
// Interrupt once the program cannot make progress without the call; when
// it is resumed, the tool runs again with the resolution available from
// ResumedInterruptFromContext. The payload needs a non-empty kind.
func RequestInterrupt(payload InterruptPayload) error {
	return &InterruptRequest{
		Payload: payload,
	}
}

// ResumedInterrupt is the resolution of an interrupt, available to the host
// tool that requested it when the invocation resumes.
type ResumedInterrupt struct {
	InterruptID string
	Payload     InterruptPayload

	// Resolution is the JSON encoding of the resolution that resumed the
	// invocation.
	Resolution json.RawMessage
}

type resumedInterruptKey struct{}

// ResumedInterruptFromContext returns the resolution of the interrupt that
// the running host tool requested in an earlier execution.
func ResumedInterruptFromContext(ctx context.Context) (ResumedInterrupt, bool) {
	resumed, ok := ctx.Value(resumedInterruptKey{}).(ResumedInterrupt)

	return resumed, ok
}

func withResumedInterrupt(ctx context.Context, resumed ResumedInterrupt) context.Context {
	return context.WithValue(ctx, resumedInterruptKey{}, resumed)
}

// ContinueInterruptOptions configure ContinueInterrupt.
type ContinueInterruptOptions struct {
	Interrupt *Interrupt

	// Resolution resolves the interrupt. It must encode to JSON within
	// ExecutionPolicy.MaxToolOutputBytes.
	Resolution any

	// Tools must be the host tools of the interrupted invocation.
	Tools []ai.Tool

	// Messages are forwarded to the host tools.
	Messages []model.Message

	Options Options
}

// ContinueInterrupt resumes an interrupted invocation with the resolution
// of its interrupt. Host tool calls that completed before the interrupt
// are replayed from the continuation instead of running again. The result
// is the next interrupt of the same batch, a later interrupt, or the
// program's output.
func ContinueInterrupt(ctx context.Context, options ContinueInterruptOptions) (Result, error) {
	if options.Interrupt == nil {
		return Result{}, newProtocolError("a code mode interrupt is required")
	}

	interrupt := options.Interrupt

	security, err := resolveContinuationSecurity(options.Options.ContinuationSecurity)
	if err != nil {
		return Result{}, err
	}

	if err := verifyContinuation(interrupt.Continuation, security); err != nil {
		return Result{}, err
	}

	if err := checkInterruptLedger(interrupt); err != nil {
		return Result{}, err
	}

	return Run(ctx, RunOptions{
		JS:           interrupt.Continuation.JS,
		Tools:        options.Tools,
		Messages:     options.Messages,
		Options:      options.Options,
		Continuation: &interrupt.Continuation,
		InterruptResolution: &InterruptResolution{
			InterruptID: interrupt.InterruptID,
			Resolution:  options.Resolution,
		},
	})
}

// IsInterrupt reports whether value is an authentic Interrupt: an
// *Interrupt, an Interrupt or its JSON encoding whose continuation verifies
// with security and matches the interrupt's metadata.
func IsInterrupt(value any, security ContinuationSecurity) bool {
	_, ok := interruptFromValue(value, security)

	return ok
}

// FindInterrupt returns the authentic Interrupt in value. Value may be an
// interrupt as accepted by IsInterrupt, a model.JSONOutput or
// model.TextOutput carrying one, an ai.ToolResult or []ai.ToolResult whose
// output is one, or a *ai.GenerateTextResult whose tool results contain
// one.
func FindInterrupt(value any, security ContinuationSecurity) (*Interrupt, bool) {
	switch typed := value.(type) {
	case *ai.GenerateTextResult:
		if typed == nil {
			return nil, false
		}

		return findInterruptInToolResults(typed.ToolResults, security)

	case []ai.ToolResult:
		return findInterruptInToolResults(typed, security)

	case ai.ToolResult:
		return interruptFromValue(typed.Output, security)

	default:
		return interruptFromValue(value, security)
	}
}

func findInterruptInToolResults(results []ai.ToolResult, security ContinuationSecurity) (*Interrupt, bool) {
	for _, result := range results {
		if interrupt, ok := interruptFromValue(result.Output, security); ok {
			return interrupt, true
		}
	}

	return nil, false
}

// interruptFromValue decodes and authenticates an interrupt.
func interruptFromValue(value any, security ContinuationSecurity) (*Interrupt, bool) {
	interrupt, ok := decodeInterrupt(value)
	if !ok || interrupt.Type != interruptType {
		return nil, false
	}

	resolved, err := resolveContinuationSecurity(security)
	if err != nil {
		return nil, false
	}

	if verifyContinuation(interrupt.Continuation, resolved) != nil {
		return nil, false
	}

	if checkInterruptLedger(interrupt) != nil {
		return nil, false
	}

	return interrupt, true
}

func decodeInterrupt(value any) (*Interrupt, bool) {
	switch typed := value.(type) {
	case nil:
		return nil, false

	case *Interrupt:
		return typed, typed != nil

	case Interrupt:
		return &typed, true

	case model.JSONOutput:
		return decodeInterruptJSON(typed.Value)

	case model.TextOutput:
		return decodeInterruptJSON([]byte(typed.Value))

	case json.RawMessage:
		return decodeInterruptJSON(typed)

	case []byte:
		return decodeInterruptJSON(typed)

	case string:
		return decodeInterruptJSON([]byte(typed))

	default:
		encoded, err := json.Marshal(typed)
		if err != nil {
			return nil, false
		}

		return decodeInterruptJSON(encoded)
	}
}

func decodeInterruptJSON(data []byte) (*Interrupt, bool) {
	var interrupt Interrupt
	if err := json.Unmarshal(data, &interrupt); err != nil {
		return nil, false
	}

	return &interrupt, true
}

// checkInterruptLedger verifies that the interrupt's metadata matches the
// next pending interruption of its signed continuation.
func checkInterruptLedger(interrupt *Interrupt) error {
	continuation := interrupt.Continuation

	if continuation.OuterToolCallID != interrupt.OuterToolCallID {
		return newProtocolError("code mode interrupt outer tool call id does not match its continuation")
	}

	next := len(continuation.Resolutions)
	if next >= len(continuation.PendingInterruptions) {
		return newProtocolError("code mode interrupt metadata does not match the signed continuation ledger")
	}

	pending := continuation.PendingInterruptions[next]

	matches := pending.InterruptID == interrupt.InterruptID &&
		pending.ToolCallID == interrupt.ToolCallID &&
		pending.ToolName == interrupt.ToolName &&
		jsonEqual(pending.Input, interrupt.Input) &&
		payloadEqual(pending.Payload, interrupt.Payload)

	if !matches {
		return newProtocolError("code mode interrupt metadata does not match the signed continuation ledger")
	}

	return nil
}

func payloadEqual(left, right InterruptPayload) bool {
	leftJSON, leftErr := codeModeCanonicalJSON(left)
	rightJSON, rightErr := codeModeCanonicalJSON(right)

	return leftErr == nil && rightErr == nil && bytes.Equal(leftJSON, rightJSON)
}

func jsonEqual(left, right json.RawMessage) bool {
	if len(left) == 0 || len(right) == 0 {
		return len(left) == len(right)
	}

	leftJSON, leftErr := codeModeCanonicalJSON(left)
	rightJSON, rightErr := codeModeCanonicalJSON(right)

	return leftErr == nil && rightErr == nil && bytes.Equal(leftJSON, rightJSON)
}

// interruptAt exposes one pending interruption of a continuation.
func interruptAt(continuation Continuation, index int) (*Interrupt, error) {
	if index >= len(continuation.PendingInterruptions) {
		return nil, newProtocolError("code mode continuation has no pending interruption at the requested index")
	}

	pending := clonePendingInterruption(continuation.PendingInterruptions[index])

	return &Interrupt{
		Type:            interruptType,
		InterruptID:     pending.InterruptID,
		ToolName:        pending.ToolName,
		ToolCallID:      pending.ToolCallID,
		OuterToolCallID: continuation.OuterToolCallID,
		Input:           pending.Input,
		Payload:         pending.Payload,
		Continuation:    continuation,
	}, nil
}
