package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// FileReporterOptions configure NewFileReporter.
type FileReporterOptions struct {
	// Dir is the directory of the events file. It is created when missing,
	// accessible only by its owner.
	Dir string

	// FailOnly buffers each turn's records and writes them only when the turn
	// produced an error: an error diagnostic, a failed tool or an error end.
	FailOnly bool

	// FileName is the file name within Dir. Empty selects "events.jsonl".
	FileName string
}

// FileReporter is a telemetry integration that writes one JSON Lines file
// with both the span lifecycle of each turn (turn, step and tool) and the
// forwarded diagnostics. Register it in AgentSettings.Telemetry; it
// implements DiagnosticConsumer, so it also receives diagnostics. The file
// may hold prompts and tool data, so it is readable only by its owner.
// Writes are best-effort and never fail a turn.
type FileReporter struct {
	ai.NopTelemetry

	path     string
	dir      string
	failOnly bool

	mu           sync.Mutex
	turns        map[string]*reportedTurn
	lastOpenCall string
	dirReady     bool
}

// reportedTurn buffers the records of one turn.
type reportedTurn struct {
	records []fileRecord
	errored bool
}

// fileRecordKind identifies a record of the events file.
type fileRecordKind string

// File record kinds.
const (
	recordTurnStart  fileRecordKind = "turn-start"
	recordStepStart  fileRecordKind = "step-start"
	recordToolStart  fileRecordKind = "tool-start"
	recordToolEnd    fileRecordKind = "tool-end"
	recordStepFinish fileRecordKind = "step-finish"
	recordTurnFinish fileRecordKind = "turn-finish"
	recordError      fileRecordKind = "error"
	recordTurnAbort  fileRecordKind = "turn-abort"
	recordDiagnostic fileRecordKind = "diagnostic"
)

// fileRecord is one line of the events file. Each kind sets the fields
// that describe it.
type fileRecord struct {
	Kind      fileRecordKind `json:"kind"`
	Timestamp int64          `json:"ts"`
	CallID    string         `json:"callId,omitempty"`

	OperationID string `json:"operationId,omitempty"`
	Provider    string `json:"provider,omitempty"`
	ModelID     string `json:"modelId,omitempty"`

	Step *int `json:"step,omitempty"`

	ToolName   string `json:"toolName,omitempty"`
	ToolCallID string `json:"toolCallId,omitempty"`

	// Input is the prompt of a turn-start record or the input of a
	// tool-start record.
	Input json.RawMessage `json:"input,omitempty"`

	IsError *bool `json:"isError,omitempty"`

	// Output is a tool's output or error text, or the content of a step.
	Output any `json:"output,omitempty"`

	Usage        *ai.LanguageModelUsage    `json:"usage,omitempty"`
	FinishReason model.UnifiedFinishReason `json:"finishReason,omitempty"`
	Error        string                    `json:"error,omitempty"`
	Reason       string                    `json:"reason,omitempty"`
	Diagnostic   *Diagnostic               `json:"diagnostic,omitempty"`
}

// turnInput is the prompt of a turn-start record.
type turnInput struct {
	Messages     []model.Message `json:"messages"`
	Instructions string          `json:"instructions"`
}

// NewFileReporter returns a file reporter.
func NewFileReporter(options FileReporterOptions) *FileReporter {
	fileName := options.FileName
	if fileName == "" {
		fileName = "events.jsonl"
	}

	return &FileReporter{
		path:     filepath.Join(options.Dir, fileName),
		dir:      options.Dir,
		failOnly: options.FailOnly,
		turns:    make(map[string]*reportedTurn),
	}
}

// OnGenerateTextStart records the start of a turn and, unless inputs are
// not recorded, its prompt.
func (r *FileReporter) OnGenerateTextStart(_ context.Context, event ai.TelemetryEvent[ai.GenerateTextStartEvent]) {
	record := fileRecord{
		Kind:        recordTurnStart,
		CallID:      event.Event.CallID,
		OperationID: event.Event.OperationID,
		Provider:    event.Event.Provider,
		ModelID:     event.Event.ModelID,
	}

	if event.RecordInputs {
		record.Input = encodeTurnInput(turnInput{
			Messages:     event.Event.Messages,
			Instructions: event.Event.System,
		})
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.lastOpenCall = event.Event.CallID
	r.record(record)
}

// encodeTurnInput encodes a turn's prompt, or returns nil when it cannot be
// encoded.
func encodeTurnInput(input turnInput) json.RawMessage {
	encoded, err := json.Marshal(input)
	if err != nil {
		return nil
	}

	return encoded
}

// OnGenerateTextStepStart records the start of a step.
func (r *FileReporter) OnGenerateTextStepStart(_ context.Context, event ai.TelemetryEvent[ai.GenerateTextStepStartEvent]) {
	step := event.Event.StepNumber

	r.mu.Lock()
	defer r.mu.Unlock()

	r.record(fileRecord{
		Kind:   recordStepStart,
		CallID: event.Event.CallID,
		Step:   &step,
	})
}

// OnToolExecutionStart records the start of a tool execution.
func (r *FileReporter) OnToolExecutionStart(_ context.Context, event ai.TelemetryEvent[ai.ToolExecutionStartEvent]) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.record(fileRecord{
		Kind:       recordToolStart,
		CallID:     event.Event.CallID,
		ToolName:   event.Event.ToolCall.ToolName,
		ToolCallID: event.Event.ToolCall.ToolCallID,
		Input:      event.Event.ToolCall.Input,
	})
}

// OnToolExecutionEnd records the end of a tool execution.
func (r *FileReporter) OnToolExecutionEnd(_ context.Context, event ai.TelemetryEvent[ai.ToolExecutionEndEvent]) {
	record := fileRecord{
		Kind:       recordToolEnd,
		CallID:     event.Event.CallID,
		ToolCallID: event.Event.ToolCall.ToolCallID,
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	switch output := event.Event.Output.(type) {
	case ai.ToolError:
		isError := true
		record.IsError = &isError
		record.Output = errorText(output.Error)
		r.bucket(event.Event.CallID).errored = true
	case ai.ToolResult:
		isError := false
		record.IsError = &isError
		record.Output = output.Output
	}

	r.record(record)
}

// OnGenerateTextStepEnd records a finished step and, unless outputs are
// not recorded, its content.
func (r *FileReporter) OnGenerateTextStepEnd(_ context.Context, event ai.TelemetryEvent[ai.StepResult]) {
	usage := event.Event.Usage

	record := fileRecord{
		Kind:   recordStepFinish,
		CallID: event.Event.CallID,
		Usage:  &usage,
	}

	if event.RecordOutputs {
		record.Output = event.Event.Content
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.record(record)
}

// OnGenerateTextEnd records the end of a turn and flushes its records.
func (r *FileReporter) OnGenerateTextEnd(_ context.Context, event ai.TelemetryEvent[ai.GenerateTextEndEvent]) {
	usage := event.Event.TotalUsage

	r.mu.Lock()
	defer r.mu.Unlock()

	r.record(fileRecord{
		Kind:         recordTurnFinish,
		CallID:       event.Event.CallID,
		FinishReason: event.Event.FinishReason,
		Usage:        &usage,
	})
	r.finishTurn(event.Event.CallID)
}

// OnError records a failed turn and flushes its records.
func (r *FileReporter) OnError(_ context.Context, event ai.TelemetryEvent[ai.TelemetryErrorEvent]) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.bucket(event.Event.CallID).errored = true
	r.record(fileRecord{
		Kind:   recordError,
		CallID: event.Event.CallID,
		Error:  errorText(event.Event.Error),
	})
	r.finishTurn(event.Event.CallID)
}

// OnGenerateTextAbort records a cancelled turn and flushes its records.
func (r *FileReporter) OnGenerateTextAbort(_ context.Context, event ai.TelemetryEvent[ai.StreamTextAbortEvent]) {
	record := fileRecord{
		Kind:   recordTurnAbort,
		CallID: event.Event.CallID,
	}

	if event.Event.Reason != nil {
		record.Reason = event.Event.Reason.Error()
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.record(record)
	r.finishTurn(event.Event.CallID)
}

// IngestDiagnostic records a diagnostic in the most recently started turn.
func (r *FileReporter) IngestDiagnostic(diagnostic Diagnostic) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if diagnostic.Level == DebugError && r.lastOpenCall != "" {
		r.bucket(r.lastOpenCall).errored = true
	}

	r.recordAt(diagnostic.Timestamp, fileRecord{
		Kind:       recordDiagnostic,
		Diagnostic: &diagnostic,
	})
}

// record adds a record stamped with the current time. The caller holds mu.
func (r *FileReporter) record(record fileRecord) {
	r.recordAt(time.Now(), record)
}

// recordAt adds a record to the turn of its call, or of the most recently
// started turn, and writes it directly outside a turn. The caller holds mu.
func (r *FileReporter) recordAt(timestamp time.Time, record fileRecord) {
	record.Timestamp = timestamp.UnixMilli()

	callID := record.CallID
	if callID == "" {
		callID = r.lastOpenCall
	}

	if callID == "" {
		r.flush([]fileRecord{
			record,
		})

		return
	}

	bucket := r.bucket(callID)
	bucket.records = append(bucket.records, record)
}

// bucket returns the buffer of a turn. The caller holds mu.
func (r *FileReporter) bucket(callID string) *reportedTurn {
	turn, ok := r.turns[callID]
	if !ok {
		turn = &reportedTurn{}
		r.turns[callID] = turn
	}

	return turn
}

// finishTurn writes a turn's records unless it succeeded in fail-only mode.
// The caller holds mu.
func (r *FileReporter) finishTurn(callID string) {
	turn, ok := r.turns[callID]
	if !ok {
		return
	}

	delete(r.turns, callID)

	if r.lastOpenCall == callID {
		r.lastOpenCall = ""
	}

	if r.failOnly && !turn.errored {
		return
	}

	r.flush(turn.records)
}

// flush appends records to the file, best-effort. The directory and the
// file are accessible only by their owner. The caller holds mu.
func (r *FileReporter) flush(records []fileRecord) {
	if len(records) == 0 {
		return
	}

	if !r.dirReady {
		if err := os.MkdirAll(r.dir, 0o700); err != nil {
			return
		}

		r.dirReady = true
	}

	file, err := os.OpenFile(r.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}

	defer func() {
		_ = file.Close()
	}()

	encoder := json.NewEncoder(file)
	for _, record := range records {
		_ = encoder.Encode(record)
	}
}

// TraceTreeReporterOptions configure NewTraceTreeReporter.
type TraceTreeReporterOptions struct {
	// Writer receives the rendered tree. Nil writes to standard error.
	Writer io.Writer
}

// TraceTreeReporter is a telemetry integration that renders a text tree of
// each turn's spans (turn, steps and tools with durations) when the turn
// ends. It is a zero-setup local debugging aid; an OpenTelemetry backend is
// a strict superset.
type TraceTreeReporter struct {
	ai.NopTelemetry

	writer io.Writer

	mu    sync.Mutex
	turns map[string]*traceTurn
}

// traceTurn is the span tree of one turn.
type traceTurn struct {
	root  *traceNode
	step  *traceNode
	tools map[string]*traceNode
}

// traceNode is one span.
type traceNode struct {
	label    string
	start    time.Time
	end      time.Time
	children []*traceNode
}

// NewTraceTreeReporter returns a trace tree reporter.
func NewTraceTreeReporter(options TraceTreeReporterOptions) *TraceTreeReporter {
	writer := options.Writer
	if writer == nil {
		writer = os.Stderr
	}

	return &TraceTreeReporter{
		writer: writer,
		turns:  make(map[string]*traceTurn),
	}
}

// OnGenerateTextStart opens the turn span.
func (r *TraceTreeReporter) OnGenerateTextStart(_ context.Context, event ai.TelemetryEvent[ai.GenerateTextStartEvent]) {
	label := event.Event.OperationID
	if label == "" {
		label = "turn"
	}

	if event.Event.ModelID != "" {
		label += " " + event.Event.ModelID
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.turns[event.Event.CallID] = &traceTurn{
		root: &traceNode{
			label: label,
			start: time.Now(),
		},
		tools: make(map[string]*traceNode),
	}
}

// OnGenerateTextStepStart opens a step span.
func (r *TraceTreeReporter) OnGenerateTextStepStart(_ context.Context, event ai.TelemetryEvent[ai.GenerateTextStepStartEvent]) {
	r.mu.Lock()
	defer r.mu.Unlock()

	turn, ok := r.turns[event.Event.CallID]
	if !ok {
		return
	}

	step := &traceNode{
		label: fmt.Sprintf("step %d", event.Event.StepNumber+1),
		start: time.Now(),
	}

	turn.step = step
	turn.root.children = append(turn.root.children, step)
}

// OnToolExecutionStart opens a tool span under the current step.
func (r *TraceTreeReporter) OnToolExecutionStart(_ context.Context, event ai.TelemetryEvent[ai.ToolExecutionStartEvent]) {
	r.mu.Lock()
	defer r.mu.Unlock()

	turn, ok := r.turns[event.Event.CallID]
	if !ok {
		return
	}

	parent := turn.root
	if turn.step != nil {
		parent = turn.step
	}

	node := &traceNode{
		label: "tool " + event.Event.ToolCall.ToolName,
		start: time.Now(),
	}

	turn.tools[event.Event.ToolCall.ToolCallID] = node
	parent.children = append(parent.children, node)
}

// OnToolExecutionEnd closes a tool span.
func (r *TraceTreeReporter) OnToolExecutionEnd(_ context.Context, event ai.TelemetryEvent[ai.ToolExecutionEndEvent]) {
	r.mu.Lock()
	defer r.mu.Unlock()

	turn, ok := r.turns[event.Event.CallID]
	if !ok {
		return
	}

	node, ok := turn.tools[event.Event.ToolCall.ToolCallID]
	if !ok {
		return
	}

	node.end = time.Now()

	if _, failed := event.Event.Output.(ai.ToolError); failed {
		node.label += " [error]"
	}
}

// OnGenerateTextStepEnd closes the current step span.
func (r *TraceTreeReporter) OnGenerateTextStepEnd(_ context.Context, event ai.TelemetryEvent[ai.StepResult]) {
	r.mu.Lock()
	defer r.mu.Unlock()

	turn, ok := r.turns[event.Event.CallID]
	if !ok || turn.step == nil {
		return
	}

	turn.step.end = time.Now()
	turn.step = nil
}

// OnGenerateTextEnd closes the turn span and renders the tree.
func (r *TraceTreeReporter) OnGenerateTextEnd(_ context.Context, event ai.TelemetryEvent[ai.GenerateTextEndEvent]) {
	r.mu.Lock()
	defer r.mu.Unlock()

	turn, ok := r.turns[event.Event.CallID]
	if !ok {
		return
	}

	delete(r.turns, event.Event.CallID)
	turn.root.end = time.Now()

	var rendered strings.Builder
	rendered.WriteString("\n")
	renderTraceNode(&rendered, turn.root, 0)

	_, _ = io.WriteString(r.writer, rendered.String())
}

// renderTraceNode renders a span and its children.
func renderTraceNode(rendered *strings.Builder, node *traceNode, depth int) {
	duration := "(open)"
	if !node.end.IsZero() {
		duration = fmt.Sprintf("%dms", max(0, node.end.Sub(node.start).Milliseconds()))
	}

	fmt.Fprintf(rendered, "%s- %s %s\n", strings.Repeat("  ", depth), node.label, duration)

	for _, child := range node.children {
		renderTraceNode(rendered, child, depth+1)
	}
}
