package pi

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/internal/jsonvalue"
	"github.com/aarock1234/ai/pkg/model"
)

// toolKind is how a tool call is dispatched. Pi runs its built-in, MCP and
// extension tools itself; only host tools are handed back to the agent.
type toolKind string

// Tool kinds.
const (
	toolHost      toolKind = "host"
	toolBuiltin   toolKind = "builtin"
	toolMCP       toolKind = "mcp"
	toolExtension toolKind = "extension"
)

// isDynamic reports tools the agent did not declare.
func (k toolKind) isDynamic() bool {
	return k == toolMCP || k == toolExtension
}

// noSummary is the summary of a compaction that reported none.
const noSummary = "(no summary provided)"

// translator ports upstream's PiTranslatorState and translatePiEvent: it
// turns Pi's session events into harness stream parts. Its state encodes to
// JSON, so that a host process that reattaches to a running turn continues
// the open blocks and steps of the process that suspended it.
type translator struct {
	// PromptStarted is set once a turn_start or assistant message_start
	// opened the turn.
	PromptStarted bool `json:"promptStarted"`

	StreamedAssistantText string `json:"streamedAssistantText,omitempty"`
	CurrentTextID         string `json:"currentTextId,omitempty"`
	CurrentReasoningID    string `json:"currentReasoningId,omitempty"`
	ReasoningStarted      bool   `json:"reasoningStarted,omitempty"`

	// ObservedToolNames maps tool call IDs to their public names.
	ObservedToolNames map[string]string `json:"observedToolNames,omitempty"`

	// StreamingToolInputIDs maps content-block indices of the current
	// assistant message to the tool calls whose input is streaming.
	StreamingToolInputIDs map[int]string `json:"streamingToolInputIds,omitempty"`

	// PendingStepToolCallIDs are the calls of the current assistant message
	// that have not completed, in call order.
	PendingStepToolCallIDs []string `json:"pendingStepToolCallIds,omitempty"`

	// FirstStreamedToolCallID is the first call of the current message. Pi
	// starts it first, so its input streams live; inputs of later calls are
	// buffered until Pi starts them.
	FirstStreamedToolCallID string `json:"firstStreamedToolCallId,omitempty"`

	BufferedToolInputs map[string][]bufferedToolInput `json:"bufferedToolInputs,omitempty"`

	// EarlyFinishStepToolCallIDs are the calls an early finish step of the
	// current step already covered.
	EarlyFinishStepToolCallIDs []string `json:"earlyFinishStepToolCallIds,omitempty"`

	StepOpen            bool          `json:"stepOpen,omitempty"`
	StepUsage           *messageUsage `json:"stepUsage,omitempty"`
	TurnReasoningTokens *int          `json:"turnReasoningTokens,omitempty"`

	// LastAssistantError is the error of the latest assistant message. Pi
	// retries failed requests itself, so a later message clears it.
	LastAssistantError string `json:"lastAssistantError,omitempty"`

	// HostToolResults are the outputs the agent submitted for host tools,
	// reported unchanged instead of the text Pi echoes.
	HostToolResults map[string]json.RawMessage `json:"hostToolResults,omitempty"`

	DynamicToolCalls map[string]toolKind `json:"dynamicToolCalls,omitempty"`

	// BuiltinToolNames are the native names of the active built-in tools.
	BuiltinToolNames []string `json:"builtinToolNames,omitempty"`

	// HostToolNames are the names of the turn's application tools.
	HostToolNames []string `json:"hostToolNames,omitempty"`
}

// bufferedToolInput is a tool input part held until Pi starts its call.
type bufferedToolInput struct {
	Kind             string `json:"kind"`
	ToolName         string `json:"toolName,omitempty"`
	Delta            string `json:"delta,omitempty"`
	ProviderExecuted bool   `json:"providerExecuted,omitempty"`
	Dynamic          bool   `json:"dynamic,omitempty"`
}

// Kinds of buffered tool input parts.
const (
	inputStart = "start"
	inputDelta = "delta"
	inputEnd   = "end"
)

// newTranslator returns the state of a fresh turn.
func newTranslator(builtinToolNames []string, hostToolNames []string) *translator {
	return &translator{
		BuiltinToolNames: slices.Clone(builtinToolNames),
		HostToolNames:    slices.Clone(hostToolNames),
	}
}

// translate translates one event into zero or more stream parts. It does
// not emit the turn's stream start or finish, which the session owns.
func (t *translator) translate(event sessionEvent) []harness.StreamPart {
	switch event.Type {
	case "turn_start", "message_start":
		return t.start(event)
	case "message_update":
		return t.update(event)
	case "message_end", "turn_end":
		return t.end(event)
	case "tool_execution_start":
		return t.toolStart(event)
	case "tool_execution_end":
		return t.toolEnd(event)
	case "compaction_end":
		return t.compaction(event)
	default:
		return nil
	}
}

// start opens the turn and, for an assistant message, a step.
func (t *translator) start(event sessionEvent) []harness.StreamPart {
	if event.Type == "message_start" && (event.Message == nil || event.Message.Role != "assistant") {
		return nil
	}

	t.PromptStarted = true

	if event.Type == "message_start" {
		t.StepOpen = true
		t.PendingStepToolCallIDs = nil
		t.EarlyFinishStepToolCallIDs = nil
		t.StepUsage = nil
		t.FirstStreamedToolCallID = ""
		t.BufferedToolInputs = nil
	}

	t.StreamedAssistantText = ""
	t.CurrentTextID = ""
	t.CurrentReasoningID = ""
	t.ReasoningStarted = false

	// Content-block indices restart with every assistant message.
	t.StreamingToolInputIDs = nil

	return nil
}

// update translates text, reasoning and tool input deltas.
func (t *translator) update(event sessionEvent) []harness.StreamPart {
	update := event.AssistantMessageEvent
	if !t.PromptStarted || update == nil {
		return nil
	}

	switch update.Type {
	case "text_delta":
		return t.textDelta(update.Delta)
	case "thinking_delta":
		return t.reasoningDelta(update.Delta)
	case "toolcall_start":
		return t.toolInputStart(*update)
	case "toolcall_delta", "toolcall_end":
		return t.toolInputUpdate(*update)
	default:
		return nil
	}
}

// textDelta appends text, closing an open reasoning block first.
func (t *translator) textDelta(delta string) []harness.StreamPart {
	var parts []harness.StreamPart

	if t.ReasoningStarted && t.CurrentReasoningID != "" {
		parts = append(parts, harness.ReasoningEndPart{
			ID: t.CurrentReasoningID,
		})
		t.ReasoningStarted = false
		t.CurrentReasoningID = ""
	}

	if t.CurrentTextID == "" {
		t.CurrentTextID = newPartID()
		parts = append(parts, harness.TextStartPart{
			ID: t.CurrentTextID,
		})
	}

	t.StreamedAssistantText += delta

	return append(parts, harness.TextDeltaPart{
		ID:    t.CurrentTextID,
		Delta: delta,
	})
}

// reasoningDelta appends reasoning, closing an open text block first.
func (t *translator) reasoningDelta(delta string) []harness.StreamPart {
	var parts []harness.StreamPart

	if t.CurrentTextID != "" {
		parts = append(parts, harness.TextEndPart{
			ID: t.CurrentTextID,
		})
		t.CurrentTextID = ""
	}

	if t.CurrentReasoningID == "" {
		t.CurrentReasoningID = newPartID()
	}

	if !t.ReasoningStarted {
		t.ReasoningStarted = true
		parts = append(parts, harness.ReasoningStartPart{
			ID: t.CurrentReasoningID,
		})
	}

	return append(parts, harness.ReasoningDeltaPart{
		ID:    t.CurrentReasoningID,
		Delta: delta,
	})
}

// toolInputStart opens the streamed input of a tool call.
func (t *translator) toolInputStart(update assistantMessageEvent) []harness.StreamPart {
	if update.ContentIndex == nil || update.ID == "" || update.ToolName == "" {
		return nil
	}

	kind := t.kindOf(update.ToolName)

	if t.StreamingToolInputIDs == nil {
		t.StreamingToolInputIDs = make(map[int]string)
	}

	t.StreamingToolInputIDs[*update.ContentIndex] = update.ID

	if t.FirstStreamedToolCallID == "" {
		t.FirstStreamedToolCallID = update.ID
	}

	return t.bufferToolInput(update.ID, bufferedToolInput{
		Kind:             inputStart,
		ToolName:         t.publicName(update.ToolName, kind),
		ProviderExecuted: kind != toolHost,
		Dynamic:          kind.isDynamic(),
	})
}

// toolInputUpdate appends to or closes the streamed input of a tool call.
// Without a start there is no call to attach the input to; the complete
// input still arrives with the tool call.
func (t *translator) toolInputUpdate(update assistantMessageEvent) []harness.StreamPart {
	if update.ContentIndex == nil {
		return nil
	}

	id, ok := t.StreamingToolInputIDs[*update.ContentIndex]
	if !ok {
		return nil
	}

	if update.Type == "toolcall_end" {
		delete(t.StreamingToolInputIDs, *update.ContentIndex)

		return t.bufferToolInput(id, bufferedToolInput{
			Kind: inputEnd,
		})
	}

	return t.bufferToolInput(id, bufferedToolInput{
		Kind:  inputDelta,
		Delta: update.Delta,
	})
}

// bufferToolInput emits the input parts of the first streamed call and
// holds the others until Pi starts their call: in a sequential batch, a call
// that has not started must not surface before the turn pauses.
func (t *translator) bufferToolInput(id string, input bufferedToolInput) []harness.StreamPart {
	if id == t.FirstStreamedToolCallID {
		return []harness.StreamPart{
			toolInputPart(id, input),
		}
	}

	if t.BufferedToolInputs == nil {
		t.BufferedToolInputs = make(map[string][]bufferedToolInput)
	}

	t.BufferedToolInputs[id] = append(t.BufferedToolInputs[id], input)

	return nil
}

// toolInputPart converts a buffered input part.
func toolInputPart(id string, input bufferedToolInput) harness.StreamPart {
	switch input.Kind {
	case inputStart:
		return harness.ToolInputStartPart{
			ToolInputStart: model.ToolInputStart{
				ID:               id,
				ToolName:         input.ToolName,
				ProviderExecuted: input.ProviderExecuted,
				Dynamic:          input.Dynamic,
			},
		}
	case inputDelta:
		return harness.ToolInputDeltaPart{
			ToolInputDelta: model.ToolInputDelta{
				ID:    id,
				Delta: input.Delta,
			},
		}
	default:
		return harness.ToolInputEndPart{
			ToolInputEnd: model.ToolInputEnd{
				ID: id,
			},
		}
	}
}

// end closes the open blocks of a message, records its usage and tool
// calls, and closes the step at the end of the turn.
func (t *translator) end(event sessionEvent) []harness.StreamPart {
	if !t.PromptStarted {
		return nil
	}

	var parts []harness.StreamPart

	fullText := assistantText(event.Message)
	missingText := t.CurrentTextID != "" &&
		strings.HasPrefix(fullText, t.StreamedAssistantText) &&
		len(fullText) > len(t.StreamedAssistantText)

	if missingText {
		parts = append(parts, harness.TextDeltaPart{
			ID:    t.CurrentTextID,
			Delta: fullText[len(t.StreamedAssistantText):],
		})
		t.StreamedAssistantText = fullText
	}

	if t.CurrentTextID != "" {
		parts = append(parts, harness.TextEndPart{
			ID: t.CurrentTextID,
		})
		t.CurrentTextID = ""
	}

	if t.ReasoningStarted && t.CurrentReasoningID != "" {
		parts = append(parts, harness.ReasoningEndPart{
			ID: t.CurrentReasoningID,
		})
		t.ReasoningStarted = false
		t.CurrentReasoningID = ""
	}

	if event.Type == "turn_end" {
		t.PendingStepToolCallIDs = nil
		if terminalError(event) == "" {
			parts = append(parts, t.finishStep()...)
		}

		return parts
	}

	t.recordAssistantMessage(event)

	return parts
}

// recordAssistantMessage records the error, usage and tool calls of an
// assistant message.
func (t *translator) recordAssistantMessage(event sessionEvent) {
	if event.Message == nil || event.Message.Role != "assistant" {
		return
	}

	t.LastAssistantError = terminalError(event)

	if usage := event.Message.Usage; usage != nil {
		t.StepUsage = usage

		if usage.Reasoning != nil {
			total := *usage.Reasoning
			if t.TurnReasoningTokens != nil {
				total += *t.TurnReasoningTokens
			}

			t.TurnReasoningTokens = &total
		}
	}

	for _, id := range toolCallIDs(event.Message) {
		if !slices.Contains(t.PendingStepToolCallIDs, id) {
			t.PendingStepToolCallIDs = append(t.PendingStepToolCallIDs, id)
		}
	}
}

// toolStart emits a tool call with its buffered input.
func (t *translator) toolStart(event sessionEvent) []harness.StreamPart {
	if event.ToolCallID == "" || event.ToolName == "" {
		return nil
	}

	kind := t.kindOf(event.ToolName)
	name := t.publicName(event.ToolName, kind)

	if t.ObservedToolNames == nil {
		t.ObservedToolNames = make(map[string]string)
	}

	t.ObservedToolNames[event.ToolCallID] = name

	if kind.isDynamic() {
		if t.DynamicToolCalls == nil {
			t.DynamicToolCalls = make(map[string]toolKind)
		}

		t.DynamicToolCalls[event.ToolCallID] = kind
	}

	var parts []harness.StreamPart

	for _, input := range t.BufferedToolInputs[event.ToolCallID] {
		parts = append(parts, toolInputPart(event.ToolCallID, input))
	}

	delete(t.BufferedToolInputs, event.ToolCallID)

	call := harness.ToolCallPart{
		ToolCall: model.ToolCall{
			ToolCallID:       event.ToolCallID,
			ToolName:         name,
			Input:            toolInput(event.Args),
			ProviderExecuted: kind != toolHost,
			Dynamic:          kind.isDynamic(),
		},
	}

	if name != event.ToolName {
		call.NativeName = event.ToolName
	}

	return append(parts, call)
}

// toolInput returns a call's JSON input, an empty object when Pi reported
// none.
func toolInput(args json.RawMessage) string {
	if len(args) == 0 || string(args) == "null" {
		return "{}"
	}

	return string(args)
}

// toolEnd emits a tool result and closes the step when it was the last
// unsettled call.
func (t *translator) toolEnd(event sessionEvent) []harness.StreamPart {
	if event.ToolCallID == "" {
		return nil
	}

	name, ok := t.ObservedToolNames[event.ToolCallID]
	if !ok && event.ToolName != "" {
		name = t.publicName(event.ToolName, t.kindOf(event.ToolName))
	}

	if name == "" {
		return nil
	}

	dynamicKind := t.DynamicToolCalls[event.ToolCallID]
	delete(t.DynamicToolCalls, event.ToolCallID)

	result, isHostResult := t.HostToolResults[event.ToolCallID]

	switch {
	case isHostResult:
		delete(t.HostToolResults, event.ToolCallID)
	case dynamicKind == toolMCP:
		result = parseMCPToolResult(unwrapToolResult(event.Result))
	default:
		result = unwrapToolResult(event.Result)
	}

	t.PendingStepToolCallIDs = slices.DeleteFunc(t.PendingStepToolCallIDs, func(id string) bool {
		return id == event.ToolCallID
	})

	parts := []harness.StreamPart{
		harness.ToolResultPart{
			ToolResult: model.ToolResult{
				ToolCallID: event.ToolCallID,
				ToolName:   name,
				Result:     result,
				IsError:    event.IsError,
				Dynamic:    dynamicKind != "",
			},
		},
	}

	return append(parts, t.finishStep()...)
}

// compaction reports a compaction Pi completed. Aborted and result-less
// compactions did nothing. Pi checks the threshold after the turn's last
// assistant message, so the step is often finished already and the
// compaction gets its own inferred step.
func (t *translator) compaction(event sessionEvent) []harness.StreamPart {
	if event.Aborted {
		return nil
	}

	var result struct {
		Summary      *string `json:"summary"`
		TokensBefore *int    `json:"tokensBefore"`
	}

	if err := json.Unmarshal(event.Result, &result); err != nil || !jsonvalue.IsObject(event.Result) {
		return nil
	}

	summary := noSummary
	if result.Summary != nil {
		summary = *result.Summary
	}

	trigger := harness.CompactionAuto
	if event.Reason == "manual" {
		trigger = harness.CompactionManual
	}

	parts := []harness.StreamPart{
		harness.CompactionPart{
			Trigger:      trigger,
			Summary:      summary,
			TokensBefore: result.TokensBefore,
		},
	}

	if t.StepOpen {
		return parts
	}

	return append(parts, inferredFinishStep(messageUsage{}))
}

// finishStep closes the open step once none of its calls is unsettled.
func (t *translator) finishStep() []harness.StreamPart {
	if !t.StepOpen || len(t.PendingStepToolCallIDs) > 0 {
		return nil
	}

	t.StepOpen = false
	t.PendingStepToolCallIDs = nil
	t.EarlyFinishStepToolCallIDs = nil

	usage := messageUsage{}
	if t.StepUsage != nil {
		usage = *t.StepUsage
	}

	t.StepUsage = nil

	return []harness.StreamPart{
		inferredFinishStep(usage),
	}
}

// earlyFinishStep ports upstream takePiEarlyFinishStep: a finish step for a
// tool batch that cannot progress without host input, while the step stays
// open. In a parallel batch every unsettled call must wait on host input; a
// call already covered by an earlier early finish step does not trigger
// another.
func (t *translator) earlyFinishStep(waiting func(toolCallID string) bool) []harness.StreamPart {
	if !t.StepOpen || len(t.PendingStepToolCallIDs) == 0 {
		return nil
	}

	for _, id := range t.PendingStepToolCallIDs {
		if !waiting(id) {
			return nil
		}
	}

	covered := true
	for _, id := range t.PendingStepToolCallIDs {
		if !slices.Contains(t.EarlyFinishStepToolCallIDs, id) {
			covered = false
			t.EarlyFinishStepToolCallIDs = append(t.EarlyFinishStepToolCallIDs, id)
		}
	}

	if covered {
		return nil
	}

	usage := messageUsage{}
	if t.StepUsage != nil {
		usage = *t.StepUsage
	}

	return []harness.StreamPart{
		harness.FinishStepPart{
			FinishReason: model.FinishReason{
				Unified: model.FinishToolCalls,
				Raw:     "host-input",
			},
			Usage:           harnessUsage(usage),
			HarnessMetadata: inferredStepMetadata(),
		},
	}
}

// recordHostToolResult keeps the output the agent submitted for a host tool.
func (t *translator) recordHostToolResult(toolCallID string, output json.RawMessage) {
	if t.HostToolResults == nil {
		t.HostToolResults = make(map[string]json.RawMessage)
	}

	t.HostToolResults[toolCallID] = output
}

// kindOf returns how a native tool name is dispatched. Pi's native MCP names
// every server tool mcp__<server>__<tool>.
func (t *translator) kindOf(nativeName string) toolKind {
	switch {
	case slices.Contains(t.HostToolNames, nativeName):
		return toolHost
	case slices.Contains(t.BuiltinToolNames, nativeName):
		return toolBuiltin
	case strings.HasPrefix(nativeName, "mcp__"):
		return toolMCP
	default:
		return toolExtension
	}
}

// publicName returns the name a tool call is reported under: the common
// name of a built-in tool, and the native name otherwise.
func (t *translator) publicName(nativeName string, kind toolKind) string {
	if kind == toolBuiltin {
		return commonName(nativeName)
	}

	return nativeName
}

// inferredFinishStep returns a finish step Pi did not report explicitly.
func inferredFinishStep(usage messageUsage) harness.FinishStepPart {
	return harness.FinishStepPart{
		FinishReason: model.FinishReason{
			Unified: model.FinishStop,
			Raw:     "stop",
		},
		Usage:           harnessUsage(usage),
		HarnessMetadata: inferredStepMetadata(),
	}
}

// inferredStepMetadata marks inferred step boundaries.
func inferredStepMetadata() harness.Metadata {
	return harness.Metadata{
		HarnessID: map[string]any{
			"inferredStep": true,
		},
	}
}

// harnessUsage converts Pi usage. Pi's input excludes cached tokens.
func harnessUsage(usage messageUsage) model.Usage {
	converted := model.Usage{
		InputTokens: model.InputTokens{
			Total:      new(usage.Input + usage.CacheRead + usage.CacheWrite),
			NoCache:    new(usage.Input),
			CacheRead:  new(usage.CacheRead),
			CacheWrite: new(usage.CacheWrite),
		},
		OutputTokens: model.OutputTokens{
			Total: new(usage.Output),
		},
	}

	if usage.Reasoning != nil {
		converted.OutputTokens.Text = new(usage.Output - *usage.Reasoning)
		converted.OutputTokens.Reasoning = new(*usage.Reasoning)
	}

	return converted
}

// unwrapToolResult ports upstream unwrapPiToolResult. A text-only result
// envelope without details becomes its joined text; envelopes with details,
// no content, extra fields or non-text content are kept.
func unwrapToolResult(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("null")
	}

	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return raw
	}

	var content []contentBlock
	if err := json.Unmarshal(envelope["content"], &content); err != nil {
		return raw
	}

	if !isTextOnlyEnvelope(envelope, content) {
		return raw
	}

	var text strings.Builder
	for _, block := range content {
		text.WriteString(block.Text)
	}

	encoded, err := json.Marshal(text.String())
	if err != nil {
		return raw
	}

	return encoded
}

// isTextOnlyEnvelope reports an envelope with only text content and no
// details.
func isTextOnlyEnvelope(envelope map[string]json.RawMessage, content []contentBlock) bool {
	if len(content) == 0 {
		return false
	}

	if details, ok := envelope["details"]; ok && string(details) != "null" {
		return false
	}

	for key := range envelope {
		if key != "content" && key != "details" {
			return false
		}
	}

	for _, block := range content {
		if block.Type != "text" {
			return false
		}
	}

	return true
}

// parseMCPToolResult decodes the JSON text of an MCP tool result, and keeps
// the text when it is not JSON.
func parseMCPToolResult(result json.RawMessage) json.RawMessage {
	var text string
	if err := json.Unmarshal(result, &text); err != nil {
		return result
	}

	if json.Valid([]byte(text)) {
		return json.RawMessage(text)
	}

	return result
}

// newPartID returns a random stream part ID.
func newPartID() string {
	random := make([]byte, 8)
	_, _ = rand.Read(random)

	return hex.EncodeToString(random)
}
