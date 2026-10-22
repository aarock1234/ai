package tui

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
)

// sectionKind is the kind of a transcript section, which selects its
// color.
type sectionKind int

const (
	sectionUser sectionKind = iota
	sectionAssistant
	sectionReasoning
	sectionTool
	sectionError
)

// section is one card of the transcript.
type section struct {
	kind       sectionKind
	title      string
	rightTitle string
	content    string

	// response groups the sections of one assistant response, including
	// its continuations after tool approvals. User sections have none.
	response int

	// displayMode is the display mode of reasoning and tool sections.
	displayMode PartDisplayMode

	// isStatistic marks assistant sections that show the response
	// statistic as their right title.
	isStatistic bool

	cache renderedSection
}

// isAssistantOutput reports whether the section is part of an assistant
// response, so that earlier auto-collapsed sections of the response
// collapse.
func (s *section) isAssistantOutput() bool {
	return s.response > 0
}

// toolState is the progress of a tool call.
type toolState int

const (
	toolInputStreaming toolState = iota
	toolInputAvailable
	toolApprovalRequested
	toolApprovalResponded
	toolOutputAvailable
	toolOutputError
	toolOutputDenied
)

// toolInvocation is the state of one tool call in the transcript.
type toolInvocation struct {
	name      string
	title     string
	state     toolState
	input     string
	hasInput  bool
	output    string
	errorText string
	approved  bool
	reason    string

	section *section
}

// transcript holds the sections of the conversation and turns stream parts
// into them.
type transcript struct {
	settings displaySettings
	sections []*section

	// response is the current response group.
	response int

	// step numbers the steps of the current turn, so that text and
	// reasoning blocks of different steps never share a key.
	step int

	blocks map[string]*streamedBlock
	tools  map[string]*toolInvocation

	// approvalTools maps the approval IDs of a remote stream to their tool
	// call IDs.
	approvalTools map[string]string
}

// streamedBlock is a text or reasoning block being streamed.
type streamedBlock struct {
	text    strings.Builder
	section *section
}

func newTranscript(settings displaySettings) *transcript {
	return &transcript{
		settings:      settings,
		blocks:        map[string]*streamedBlock{},
		tools:         map[string]*toolInvocation{},
		approvalTools: map[string]string{},
	}
}

// addUser adds a user prompt.
func (t *transcript) addUser(prompt string) {
	t.sections = append(t.sections, &section{
		kind:    sectionUser,
		title:   "User",
		content: prompt,
	})
}

// addError adds an error section. Like user sections, it belongs to no
// response.
func (t *transcript) addError(message string) {
	t.sections = append(t.sections, &section{
		kind:    sectionError,
		title:   "Error",
		content: message,
	})
}

// beginResponse starts the response group of a new prompt.
func (t *transcript) beginResponse() {
	t.response++
}

// apply renders one stream part into the transcript.
func (t *transcript) apply(part ai.StreamPart) {
	switch typed := part.(type) {
	case ai.StartStepPart:
		t.step++
	case ai.TextDeltaPart:
		t.appendBlock(sectionAssistant, typed.ID, typed.Text)
	case ai.ReasoningDeltaPart:
		t.appendBlock(sectionReasoning, typed.ID, typed.Text)
	case ai.ToolInputStartPart:
		t.startToolInput(typed)
	case ai.ToolInputDeltaPart:
		t.updateTool(typed.ID, func(tool *toolInvocation) {
			tool.input += typed.Delta
		})
	case ai.ToolCall:
		t.receiveToolCall(typed)
	case ai.ToolApprovalRequest:
		t.receiveToolCall(typed.ToolCall)
		t.updateTool(typed.ToolCall.ToolCallID, func(tool *toolInvocation) {
			tool.state = toolApprovalRequested
		})
	case ai.ToolApprovalResponse:
		t.recordApproval(typed.ToolCall.ToolCallID, typed.Approved, typed.Reason)
	case ai.ToolResult:
		t.updateTool(typed.ToolCallID, func(tool *toolInvocation) {
			tool.state = toolOutputAvailable
			tool.output = formatValue(typed.Output)
		})
	case ai.ToolError:
		t.updateTool(typed.ToolCallID, func(tool *toolInvocation) {
			tool.state = toolOutputError
			tool.errorText = errorText(typed.Error)
		})
	case ai.ToolOutputDeniedPart:
		t.updateTool(typed.ToolCallID, func(tool *toolInvocation) {
			tool.state = toolOutputDenied
		})
	case ai.ErrorPart:
		t.addError(errorText(typed.Err))
	}
}

// recordApproval records the user's decision on a tool call.
func (t *transcript) recordApproval(toolCallID string, approved bool, reason string) {
	t.updateTool(toolCallID, func(tool *toolInvocation) {
		tool.state = toolApprovalResponded
		tool.approved = approved
		tool.reason = reason
	})
}

// appendBlock appends streamed text to a text or reasoning block. Its
// section appears once the block has visible text.
func (t *transcript) appendBlock(kind sectionKind, id string, delta string) {
	if kind == sectionReasoning && t.settings.reasoning == DisplayHidden {
		return
	}

	key := fmt.Sprintf("%d:%d:%s", t.response, t.step, id)

	block, ok := t.blocks[key]
	if !ok {
		block = &streamedBlock{}
		t.blocks[key] = block
	}

	block.text.WriteString(delta)

	content := strings.TrimSpace(block.text.String())
	if content == "" {
		return
	}

	if block.section == nil {
		block.section = t.newBlockSection(kind)
		t.sections = append(t.sections, block.section)
	}

	block.section.content = content
}

func (t *transcript) newBlockSection(kind sectionKind) *section {
	if kind == sectionReasoning {
		return &section{
			kind:        sectionReasoning,
			title:       "Reasoning",
			response:    t.response,
			displayMode: t.settings.reasoning,
		}
	}

	return &section{
		kind:        sectionAssistant,
		title:       "Assistant",
		response:    t.response,
		displayMode: DisplayFull,
		isStatistic: true,
	}
}

func (t *transcript) startToolInput(part ai.ToolInputStartPart) {
	tool := t.ensureTool(part.ID, part.ToolName, part.Title)
	if tool == nil {
		return
	}

	tool.state = toolInputStreaming
	t.renderTool(tool)
}

func (t *transcript) receiveToolCall(call ai.ToolCall) {
	t.receiveToolInput(toolInput{
		toolCallID: call.ToolCallID,
		name:       call.ToolName,
		title:      call.Title,
		input:      call.Input,
	})
}

// toolInput is the complete input of a tool call.
type toolInput struct {
	toolCallID string
	name       string
	title      string
	input      json.RawMessage
}

// receiveToolInput shows the complete input of a tool call.
func (t *transcript) receiveToolInput(received toolInput) {
	tool := t.ensureTool(received.toolCallID, received.name, received.title)
	if tool == nil {
		return
	}

	if tool.state <= toolInputAvailable {
		tool.state = toolInputAvailable
	}

	tool.input = formatJSON(received.input)
	tool.hasInput = true
	t.renderTool(tool)
}

// ensureTool returns the invocation of a tool call, adding its section on
// first use. It returns nil when tools are hidden.
func (t *transcript) ensureTool(toolCallID string, name string, title string) *toolInvocation {
	if t.settings.tools == DisplayHidden {
		return nil
	}

	if tool, ok := t.tools[toolCallID]; ok {
		return tool
	}

	tool := &toolInvocation{
		name:  name,
		title: title,
		section: &section{
			kind:        sectionTool,
			response:    t.response,
			displayMode: t.settings.tools,
		},
	}

	t.tools[toolCallID] = tool
	t.sections = append(t.sections, tool.section)

	return tool
}

func (t *transcript) updateTool(toolCallID string, update func(tool *toolInvocation)) {
	tool, ok := t.tools[toolCallID]
	if !ok {
		return
	}

	update(tool)
	t.renderTool(tool)
}

// renderTool updates the section of a tool call from its state.
func (t *transcript) renderTool(tool *toolInvocation) {
	displayName := tool.name
	if tool.title != "" {
		displayName = tool.title
	}

	inputText := "Input: (streaming...)"
	if tool.hasInput || tool.input != "" {
		inputText = "Input:\n" + tool.input
	}

	target := tool.section
	target.kind = sectionTool
	target.title = "Tool · " + displayName
	target.rightTitle = toolStatus(tool)
	target.content = inputText

	switch tool.state {
	case toolOutputAvailable:
		target.content = inputText + "\n\nOutput:\n" + tool.output
	case toolOutputError:
		target.kind = sectionError
		target.title = "Tool Error · " + displayName
		target.content = inputText + "\n\nError:\n" + tool.errorText
	case toolOutputDenied:
		reason := tool.reason
		if reason == "" {
			reason = "denied"
		}

		target.kind = sectionError
		target.title = "Tool Denied · " + displayName
		target.content = inputText + "\n\nReason: " + reason
	}
}

func toolStatus(tool *toolInvocation) string {
	switch tool.state {
	case toolInputStreaming:
		return "waiting"
	case toolApprovalRequested:
		return "approval requested"
	case toolApprovalResponded:
		if tool.approved {
			return "executing"
		}

		return "denied"
	case toolOutputAvailable, toolOutputError:
		return "done"
	case toolOutputDenied:
		return "denied"
	default:
		return "executing"
	}
}

// isCollapsed reports whether a section shows only its header: always in
// collapsed mode, and in auto-collapsed mode once a later section of the
// same response exists.
func (t *transcript) isCollapsed(index int) bool {
	current := t.sections[index]

	switch current.displayMode {
	case DisplayCollapsed:
		return true
	case DisplayAutoCollapsed:
		return t.hasLaterResponseSection(index)
	default:
		return false
	}
}

func (t *transcript) hasLaterResponseSection(index int) bool {
	current := t.sections[index]

	for _, later := range t.sections[index+1:] {
		if later.isAssistantOutput() && later.response == current.response {
			return true
		}
	}

	return false
}

// formatValue formats a tool output: strings as they are and other values
// as indented JSON.
func formatValue(value any) string {
	if text, ok := value.(string); ok {
		return text
	}

	if document, ok := value.(json.RawMessage); ok {
		return formatJSON(document)
	}

	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Sprint(value)
	}

	return string(encoded)
}

// formatJSON indents a JSON document, returning invalid documents as they
// are.
func formatJSON(document json.RawMessage) string {
	var indented bytes.Buffer
	if err := json.Indent(&indented, document, "", "  "); err != nil {
		return string(document)
	}

	return indented.String()
}

func errorText(err error) string {
	if err == nil {
		return "unknown error"
	}

	return err.Error()
}
