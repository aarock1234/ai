package tui

import (
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/ai/ui"
)

// cursorBlinkInterval is the blink interval of the input cursor.
const cursorBlinkInterval = 500 * time.Millisecond

// deniedReason is the reason recorded when the user denies a tool call.
const deniedReason = "Denied by user."

const (
	activeControls = "↑/↓ · PgUp/PgDn · Esc/Ctrl+C"

	inputStatus                 = "Type a prompt and press Enter · " + activeControls
	processingStatus            = "Processing input... " + activeControls
	processingToolResultsStatus = "Processing tool results... " + activeControls
	streamingStatus             = "Streaming... " + activeControls
	executingToolsStatus        = "Executing tools... " + activeControls
	doneStatus                  = "Done · Enter another prompt · " + activeControls
)

// sessionMode is what the UI waits for.
type sessionMode int

const (
	modeInput sessionMode = iota
	modeStreaming
	modeApproval
)

// cursorBlinkMsg toggles the input cursor. Generation discards ticks of
// earlier blink cycles.
type cursorBlinkMsg struct {
	generation int
}

// agentModel is the Bubble Tea model of an agent session. Bubble Tea calls
// its methods on one goroutine, which owns the conversation.
type agentModel struct {
	settings     displaySettings
	runner       *turnRunner
	conversation conversation
	transcript   *transcript
	sections     sectionRenderer

	mode   sessionMode
	status string

	input           []rune
	cursorVisible   bool
	cursorBlinkTurn int

	width        int
	height       int
	scrollOffset int

	events                <-chan tea.Msg
	hasPendingToolResults bool
	turnShowedError       bool

	pendingApprovals  []approvalRequest
	approvalDecisions []approvalDecision

	statistics responseStatistics
}

// agentModelOptions configure newAgentModel.
type agentModelOptions struct {
	settings     displaySettings
	runner       *turnRunner
	conversation conversation
	palette      palette
}

func newAgentModel(options agentModelOptions) *agentModel {
	return &agentModel{
		settings:      options.settings,
		runner:        options.runner,
		conversation:  options.conversation,
		transcript:    newTranscript(options.settings),
		sections:      newSectionRenderer(options.palette),
		mode:          modeInput,
		status:        inputStatus,
		cursorVisible: true,
		width:         80,
		height:        24,
		statistics: responseStatistics{
			byResponse: map[int]stepStatistics{},
		},
	}
}

// Init starts blinking the input cursor.
func (m *agentModel) Init() tea.Cmd {
	return m.blinkCursor()
}

// Update handles one message.
func (m *agentModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch typed := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = typed.Width
		m.height = typed.Height
		m.scrollOffset = m.clampScrollOffset(m.scrollOffset)

		return m, nil
	case tea.KeyMsg:
		return m, m.handleKey(typed)
	case cursorBlinkMsg:
		return m, m.handleCursorBlink(typed)
	case streamPartMsg:
		m.handleStreamPart(typed.part)

		return m, waitForTurnEvent(m.events)
	case chunkMsg:
		m.handleChunk(typed.chunk)

		return m, waitForTurnEvent(m.events)
	case turnFinishedMsg:
		return m, m.handleTurnFinished(typed)
	}

	return m, nil
}

// View renders the screen.
func (m *agentModel) View() string {
	return renderScreen(screen{
		width:         m.width,
		height:        m.height,
		title:         m.settings.title,
		rightTitle:    formatTokenCount(m.statistics.totalTokens, m.settings.contextSize),
		bodyLines:     m.visibleBodyLines(),
		input:         string(m.input),
		inputActive:   m.mode == modeInput,
		cursorVisible: m.cursorVisible,
		status:        m.status,
	})
}

func (m *agentModel) handleKey(key tea.KeyMsg) tea.Cmd {
	switch key.Type {
	case tea.KeyEsc, tea.KeyCtrlC:
		m.runner.stop()

		return tea.Quit
	case tea.KeyUp:
		m.scroll(1)
	case tea.KeyDown:
		m.scroll(-1)
	case tea.KeyPgUp:
		m.scroll(bodyContentHeight(m.height))
	case tea.KeyPgDown:
		m.scroll(-bodyContentHeight(m.height))
	case tea.KeyCtrlL:
		return tea.ClearScreen
	}

	switch m.mode {
	case modeInput:
		return m.handleInputKey(key)
	case modeApproval:
		return m.handleApprovalKey(key)
	default:
		return nil
	}
}

func (m *agentModel) handleInputKey(key tea.KeyMsg) tea.Cmd {
	switch key.Type {
	case tea.KeyRunes, tea.KeySpace:
		m.input = append(m.input, key.Runes...)
		m.cursorVisible = true
	case tea.KeyBackspace:
		if len(m.input) > 0 {
			m.input = m.input[:len(m.input)-1]
		}
		m.cursorVisible = true
	case tea.KeyEnter:
		return m.submitPrompt()
	}

	return nil
}

func (m *agentModel) submitPrompt() tea.Cmd {
	prompt := string(m.input)
	m.input = nil

	m.changeBody(func() {
		m.transcript.addUser(prompt)
		m.transcript.beginResponse()
	})

	m.conversation.addPrompt(prompt)

	return m.startTurn()
}

// startTurn streams the agent over the conversation.
func (m *agentModel) startTurn() tea.Cmd {
	m.mode = modeStreaming
	m.status = processingStatus
	m.hasPendingToolResults = false
	m.turnShowedError = false
	m.statistics.reset(m.transcript.response)
	m.events = m.runner.start(m.conversation.nextTurn())

	return waitForTurnEvent(m.events)
}

func (m *agentModel) handleStreamPart(part ai.StreamPart) {
	m.updateStatus(part)

	switch typed := part.(type) {
	case ai.ErrorPart:
		m.turnShowedError = true
	case ai.ToolApprovalRequest:
		if !typed.IsAutomatic {
			m.pendingApprovals = append(m.pendingApprovals, localApprovalRequest(typed))
		}
	case ai.FinishStepPart:
		m.statistics.recordStep(typed)
	case ai.FinishPart:
		m.statistics.recordFinish(m.transcript.response, typed)
	}

	m.changeBody(func() {
		m.transcript.apply(part)
	})
}

// updateStatus shows what the agent is doing.
func (m *agentModel) updateStatus(part ai.StreamPart) {
	switch typed := part.(type) {
	case ai.StartStepPart:
		m.status = processingStatus
		if m.hasPendingToolResults {
			m.status = processingToolResultsStatus
		}
		m.hasPendingToolResults = false
	case ai.ToolResult, ai.ToolError, ai.ToolOutputDeniedPart:
		m.status = processingToolResultsStatus
		m.hasPendingToolResults = true
	case ai.ToolCall:
		m.status = executingToolsStatus
	case ai.ToolApprovalResponse:
		if typed.Approved {
			m.status = executingToolsStatus
		}
	case ai.TextStartPart, ai.TextDeltaPart, ai.ReasoningStartPart, ai.ReasoningDeltaPart, ai.ToolInputStartPart, ai.ToolInputDeltaPart:
		m.status = streamingStatus
	}
}

// handleChunk shows one chunk of a remote agent's stream.
func (m *agentModel) handleChunk(chunk ui.Chunk) {
	m.updateChunkStatus(chunk)

	if _, isError := chunk.(ui.ErrorChunk); isError {
		m.turnShowedError = true
	}

	m.changeBody(func() {
		m.transcript.applyChunk(chunk)
	})
}

// updateChunkStatus shows what a remote agent is doing.
func (m *agentModel) updateChunkStatus(chunk ui.Chunk) {
	switch typed := chunk.(type) {
	case ui.StartStepChunk:
		m.status = processingStatus
		if m.hasPendingToolResults {
			m.status = processingToolResultsStatus
		}
		m.hasPendingToolResults = false
	case ui.ToolOutputAvailableChunk, ui.ToolOutputErrorChunk, ui.ToolOutputDeniedChunk:
		m.status = processingToolResultsStatus
		m.hasPendingToolResults = true
	case ui.ToolInputAvailableChunk:
		m.status = executingToolsStatus
	case ui.ToolApprovalResponseChunk:
		if typed.Approved {
			m.status = executingToolsStatus
		}
	case ui.TextStartChunk, ui.TextDeltaChunk, ui.ReasoningStartChunk, ui.ReasoningDeltaChunk, ui.ToolInputStartChunk, ui.ToolInputDeltaChunk:
		m.status = streamingStatus
	}
}

func (m *agentModel) handleTurnFinished(finished turnFinishedMsg) tea.Cmd {
	m.events = nil
	m.pendingApprovals = append(m.pendingApprovals, m.conversation.finishTurn(finished)...)

	if finished.responseMessage != nil {
		m.statistics.recordMetadata(m.transcript.response, finished.responseMessage.Metadata)
	}

	if finished.err != nil {
		m.pendingApprovals = nil

		if !m.turnShowedError {
			m.changeBody(func() {
				m.transcript.addError(errorText(finished.err))
			})
		}
	}

	if len(m.pendingApprovals) > 0 {
		m.mode = modeApproval
		m.status = approvalStatus(m.pendingApprovals[0])

		return nil
	}

	m.mode = modeInput
	m.status = doneStatus

	return m.blinkCursor()
}

func (m *agentModel) handleApprovalKey(key tea.KeyMsg) tea.Cmd {
	if key.Type != tea.KeyRunes || len(key.Runes) != 1 {
		return nil
	}

	switch key.Runes[0] {
	case 'y', 'Y':
		return m.respondToApproval(true, "")
	case 'n', 'N':
		return m.respondToApproval(false, deniedReason)
	default:
		return nil
	}
}

// respondToApproval records the decision on the first pending approval and
// continues the turn once every approval is decided.
func (m *agentModel) respondToApproval(approved bool, reason string) tea.Cmd {
	request := m.pendingApprovals[0]
	m.pendingApprovals = m.pendingApprovals[1:]

	m.approvalDecisions = append(m.approvalDecisions, approvalDecision{
		request:  request,
		approved: approved,
		reason:   reason,
	})

	m.changeBody(func() {
		m.transcript.recordApproval(request.toolCallID, approved, reason)
	})

	if len(m.pendingApprovals) > 0 {
		m.status = approvalStatus(m.pendingApprovals[0])

		return nil
	}

	m.conversation.addApprovalDecisions(m.approvalDecisions)
	m.approvalDecisions = nil

	return m.startTurn()
}

func approvalStatus(request approvalRequest) string {
	return "Approve tool " + request.displayName() + "? y/n · " + activeControls
}

// blinkCursor starts a new blink cycle of the input cursor.
func (m *agentModel) blinkCursor() tea.Cmd {
	m.cursorBlinkTurn++
	m.cursorVisible = true
	generation := m.cursorBlinkTurn

	return tea.Tick(cursorBlinkInterval, func(time.Time) tea.Msg {
		return cursorBlinkMsg{
			generation: generation,
		}
	})
}

func (m *agentModel) handleCursorBlink(blink cursorBlinkMsg) tea.Cmd {
	if blink.generation != m.cursorBlinkTurn || m.mode != modeInput {
		return nil
	}

	m.cursorVisible = !m.cursorVisible
	generation := m.cursorBlinkTurn

	return tea.Tick(cursorBlinkInterval, func(time.Time) tea.Msg {
		return cursorBlinkMsg{
			generation: generation,
		}
	})
}

// changeBody applies a change to the transcript. When the view is
// scrolled up, the offset grows with the transcript, so that the visible
// lines stay in place.
func (m *agentModel) changeBody(change func()) {
	previousLineCount := len(m.bodyLines())

	change()

	if m.scrollOffset != 0 {
		m.scrollOffset = m.clampScrollOffset(m.scrollOffset + len(m.bodyLines()) - previousLineCount)
	}
}

func (m *agentModel) scroll(lines int) {
	m.scrollOffset = m.clampScrollOffset(m.scrollOffset + lines)
}

func (m *agentModel) clampScrollOffset(offset int) int {
	maxOffset := max(0, len(m.bodyLines())-bodyContentHeight(m.height))

	return min(max(0, offset), maxOffset)
}

// visibleBodyLines returns the transcript lines in view.
func (m *agentModel) visibleBodyLines() []string {
	lines := m.bodyLines()
	height := bodyContentHeight(m.height)

	start := max(0, len(lines)-height-m.scrollOffset)
	end := min(len(lines), start+height)

	return lines[start:end]
}

// bodyLines returns every line of the transcript.
func (m *agentModel) bodyLines() []string {
	if len(m.transcript.sections) == 0 {
		return []string{
			"Waiting for input...",
		}
	}

	width := transcriptWidth(m.width)
	var lines []string

	for index, current := range m.transcript.sections {
		rightTitle := current.rightTitle
		if current.isStatistic {
			rightTitle = m.statistics.format(current.response, m.settings.responseStatistics)
		}

		lines = append(lines, m.sections.lines(current, rightTitle, m.transcript.isCollapsed(index), width)...)
	}

	return lines
}
