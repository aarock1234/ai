// Package tui runs AI SDK agents in a full-screen terminal UI, a port of
// the Vercel AI SDK's @ai-sdk/tui. It runs a local ai.Agent, or a remote
// agent through a ui.ChatTransport. The UI streams text and reasoning,
// shows tool calls as cards, asks the user to approve tool calls that need
// approval, renders markdown, and keeps the transcript scrollable above a
// pinned prompt.
//
// Controls:
//
//	Enter          submit the prompt
//	y / n          approve or deny a tool call
//	Up / Down      scroll the transcript
//	PgUp / PgDn    scroll the transcript by a page
//	Ctrl+L         repaint
//	Esc / Ctrl+C   exit
//
// The package is a separate module, so that the SDK itself does not depend
// on a terminal UI library.
package tui

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/ai/ui"
)

// PartDisplayMode controls how reasoning and tool sections are shown.
type PartDisplayMode string

// Part display modes.
const (
	// DisplayFull shows the section header and its content.
	DisplayFull PartDisplayMode = "full"

	// DisplayCollapsed shows only the section header.
	DisplayCollapsed PartDisplayMode = "collapsed"

	// DisplayAutoCollapsed shows the content until a later section of the
	// same response appears, then collapses the section.
	DisplayAutoCollapsed PartDisplayMode = "auto-collapsed"

	// DisplayHidden omits the section.
	DisplayHidden PartDisplayMode = "hidden"
)

// ResponseStatisticsMode selects the statistic shown in the header of
// assistant sections.
type ResponseStatisticsMode string

// Response statistics modes.
const (
	// StatisticsOutputTokensPerSecond shows the output token throughput.
	StatisticsOutputTokensPerSecond ResponseStatisticsMode = "outputTokensPerSecond"

	// StatisticsOutputTokenCount shows the number of output tokens.
	StatisticsOutputTokenCount ResponseStatisticsMode = "outputTokenCount"
)

// RunAgentOptions configure RunAgent. Set exactly one of Agent and
// Transport.
//
// Go adaptations: upstream also forwards a sandbox session to the tools of
// a local agent, which package ai has no option for. Input and Output
// replace upstream's test-only terminal streams.
type RunAgentOptions struct {
	// Agent is a local agent to run.
	Agent ai.Agent

	// Transport connects to a remote agent, such as a chat endpoint served
	// with ui.PipeAgentStreamToResponse through ui.NewDefaultChatTransport.
	// Every turn sends the whole conversation as UI messages.
	Transport ui.ChatTransport

	// Title is shown in the header of the UI.
	Title string

	// Tools controls how tool calls are shown. Empty uses
	// DisplayAutoCollapsed.
	Tools PartDisplayMode

	// Reasoning controls how reasoning is shown. Empty uses
	// DisplayAutoCollapsed.
	Reasoning PartDisplayMode

	// ResponseStatistics selects the statistic of assistant sections. Empty
	// uses StatisticsOutputTokensPerSecond.
	ResponseStatistics ResponseStatisticsMode

	// ContextSize is the model's context window in tokens. When it is set,
	// the header shows the total token usage as a percentage of it.
	ContextSize int

	// Input is the terminal input. Nil uses standard input.
	Input io.Reader

	// Output is the terminal output. Nil uses standard output.
	Output io.Writer
}

// ErrNoAgent reports that RunAgent was called without an agent or a
// transport.
var ErrNoAgent = errors.New("tui: an agent or a transport is required")

// ErrAgentAndTransport reports that RunAgent was called with both an agent
// and a transport.
var ErrAgentAndTransport = errors.New("tui: set either an agent or a transport, not both")

// newConversation returns the conversation of the configured agent.
func newConversation(options RunAgentOptions) (conversation, error) {
	switch {
	case options.Agent != nil && options.Transport != nil:
		return nil, ErrAgentAndTransport
	case options.Agent != nil:
		return newLocalConversation(options.Agent), nil
	case options.Transport != nil:
		return newRemoteConversation(options.Transport), nil
	default:
		return nil, ErrNoAgent
	}
}

// RunAgent runs the agent in the terminal UI until the user exits or ctx is
// cancelled. Each prompt continues the conversation; tool calls that need
// approval pause the agent until the user approves or denies them.
func RunAgent(ctx context.Context, options RunAgentOptions) error {
	session, err := newConversation(options)
	if err != nil {
		return err
	}

	settings, err := newDisplaySettings(options)
	if err != nil {
		return err
	}

	input := options.Input
	if input == nil {
		input = os.Stdin
	}

	output := options.Output
	if output == nil {
		output = os.Stdout
	}

	runner := newTurnRunner(ctx)
	defer runner.wait()
	defer runner.stop()

	program := tea.NewProgram(
		newAgentModel(agentModelOptions{
			settings:     settings,
			runner:       runner,
			conversation: session,
			palette:      newPalette(output),
		}),
		tea.WithContext(ctx),
		tea.WithInput(input),
		tea.WithOutput(output),
		tea.WithAltScreen(),
	)

	if _, err := program.Run(); err != nil {
		if ctx.Err() != nil {
			return context.Cause(ctx)
		}

		return fmt.Errorf("run terminal ui: %w", err)
	}

	return nil
}

// displaySettings are the resolved display options of a session.
type displaySettings struct {
	title              string
	tools              PartDisplayMode
	reasoning          PartDisplayMode
	responseStatistics ResponseStatisticsMode
	contextSize        int
}

func newDisplaySettings(options RunAgentOptions) (displaySettings, error) {
	tools, err := resolveDisplayMode("Tools", options.Tools)
	if err != nil {
		return displaySettings{}, err
	}

	reasoning, err := resolveDisplayMode("Reasoning", options.Reasoning)
	if err != nil {
		return displaySettings{}, err
	}

	statistics := options.ResponseStatistics
	switch statistics {
	case "":
		statistics = StatisticsOutputTokensPerSecond
	case StatisticsOutputTokensPerSecond, StatisticsOutputTokenCount:
	default:
		return displaySettings{}, fmt.Errorf("tui: unknown ResponseStatistics mode %q", statistics)
	}

	return displaySettings{
		title:              options.Title,
		tools:              tools,
		reasoning:          reasoning,
		responseStatistics: statistics,
		contextSize:        options.ContextSize,
	}, nil
}

func resolveDisplayMode(option string, mode PartDisplayMode) (PartDisplayMode, error) {
	switch mode {
	case "":
		return DisplayAutoCollapsed, nil
	case DisplayFull, DisplayCollapsed, DisplayAutoCollapsed, DisplayHidden:
		return mode, nil
	default:
		return "", fmt.Errorf("tui: unknown %s display mode %q", option, mode)
	}
}
