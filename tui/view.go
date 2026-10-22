package tui

import (
	"io"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

const (
	minimumScreenWidth  = 20
	minimumScreenHeight = 8

	// chromeHeight is the height of everything around the transcript: its
	// borders and the three lines of the input box.
	chromeHeight = 5
)

// palette holds the styles of the UI, created for the terminal output so
// that colors match its capabilities.
type palette struct {
	bold   lipgloss.Style
	italic lipgloss.Style
	dim    lipgloss.Style

	sections map[sectionKind]sectionStyle
}

// sectionStyle is the color and the horizontal border of a section kind.
type sectionStyle struct {
	color  lipgloss.Style
	border string
}

func newPalette(output io.Writer) palette {
	renderer := lipgloss.NewRenderer(output)

	colored := func(color string) lipgloss.Style {
		return renderer.NewStyle().Foreground(lipgloss.Color(color))
	}

	return palette{
		bold:   renderer.NewStyle().Bold(true),
		italic: renderer.NewStyle().Italic(true),
		dim:    renderer.NewStyle().Faint(true),
		sections: map[sectionKind]sectionStyle{
			sectionUser: {
				color:  colored("14"),
				border: "─",
			},
			sectionAssistant: {
				color:  colored("10"),
				border: "─",
			},
			sectionReasoning: {
				color:  colored("12"),
				border: "·",
			},
			sectionTool: {
				color:  colored("13"),
				border: "─",
			},
			sectionError: {
				color:  colored("9"),
				border: "─",
			},
		},
	}
}

// screen is everything one frame shows.
type screen struct {
	width  int
	height int

	title      string
	rightTitle string

	bodyLines []string

	input         string
	inputActive   bool
	cursorVisible bool
	status        string
}

// renderScreen lays out the transcript box above the input or status box.
func renderScreen(state screen) string {
	width := max(minimumScreenWidth, state.width)
	bodyHeight := bodyContentHeight(state.height)

	visibleBody := state.bodyLines[:min(len(state.bodyLines), bodyHeight)]

	lines := make([]string, 0, bodyHeight+chromeHeight)
	lines = append(lines, topBorder(width, state.title, state.rightTitle))

	for index := range bodyHeight {
		line := ""
		if index < len(visibleBody) {
			line = visibleBody[index]
		}

		lines = append(lines, boxLine(line, width))
	}

	lines = append(lines, bottomBorder(width))

	if state.inputActive {
		cursor := "█"
		if !state.cursorVisible {
			cursor = " "
		}

		lines = append(lines,
			topBorder(width, "Input", ""),
			boxLine("> "+sanitizeTerminalText(state.input, false)+cursor, width),
		)
	} else {
		lines = append(lines,
			topBorder(width, "Status", ""),
			boxLine(sanitizeTerminalText(state.status, false), width),
		)
	}

	lines = append(lines, bottomBorder(width))

	return strings.Join(lines, "\n")
}

// bodyContentHeight is the number of transcript lines that fit on screen.
func bodyContentHeight(height int) int {
	return max(1, max(minimumScreenHeight, height)-chromeHeight)
}

// transcriptWidth is the width of transcript sections inside the outer box.
func transcriptWidth(width int) int {
	return max(minimumScreenWidth, width) - 4
}

func topBorder(width int, title string, rightTitle string) string {
	contentWidth := max(0, width-2)

	label := ""
	if title != "" {
		label = truncate(" "+sanitizeTerminalText(title, false)+" ", contentWidth)
	}

	rightLabel := ""
	if rightTitle != "" {
		rightLabel = truncate(" "+sanitizeTerminalText(rightTitle, false)+" ", contentWidth-lipgloss.Width(label))
	}

	remaining := max(0, contentWidth-lipgloss.Width(label)-lipgloss.Width(rightLabel))

	return "┌" + label + strings.Repeat("─", remaining) + rightLabel + "┐"
}

func bottomBorder(width int) string {
	return "└" + strings.Repeat("─", width-2) + "┘"
}

func boxLine(line string, width int) string {
	contentWidth := width - 4
	visible := truncate(line, contentWidth)
	padding := strings.Repeat(" ", max(0, contentWidth-lipgloss.Width(visible)))

	return "│ " + visible + padding + " │"
}

// renderedSection caches the lines of a section for the inputs that
// produced them.
type renderedSection struct {
	valid      bool
	width      int
	kind       sectionKind
	title      string
	rightTitle string
	content    string
	collapsed  bool
	lines      []string
}

// sectionRenderer renders transcript sections as bordered cards.
type sectionRenderer struct {
	palette  palette
	markdown markdownRenderer
}

func newSectionRenderer(colors palette) sectionRenderer {
	return sectionRenderer{
		palette: colors,
		markdown: markdownRenderer{
			palette: colors,
		},
	}
}

// lines returns the lines of a section, reusing its cache when nothing
// changed.
func (r sectionRenderer) lines(current *section, rightTitle string, collapsed bool, width int) []string {
	cached := current.cache
	if cached.valid &&
		cached.width == width &&
		cached.kind == current.kind &&
		cached.title == current.title &&
		cached.rightTitle == rightTitle &&
		cached.content == current.content &&
		cached.collapsed == collapsed {
		return cached.lines
	}

	lines := r.render(current, rightTitle, collapsed, width)

	current.cache = renderedSection{
		valid:      true,
		width:      width,
		kind:       current.kind,
		title:      current.title,
		rightTitle: rightTitle,
		content:    current.content,
		collapsed:  collapsed,
		lines:      lines,
	}

	return lines
}

func (r sectionRenderer) render(current *section, rightTitle string, collapsed bool, width int) []string {
	style := r.palette.sections[current.kind]
	contentWidth := max(1, width-4)

	title := " " + sanitizeTerminalText(current.title, false) + " "

	rightLabel := ""
	if rightTitle != "" {
		rightLabel = " " + sanitizeTerminalText(rightTitle, false) + " "
	}

	borderWidth := max(0, width-2-lipgloss.Width(title)-lipgloss.Width(rightLabel))
	top := style.color.Render("╭" + title + strings.Repeat(style.border, borderWidth) + rightLabel + "╮")
	bottom := style.color.Render("╰" + strings.Repeat(style.border, max(0, width-2)) + "╯")

	if collapsed {
		return []string{
			top,
			bottom,
		}
	}

	content := r.palette.dim.Render("(streaming...)")
	if current.content != "" {
		content = r.markdown.render(current.content)
	}

	wrapped := lipgloss.NewStyle().Width(contentWidth).Render(content)
	lines := []string{
		top,
	}

	for line := range strings.SplitSeq(wrapped, "\n") {
		lines = append(lines, sectionLine(line, contentWidth, style.color))
	}

	return append(lines, bottom)
}

func sectionLine(line string, contentWidth int, color lipgloss.Style) string {
	visible := truncate(line, contentWidth)
	padding := strings.Repeat(" ", max(0, contentWidth-lipgloss.Width(visible)))
	border := color.Render("│")

	return border + " " + visible + padding + " " + border
}

// truncate cuts a line, which may contain styles, to a visible width.
func truncate(line string, width int) string {
	if width <= 0 {
		return ""
	}

	return lipgloss.NewStyle().MaxWidth(width).Render(line)
}
