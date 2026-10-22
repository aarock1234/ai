package tui

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// tableAlignment is the alignment of a markdown table column.
type tableAlignment int

const (
	alignLeft tableAlignment = iota
	alignCenter
	alignRight
)

// minimumTableColumnWidth is the narrowest rendered table column.
const minimumTableColumnWidth = 3

var (
	unorderedListItemPattern  = regexp.MustCompile(`^(\s*)[-+*]\s+(.*)$`)
	tableSeparatorPattern     = regexp.MustCompile(`^(:)?-{3,}(:)?$`)
	inlineCodePattern         = regexp.MustCompile("`([^`]+)`")
	strongAsteriskPattern     = regexp.MustCompile(`\*\*([^*\n]+)\*\*`)
	strongUnderscorePattern   = regexp.MustCompile(`__([^_\n]+)__`)
	emphasisAsteriskPattern   = regexp.MustCompile(`\*([^*\n]+)\*`)
	emphasisUnderscorePattern = regexp.MustCompile(`_([^_\n]+)_`)
)

// markdownRenderer renders the subset of markdown that the terminal shows:
// headings, lists, quotes, tables, bold and italic text, and inline code.
type markdownRenderer struct {
	palette palette
}

// render renders sanitized markdown as terminal text.
func (m markdownRenderer) render(input string) string {
	lines := strings.Split(sanitizeTerminalText(input, true), "\n")
	var output []string

	for index := 0; index < len(lines); index++ {
		if table, ok := parseTable(lines, index); ok {
			output = append(output, m.renderTable(table)...)
			index = table.endIndex - 1

			continue
		}

		output = append(output, m.renderLine(lines[index]))
	}

	return strings.Join(output, "\n")
}

func (m markdownRenderer) renderLine(line string) string {
	if heading, ok := strings.CutPrefix(line, "### "); ok {
		return m.renderInline("▶ " + heading)
	}

	if heading, ok := strings.CutPrefix(line, "## "); ok {
		return m.renderInline("■ " + heading)
	}

	if heading, ok := strings.CutPrefix(line, "# "); ok {
		return m.renderInline("█ " + heading)
	}

	if match := unorderedListItemPattern.FindStringSubmatch(line); match != nil {
		indentation, text := match[1], match[2]
		if text == "" {
			return m.renderInline(indentation + "•")
		}

		return m.renderInline(indentation + "• " + text)
	}

	if quote, ok := strings.CutPrefix(line, "> "); ok {
		return m.renderInline("│ " + quote)
	}

	return m.renderInline(line)
}

// renderInline removes inline code markers and styles bold and italic
// text.
func (m markdownRenderer) renderInline(input string) string {
	output := inlineCodePattern.ReplaceAllString(input, "$1")
	output = replaceStyled(output, strongAsteriskPattern, m.palette.bold)
	output = replaceStyled(output, strongUnderscorePattern, m.palette.bold)
	output = replaceStyled(output, emphasisAsteriskPattern, m.palette.italic)

	return replaceStyled(output, emphasisUnderscorePattern, m.palette.italic)
}

func replaceStyled(input string, pattern *regexp.Regexp, style lipgloss.Style) string {
	return pattern.ReplaceAllStringFunc(input, func(match string) string {
		return style.Render(pattern.FindStringSubmatch(match)[1])
	})
}

// markdownTable is a parsed markdown table.
type markdownTable struct {
	header     []string
	alignments []tableAlignment
	rows       [][]string

	// endIndex is the index of the first line after the table.
	endIndex int
}

// parseTable parses the table that starts at the line, if any: a header
// row of at least two cells, a separator row and any number of rows.
func parseTable(lines []string, startIndex int) (markdownTable, bool) {
	header, ok := parseTableCells(lines[startIndex])
	if !ok || len(header) < 2 || startIndex+1 >= len(lines) {
		return markdownTable{}, false
	}

	separator, ok := parseTableCells(lines[startIndex+1])
	if !ok || len(separator) != len(header) {
		return markdownTable{}, false
	}

	alignments, ok := parseTableAlignments(separator)
	if !ok {
		return markdownTable{}, false
	}

	table := markdownTable{
		header:     header,
		alignments: alignments,
		endIndex:   startIndex + 2,
	}

	for table.endIndex < len(lines) {
		row, ok := parseTableCells(lines[table.endIndex])
		if !ok {
			break
		}

		table.rows = append(table.rows, normalizeTableRow(row, len(header)))
		table.endIndex++
	}

	return table, true
}

// parseTableCells splits a table row into trimmed cells. Escaped pipes are
// part of a cell.
func parseTableCells(line string) ([]string, bool) {
	if !strings.Contains(line, "|") {
		return nil, false
	}

	row := strings.TrimSpace(line)
	row = strings.TrimPrefix(row, "|")
	if strings.HasSuffix(row, "|") && !strings.HasSuffix(row, `\|`) {
		row = strings.TrimSuffix(row, "|")
	}

	var cells []string
	var cell strings.Builder
	characters := []rune(row)

	for index := 0; index < len(characters); index++ {
		character := characters[index]

		if character == '\\' && index+1 < len(characters) && characters[index+1] == '|' {
			cell.WriteRune('|')
			index++

			continue
		}

		if character == '|' {
			cells = append(cells, strings.TrimSpace(cell.String()))
			cell.Reset()

			continue
		}

		cell.WriteRune(character)
	}

	return append(cells, strings.TrimSpace(cell.String())), true
}

func parseTableAlignments(cells []string) ([]tableAlignment, bool) {
	alignments := make([]tableAlignment, 0, len(cells))

	for _, cell := range cells {
		match := tableSeparatorPattern.FindStringSubmatch(cell)
		if match == nil {
			return nil, false
		}

		alignedLeft := match[1] != ""
		alignedRight := match[2] != ""

		switch {
		case alignedLeft && alignedRight:
			alignments = append(alignments, alignCenter)
		case alignedRight:
			alignments = append(alignments, alignRight)
		default:
			alignments = append(alignments, alignLeft)
		}
	}

	return alignments, true
}

func normalizeTableRow(row []string, length int) []string {
	normalized := make([]string, length)
	copy(normalized, row)

	return normalized
}

// renderTable renders a table with a bold header, a separator line and
// aligned columns.
func (m markdownRenderer) renderTable(table markdownTable) []string {
	header := make([]string, len(table.header))
	for index, cell := range table.header {
		header[index] = m.palette.bold.Render(m.renderInline(cell))
	}

	rows := make([][]string, len(table.rows))
	for index, row := range table.rows {
		rows[index] = make([]string, len(row))
		for column, cell := range row {
			rows[index][column] = m.renderInline(cell)
		}
	}

	widths := columnWidths(append([][]string{header}, rows...), len(table.alignments))

	separators := make([]string, len(widths))
	for index, width := range widths {
		separators[index] = strings.Repeat("─", width)
	}

	lines := []string{
		formatTableRow(header, widths, table.alignments),
		strings.Join(separators, "  "),
	}

	for _, row := range rows {
		lines = append(lines, formatTableRow(row, widths, table.alignments))
	}

	return lines
}

func columnWidths(rows [][]string, columns int) []int {
	widths := make([]int, columns)

	for column := range widths {
		widths[column] = minimumTableColumnWidth

		for _, row := range rows {
			widths[column] = max(widths[column], lipgloss.Width(row[column]))
		}
	}

	return widths
}

func formatTableRow(row []string, widths []int, alignments []tableAlignment) string {
	cells := make([]string, len(row))

	for index, cell := range row {
		cells[index] = alignTableCell(cell, widths[index], alignments[index])
	}

	return strings.Join(cells, "  ")
}

func alignTableCell(cell string, width int, alignment tableAlignment) string {
	padding := max(0, width-lipgloss.Width(cell))

	switch alignment {
	case alignRight:
		return strings.Repeat(" ", padding) + cell
	case alignCenter:
		left := padding / 2

		return strings.Repeat(" ", left) + cell + strings.Repeat(" ", padding-left)
	default:
		return cell + strings.Repeat(" ", padding)
	}
}

// sanitizeTerminalText escapes terminal control characters in untrusted
// text before the UI adds its own styles: each control character becomes
// a \uXXXX escape, which also neutralizes escape sequences split across
// stream chunks. Tabs become four spaces, and newlines are kept when
// multiline is set.
func sanitizeTerminalText(input string, multiline bool) string {
	var output strings.Builder

	for _, character := range input {
		switch {
		case character == '\n' && multiline:
			output.WriteRune('\n')
		case character == '\t':
			output.WriteString("    ")
		case isControlCharacter(character):
			fmt.Fprintf(&output, `\u%04x`, character)
		default:
			output.WriteRune(character)
		}
	}

	return output.String()
}

func isControlCharacter(character rune) bool {
	return character < 0x20 || (character >= 0x7f && character <= 0x9f)
}
