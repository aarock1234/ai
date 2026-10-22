package sandboxshell

import (
	"bufio"
	"context"
	"crypto/md5"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// expandCountShorthand rewrites the obsolete "-N" and "+N" count forms of
// head and tail into "-n N" and "-n +N".
func expandCountShorthand(args []string) []string {
	expanded := make([]string, 0, len(args))

	for _, arg := range args {
		digits := strings.TrimLeft(arg, "-+")
		isShorthand := len(arg) > 1 && len(digits) == len(arg)-1 && isDecimal(digits)

		switch {
		case isShorthand && arg[0] == '-':
			expanded = append(expanded, "-n", digits)
		case isShorthand && arg[0] == '+':
			expanded = append(expanded, "-n", arg)
		default:
			expanded = append(expanded, arg)
		}
	}

	return expanded
}

// isDecimal reports whether text is a nonempty run of decimal digits.
func isDecimal(text string) bool {
	if text == "" {
		return false
	}

	for _, character := range text {
		if character < '0' || character > '9' {
			return false
		}
	}

	return true
}

// countOptions are the flags of head and tail.
type countOptions struct {
	count     int
	bytes     bool
	fromStart bool
	quiet     bool
	verbose   bool
}

// parseCountOptions reads the flags of head and tail.
func parseCountOptions(call *commandCall) (countOptions, []string, bool) {
	call.args = expandCountShorthand(call.args)

	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-n",
			"-c",
			"--lines",
			"--bytes",
		},
	})
	if !ok {
		return countOptions{}, nil, false
	}

	options := countOptions{
		count: 10,
	}

	for _, flag := range parsed.flags {
		switch flag.name {
		case "-n", "--lines", "-c", "--bytes":
			options.bytes = flag.name == "-c" || flag.name == "--bytes"
			options.fromStart = strings.HasPrefix(flag.value, "+")

			count, err := strconv.Atoi(strings.TrimLeft(flag.value, "+-"))
			if err != nil {
				call.report("invalid number of lines: '%s'", flag.value)

				return countOptions{}, nil, false
			}

			options.count = count
		case "-q", "--quiet", "--silent":
			options.quiet = true
		case "-v", "--verbose":
			options.verbose = true
		default:
			call.invalidFlag(flag)

			return countOptions{}, nil, false
		}
	}

	return options, parsed.operands, true
}

// runHead carries out "head [-n count] [-c count] [file...]", streaming the
// standard input.
func runHead(ctx context.Context, call *commandCall) int {
	options, operands, ok := parseCountOptions(call)
	if !ok {
		return exitFailure
	}

	if len(operands) == 0 {
		operands = []string{
			"-",
		}
	}

	showHeaders := options.verbose || len(operands) > 1 && !options.quiet

	ok = true
	for index, operand := range operands {
		reader, err := call.openOperand(operand)
		if err != nil {
			call.report("cannot open '%s' for reading: %s", operand, describeError(err))
			ok = false

			continue
		}

		if showHeaders {
			writeFileHeader(call, operand, index)
		}

		if err := copyHead(ctx, call.stdout, reader, options); err != nil {
			return exitFailure
		}
	}

	return statusOf(ok)
}

// copyHead copies the first lines or bytes of a reader.
func copyHead(ctx context.Context, writer io.Writer, reader io.Reader, options countOptions) error {
	if options.bytes {
		_, err := io.CopyN(writer, reader, int64(options.count))
		if errors.Is(err, io.EOF) {
			return nil
		}

		return err
	}

	buffered := bufio.NewReader(reader)
	for range options.count {
		if err := ctx.Err(); err != nil {
			return err
		}

		line, err := buffered.ReadString('\n')
		if _, writeErr := io.WriteString(writer, line); writeErr != nil {
			return writeErr
		}

		if errors.Is(err, io.EOF) {
			return nil
		}

		if err != nil {
			return err
		}
	}

	return nil
}

// writeFileHeader writes the "==> name <==" header of head and tail.
func writeFileHeader(call *commandCall, operand string, index int) {
	if index > 0 {
		call.writeOutput("\n")
	}

	name := operand
	if name == "-" {
		name = "standard input"
	}

	call.writeOutput("==> " + name + " <==\n")
}

// runTail carries out "tail [-n [+]count] [-c [+]count] [file...]".
func runTail(_ context.Context, call *commandCall) int {
	options, operands, ok := parseCountOptions(call)
	if !ok {
		return exitFailure
	}

	inputs, ok := call.readInputs(operands)
	showHeaders := options.verbose || len(inputs) > 1 && !options.quiet

	for index, input := range inputs {
		if showHeaders {
			writeFileHeader(call, input.name, index)
		}

		call.writeOutput(tailOf(string(input.contents), options))
	}

	return statusOf(ok)
}

// tailOf returns the selected end of a text: the last count lines or
// bytes, or everything from line or byte count on.
func tailOf(text string, options countOptions) string {
	if options.bytes {
		start := len(text) - options.count
		if options.fromStart {
			start = options.count - 1
		}

		return text[min(max(start, 0), len(text)):]
	}

	lines := strings.SplitAfter(text, "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	start := len(lines) - options.count
	if options.fromStart {
		start = options.count - 1
	}

	return strings.Join(lines[min(max(start, 0), len(lines)):], "")
}

// wordCounts are the counts of wc.
type wordCounts struct {
	lines      int
	words      int
	bytes      int
	characters int
}

// runWordCount carries out "wc [-lwcm] [file...]".
func runWordCount(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	var columns []string
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-l", "--lines":
			columns = append(columns, "lines")
		case "-w", "--words":
			columns = append(columns, "words")
		case "-c", "--bytes":
			columns = append(columns, "bytes")
		case "-m", "--chars":
			columns = append(columns, "characters")
		default:
			return call.invalidFlag(flag)
		}
	}

	if len(columns) == 0 {
		columns = []string{
			"lines",
			"words",
			"bytes",
		}
	}

	inputs, ok := call.readInputs(parsed.operands)

	var rows []countRow
	var total wordCounts

	for _, input := range inputs {
		counts := countWords(input.contents)
		total = addCounts(total, counts)

		rows = append(rows, countRow{
			name:   input.name,
			counts: counts,
		})
	}

	if len(rows) > 1 {
		rows = append(rows, countRow{
			name:   "total",
			counts: total,
		})
	}

	width := countWidth(total, columns, len(rows) == 1 && len(columns) == 1)
	if slices.ContainsFunc(inputs, isStandardInput) && len(columns) > 1 {
		width = max(width, 7)
	}

	for _, row := range rows {
		call.writeOutput(formatCountRow(row, columns, width))
	}

	return statusOf(ok)
}

// countRow is one output line of wc.
type countRow struct {
	name   string
	counts wordCounts
}

// isStandardInput reports whether an input is the standard input.
func isStandardInput(input namedInput) bool {
	return input.name == "-"
}

// countWords counts the lines, words, bytes and characters of contents.
func countWords(contents []byte) wordCounts {
	text := string(contents)

	return wordCounts{
		lines:      strings.Count(text, "\n"),
		words:      len(strings.Fields(text)),
		bytes:      len(contents),
		characters: utf8.RuneCount(contents),
	}
}

// addCounts sums two counts.
func addCounts(first wordCounts, second wordCounts) wordCounts {
	return wordCounts{
		lines:      first.lines + second.lines,
		words:      first.words + second.words,
		bytes:      first.bytes + second.bytes,
		characters: first.characters + second.characters,
	}
}

// countValue returns one column of counts.
func countValue(counts wordCounts, column string) int {
	switch column {
	case "lines":
		return counts.lines
	case "words":
		return counts.words
	case "characters":
		return counts.characters
	default:
		return counts.bytes
	}
}

// countWidth returns the column width of wc: the digits of the largest
// count, or no padding for a single value.
func countWidth(total wordCounts, columns []string, single bool) int {
	if single {
		return 1
	}

	width := 1
	for _, column := range columns {
		width = max(width, len(strconv.Itoa(countValue(total, column))))
	}

	return width
}

// formatCountRow formats one output line of wc.
func formatCountRow(row countRow, columns []string, width int) string {
	values := make([]string, 0, len(columns)+1)
	for _, column := range columns {
		values = append(values, fmt.Sprintf("%*d", width, countValue(row.counts, column)))
	}

	if row.name != "-" {
		values = append(values, row.name)
	}

	return strings.Join(values, " ") + "\n"
}

// runUniq carries out "uniq [-c] [-d] [-u] [-i] [input [output]]".
func runUniq(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	var options uniqOptions
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-c", "--count":
			options.count = true
		case "-d", "--repeated":
			options.repeated = true
		case "-u", "--unique":
			options.unique = true
		case "-i", "--ignore-case":
			options.ignoreCase = true
		default:
			return call.invalidFlag(flag)
		}
	}

	if len(parsed.operands) > 2 {
		return call.fail("extra operand '%s'", parsed.operands[2])
	}

	inputs, ok := call.readInputs(parsed.operands[:min(len(parsed.operands), 1)])
	if !ok {
		return exitFailure
	}

	output := uniqueLines(splitLines(inputs[0].contents), options)
	if len(parsed.operands) < 2 {
		_ = call.writeLines(output)

		return 0
	}

	contents := strings.Join(output, "\n")
	if len(output) > 0 {
		contents += "\n"
	}

	if err := call.fs().WriteFile(call.resolve(parsed.operands[1]), []byte(contents), 0o644); err != nil {
		call.reportFile(parsed.operands[1], err)

		return exitFailure
	}

	return 0
}

// uniqOptions are the flags of uniq.
type uniqOptions struct {
	count      bool
	repeated   bool
	unique     bool
	ignoreCase bool
}

// uniqueLines collapses adjacent equal lines.
func uniqueLines(lines []string, options uniqOptions) []string {
	var output []string

	for start := 0; start < len(lines); {
		end := start + 1
		for end < len(lines) && sameLine(lines[start], lines[end], options.ignoreCase) {
			end++
		}

		occurrences := end - start
		excludedAsSingle := options.repeated && occurrences == 1
		excludedAsRepeated := options.unique && occurrences > 1
		selected := !excludedAsSingle && !excludedAsRepeated

		if selected && options.count {
			output = append(output, fmt.Sprintf("%7d %s", occurrences, lines[start]))
		} else if selected {
			output = append(output, lines[start])
		}

		start = end
	}

	return output
}

// sameLine compares lines for uniq.
func sameLine(first string, second string, ignoreCase bool) bool {
	if ignoreCase {
		return strings.EqualFold(first, second)
	}

	return first == second
}

// runReverse carries out "rev [file...]".
func runReverse(_ context.Context, call *commandCall) int {
	inputs, ok := call.readInputs(call.args)

	for _, input := range inputs {
		for _, line := range splitLines(input.contents) {
			runes := []rune(line)
			slices.Reverse(runes)
			call.writeOutput(string(runes) + "\n")
		}
	}

	return statusOf(ok)
}

// runTac carries out "tac [file...]".
func runTac(_ context.Context, call *commandCall) int {
	inputs, ok := call.readInputs(call.args)

	for _, input := range inputs {
		lines := splitLines(input.contents)
		slices.Reverse(lines)
		_ = call.writeLines(lines)
	}

	return statusOf(ok)
}

// runNumberLines carries out "nl [-b a|t|n] [file...]".
func runNumberLines(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-b",
			"-w",
			"-s",
		},
	})
	if !ok {
		return exitFailure
	}

	style := "t"
	width := 6
	separator := "\t"

	for _, flag := range parsed.flags {
		switch flag.name {
		case "-b":
			style = flag.value
		case "-w":
			parsedWidth, err := strconv.Atoi(flag.value)
			if err != nil {
				return call.fail("invalid line number field width: '%s'", flag.value)
			}

			width = parsedWidth
		case "-s":
			separator = flag.value
		default:
			return call.invalidFlag(flag)
		}
	}

	inputs, ok := call.readInputs(parsed.operands)
	number := 0

	for _, input := range inputs {
		for _, line := range splitLines(input.contents) {
			numbered := style == "a" || style == "t" && line != ""
			if !numbered {
				call.writeOutput(strings.Repeat(" ", width+len(separator)) + line + "\n")

				continue
			}

			number++
			call.writeOutput(fmt.Sprintf("%*d%s%s\n", width, number, separator, line))
		}
	}

	return statusOf(ok)
}

// runBase64 carries out "base64 [-d] [-w columns] [file]".
func runBase64(_ context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-w",
			"--wrap",
		},
	})
	if !ok {
		return exitFailure
	}

	decode := false
	columns := 76

	for _, flag := range parsed.flags {
		switch flag.name {
		case "-d", "--decode":
			decode = true
		case "-i", "--ignore-garbage":
		case "-w", "--wrap":
			parsedColumns, err := strconv.Atoi(flag.value)
			if err != nil {
				return call.fail("invalid wrap size: '%s'", flag.value)
			}

			columns = parsedColumns
		default:
			return call.invalidFlag(flag)
		}
	}

	inputs, ok := call.readInputs(parsed.operands)
	if !ok {
		return exitFailure
	}

	if decode {
		return call.decodeBase64(inputs[0].contents)
	}

	call.writeOutput(wrapText(base64.StdEncoding.EncodeToString(inputs[0].contents), columns))

	return 0
}

// decodeBase64 writes the decoded form of base64 text, ignoring
// whitespace.
func (c *commandCall) decodeBase64(contents []byte) int {
	compact := strings.Join(strings.Fields(string(contents)), "")

	decoded, err := base64.StdEncoding.DecodeString(compact)
	if err != nil {
		return c.fail("invalid input")
	}

	_, _ = c.stdout.Write(decoded)

	return 0
}

// wrapText breaks text into lines of at most columns characters, with a
// final newline. Zero columns disables wrapping.
func wrapText(text string, columns int) string {
	if text == "" {
		return ""
	}

	if columns <= 0 {
		return text + "\n"
	}

	var wrapped strings.Builder
	for len(text) > columns {
		wrapped.WriteString(text[:columns] + "\n")
		text = text[columns:]
	}

	wrapped.WriteString(text + "\n")

	return wrapped.String()
}

// runMD5Sum carries out "md5sum [file...]".
func runMD5Sum(_ context.Context, call *commandCall) int {
	return call.printDigests(md5.New)
}

// runSHA1Sum carries out "sha1sum [file...]".
func runSHA1Sum(_ context.Context, call *commandCall) int {
	return call.printDigests(sha1.New)
}

// runSHA256Sum carries out "sha256sum [file...]".
func runSHA256Sum(_ context.Context, call *commandCall) int {
	return call.printDigests(sha256.New)
}

// printDigests prints the digest of every input in the coreutils format.
func (c *commandCall) printDigests(newHash func() hash.Hash) int {
	parsed, ok := c.parseCommandArgs(argSpec{})
	if !ok {
		return exitFailure
	}

	for _, flag := range parsed.flags {
		switch flag.name {
		case "-b", "-t", "--binary", "--text":
		default:
			return c.invalidFlag(flag)
		}
	}

	inputs, ok := c.readInputs(parsed.operands)

	for _, input := range inputs {
		digest := newHash()
		digest.Write(input.contents)
		c.writeOutput(hex.EncodeToString(digest.Sum(nil)) + "  " + input.name + "\n")
	}

	return statusOf(ok)
}
