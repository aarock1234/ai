package sandboxshell

import (
	"context"
	"fmt"
	"io/fs"
	"path"
	"regexp"
	"strconv"
	"strings"
)

// patternSyntax selects how grep and sed read patterns.
type patternSyntax int

// Pattern syntaxes.
const (
	basicPatterns patternSyntax = iota
	extendedPatterns
	fixedPatterns
)

// grepOptions are the flags of grep.
type grepOptions struct {
	syntax       patternSyntax
	patterns     []string
	ignoreCase   bool
	invert       bool
	wordMatch    bool
	lineMatch    bool
	count        bool
	filesWith    bool
	filesWithout bool
	lineNumbers  bool
	onlyMatch    bool
	quiet        bool
	noMessages   bool
	recursive    bool
	withNames    *bool
	before       int
	after        int
	maxCount     int
	include      []string
	exclude      []string
	excludeDirs  []string
}

// runGrep carries out grep with basic regular expressions by default.
func runGrep(ctx context.Context, call *commandCall) int {
	return call.grep(ctx, basicPatterns)
}

// runExtendedGrep carries out egrep, grep -E.
func runExtendedGrep(ctx context.Context, call *commandCall) int {
	return call.grep(ctx, extendedPatterns)
}

// runFixedGrep carries out fgrep, grep -F.
func runFixedGrep(ctx context.Context, call *commandCall) int {
	return call.grep(ctx, fixedPatterns)
}

// grep searches files or the standard input for lines matching patterns.
// It exits with 0 when a line matched, 1 when none did and 2 on errors.
func (c *commandCall) grep(ctx context.Context, syntax patternSyntax) int {
	options, operands, ok := c.parseGrepOptions(syntax)
	if !ok {
		return exitUsage
	}

	matcher, err := compileGrepPattern(options)
	if err != nil {
		return c.grepFailure("invalid regular expression: %v", err)
	}

	files, ok := c.grepTargets(ctx, operands, options)
	showNames := len(files) > 1 || options.recursive
	if options.withNames != nil {
		showNames = *options.withNames
	}

	matched := false
	for _, file := range files {
		if ctx.Err() != nil {
			return exitInterrupted
		}

		contents, err := c.readOperand(file)
		if err != nil {
			if !options.noMessages {
				c.reportFile(file, err)
			}

			ok = false

			continue
		}

		search := grepSearch{
			call:      c,
			options:   options,
			matcher:   matcher,
			name:      displayName(file),
			showNames: showNames,
		}

		if search.run(string(contents)) {
			matched = true
		}

		if matched && options.quiet {
			return 0
		}
	}

	quietlyMatched := matched && options.quiet

	switch {
	case !ok && !quietlyMatched:
		return exitUsage
	case matched:
		return 0
	default:
		return exitFailure
	}
}

// grepFailure reports an error and returns exit code 2.
func (c *commandCall) grepFailure(format string, values ...any) int {
	c.report(format, values...)

	return exitUsage
}

// displayName names an input in grep's output.
func displayName(file string) string {
	if file == "-" {
		return "(standard input)"
	}

	return file
}

// parseGrepOptions reads the flags and the pattern operand of grep.
func (c *commandCall) parseGrepOptions(syntax patternSyntax) (grepOptions, []string, bool) {
	parsed, ok := c.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-e",
			"-f",
			"-A",
			"-B",
			"-C",
			"-m",
			"--regexp",
			"--file",
			"--after-context",
			"--before-context",
			"--context",
			"--max-count",
			"--include",
			"--exclude",
			"--exclude-dir",
		},
	})
	if !ok {
		return grepOptions{}, nil, false
	}

	options := grepOptions{
		syntax: syntax,
	}

	explicitPatterns := false
	for _, flag := range parsed.flags {
		applied, explicit := c.applyGrepFlag(&options, flag)
		if !applied {
			return grepOptions{}, nil, false
		}

		explicitPatterns = explicitPatterns || explicit
	}

	operands := parsed.operands
	if !explicitPatterns {
		if len(operands) == 0 {
			c.report("missing pattern")

			return grepOptions{}, nil, false
		}

		options.patterns = append(options.patterns, strings.Split(operands[0], "\n")...)
		operands = operands[1:]
	}

	return options, operands, true
}

// applyGrepFlag applies one flag. It reports whether the flag is valid and
// whether it supplied patterns.
func (c *commandCall) applyGrepFlag(options *grepOptions, flag parsedFlag) (bool, bool) {
	switch flag.name {
	case "-e", "--regexp":
		options.patterns = append(options.patterns, strings.Split(flag.value, "\n")...)

		return true, true
	case "-f", "--file":
		contents, err := c.readOperand(flag.value)
		if err != nil {
			c.reportFile(flag.value, err)

			return false, true
		}

		options.patterns = append(options.patterns, splitLines(contents)...)

		return true, true
	}

	return c.applyGrepSetting(options, flag), false
}

// applyGrepSetting applies a flag that does not supply patterns.
func (c *commandCall) applyGrepSetting(options *grepOptions, flag parsedFlag) bool {
	switch flag.name {
	case "-E", "--extended-regexp", "-P", "--perl-regexp":
		options.syntax = extendedPatterns
	case "-F", "--fixed-strings":
		options.syntax = fixedPatterns
	case "-G", "--basic-regexp":
		options.syntax = basicPatterns
	case "-i", "-y", "--ignore-case":
		options.ignoreCase = true
	case "-v", "--invert-match":
		options.invert = true
	case "-w", "--word-regexp":
		options.wordMatch = true
	case "-x", "--line-regexp":
		options.lineMatch = true
	case "-c", "--count":
		options.count = true
	case "-l", "--files-with-matches":
		options.filesWith = true
	case "-L", "--files-without-match":
		options.filesWithout = true
	case "-n", "--line-number":
		options.lineNumbers = true
	case "-o", "--only-matching":
		options.onlyMatch = true
	case "-q", "--quiet", "--silent":
		options.quiet = true
	case "-s", "--no-messages":
		options.noMessages = true
	case "-r", "-R", "--recursive", "--dereference-recursive":
		options.recursive = true
	case "-h", "--no-filename":
		options.withNames = new(false)
	case "-H", "--with-filename":
		options.withNames = new(true)
	case "-A", "--after-context", "-B", "--before-context", "-C", "--context", "-m", "--max-count":
		return c.applyGrepCount(options, flag)
	case "--include":
		options.include = append(options.include, flag.value)
	case "--exclude":
		options.exclude = append(options.exclude, flag.value)
	case "--exclude-dir":
		options.excludeDirs = append(options.excludeDirs, flag.value)
	case "-a", "-I", "--text", "--color", "--colour", "-U", "--binary":
	default:
		c.invalidFlag(flag)

		return false
	}

	return true
}

// applyGrepCount applies a context or maximum count flag.
func (c *commandCall) applyGrepCount(options *grepOptions, flag parsedFlag) bool {
	count, err := strconv.Atoi(flag.value)
	if err != nil || count < 0 {
		c.report("invalid number: '%s'", flag.value)

		return false
	}

	switch flag.name {
	case "-A", "--after-context":
		options.after = count
	case "-B", "--before-context":
		options.before = count
	case "-C", "--context":
		options.after = count
		options.before = count
	default:
		options.maxCount = count
	}

	return true
}

// grepTargets lists the inputs: the operands, the files below directory
// operands when recursive, or the standard input.
func (c *commandCall) grepTargets(ctx context.Context, operands []string, options grepOptions) ([]string, bool) {
	if len(operands) == 0 && options.recursive {
		operands = []string{
			".",
		}
	}

	if len(operands) == 0 {
		return []string{
			"-",
		}, true
	}

	var files []string
	ok := true

	for _, operand := range operands {
		info, err := c.fs().Stat(c.resolve(operand))
		if err != nil || !info.IsDir() || operand == "-" {
			files = append(files, operand)

			continue
		}

		if !options.recursive {
			c.report("%s: Is a directory", operand)
			ok = false

			continue
		}

		files = append(files, c.walkGrepDirectory(ctx, operand, options)...)
	}

	return files, ok
}

// walkGrepDirectory returns the files below a directory that pass the
// include and exclude filters, in name order.
func (c *commandCall) walkGrepDirectory(ctx context.Context, directory string, options grepOptions) []string {
	var files []string

	_ = c.walk(ctx, directory, func(name string, info fs.FileInfo, depth int) walkDecision {
		base := path.Base(name)

		if info.IsDir() {
			if depth > 0 && matchesAnyGlob(options.excludeDirs, base) {
				return skipDirectory
			}

			return continueWalk
		}

		included := len(options.include) == 0 || matchesAnyGlob(options.include, base)
		if included && !matchesAnyGlob(options.exclude, base) {
			files = append(files, name)
		}

		return continueWalk
	})

	return files
}

// matchesAnyGlob reports whether a name matches any shell pattern.
func matchesAnyGlob(patterns []string, name string) bool {
	for _, pattern := range patterns {
		if matched, _ := path.Match(pattern, name); matched {
			return true
		}
	}

	return false
}

// compileGrepPattern compiles grep's patterns into one expression.
func compileGrepPattern(options grepOptions) (*regexp.Regexp, error) {
	alternatives := make([]string, 0, len(options.patterns))

	for _, pattern := range options.patterns {
		expression, err := translatePattern(pattern, options.syntax)
		if err != nil {
			return nil, err
		}

		switch {
		case options.lineMatch:
			expression = "^(?:" + expression + ")$"
		case options.wordMatch:
			expression = `\b(?:` + expression + `)\b`
		}

		alternatives = append(alternatives, "(?:"+expression+")")
	}

	expression := strings.Join(alternatives, "|")
	if options.ignoreCase {
		expression = "(?i)" + expression
	}

	return regexp.Compile(expression)
}

// translatePattern converts a POSIX basic, extended or fixed pattern into
// Go regular expression syntax.
func translatePattern(pattern string, syntax patternSyntax) (string, error) {
	switch syntax {
	case fixedPatterns:
		return regexp.QuoteMeta(pattern), nil
	case extendedPatterns:
		return translateExtended(pattern), nil
	default:
		return translateBasic(pattern), nil
	}
}

// translateExtended converts the GNU word boundary escapes of an extended
// pattern.
func translateExtended(pattern string) string {
	return strings.NewReplacer(`\<`, `\b`, `\>`, `\b`).Replace(pattern)
}

// translateBasic converts a basic pattern, where \( \) \{ \} \| \+ \? are
// operators and the bare characters are literal.
func translateBasic(pattern string) string {
	var output strings.Builder

	for index := 0; index < len(pattern); index++ {
		character := pattern[index]

		switch {
		case character == '[':
			end := bracketEnd(pattern, index)
			output.WriteString(pattern[index:end])
			index = end - 1
		case character == '\\' && index+1 < len(pattern):
			index++
			output.WriteString(translateBasicEscape(pattern[index]))
		case strings.IndexByte("(){}|+?", character) >= 0:
			output.WriteString(`\` + string(character))
		case character == '*' && isPatternStart(pattern, index):
			output.WriteString(`\*`)
		default:
			output.WriteByte(character)
		}
	}

	return output.String()
}

// translateBasicEscape converts the character after a backslash of a basic
// pattern.
func translateBasicEscape(character byte) string {
	switch character {
	case '(', ')', '{', '}', '|', '+', '?':
		return string(character)
	case '<', '>':
		return `\b`
	default:
		return `\` + string(character)
	}
}

// isPatternStart reports whether a star at index is literal: at the start
// of a basic pattern, after ^ or after an opening group.
func isPatternStart(pattern string, index int) bool {
	prefix := pattern[:index]

	return prefix == "" || prefix == "^" || strings.HasSuffix(prefix, `\(`)
}

// bracketEnd returns the index after the bracket expression starting at
// index, whose characters are copied verbatim.
func bracketEnd(pattern string, index int) int {
	position := index + 1
	if position < len(pattern) && pattern[position] == '^' {
		position++
	}

	if position < len(pattern) && pattern[position] == ']' {
		position++
	}

	for position < len(pattern) {
		if strings.HasPrefix(pattern[position:], "[:") {
			if end := strings.Index(pattern[position:], ":]"); end >= 0 {
				position += end + 2

				continue
			}
		}

		if pattern[position] == ']' {
			return position + 1
		}

		position++
	}

	return len(pattern)
}

// grepSearch searches one input.
type grepSearch struct {
	call      *commandCall
	options   grepOptions
	matcher   *regexp.Regexp
	name      string
	showNames bool
}

// run searches the contents and prints the results. It reports whether any
// line matched.
func (s grepSearch) run(contents string) bool {
	lines := splitLines([]byte(contents))
	matches := s.matchingLines(lines)

	switch {
	case s.options.quiet:
	case s.options.filesWith:
		if len(matches) > 0 {
			s.call.writeOutput(s.name + "\n")
		}
	case s.options.filesWithout:
		if len(matches) == 0 {
			s.call.writeOutput(s.name + "\n")
		}
	case s.options.count:
		s.call.writeOutput(s.prefix(0, ':', false) + strconv.Itoa(len(matches)) + "\n")
	default:
		s.printMatches(lines, matches)
	}

	return len(matches) > 0
}

// matchingLines returns the indexes of the selected lines.
func (s grepSearch) matchingLines(lines []string) []int {
	var matches []int

	for index, line := range lines {
		if s.matcher.MatchString(line) == s.options.invert {
			continue
		}

		matches = append(matches, index)
		if s.options.maxCount > 0 && len(matches) == s.options.maxCount {
			break
		}
	}

	return matches
}

// printMatches prints the selected lines with their context.
func (s grepSearch) printMatches(lines []string, matches []int) {
	isMatch := make(map[int]bool, len(matches))
	for _, index := range matches {
		isMatch[index] = true
	}

	hasContext := s.options.before > 0 || s.options.after > 0
	nextUnprinted := 0

	for _, match := range matches {
		start := max(match-s.options.before, nextUnprinted)
		if hasContext && nextUnprinted > 0 && start > nextUnprinted {
			s.call.writeOutput("--\n")
		}

		end := min(match+s.options.after, len(lines)-1)
		for index := start; index <= end; index++ {
			if index < nextUnprinted {
				continue
			}

			if index > match && isMatch[index] {
				break
			}

			s.printLine(lines[index], index, isMatch[index])
			nextUnprinted = index + 1
		}
	}
}

// printLine prints one selected or context line.
func (s grepSearch) printLine(line string, index int, selected bool) {
	separator := byte('-')
	if selected {
		separator = ':'
	}

	if !s.options.onlyMatch || s.options.invert {
		s.call.writeOutput(s.prefix(index, separator, s.options.lineNumbers) + line + "\n")

		return
	}

	if !selected {
		return
	}

	for _, match := range s.matcher.FindAllString(line, -1) {
		if match != "" {
			s.call.writeOutput(s.prefix(index, separator, s.options.lineNumbers) + match + "\n")
		}
	}
}

// prefix returns the file name and line number prefix of an output line.
func (s grepSearch) prefix(index int, separator byte, withLineNumber bool) string {
	var prefix strings.Builder

	if s.showNames {
		prefix.WriteString(s.name)
		prefix.WriteByte(separator)
	}

	if withLineNumber {
		fmt.Fprintf(&prefix, "%d%c", index+1, separator)
	}

	return prefix.String()
}
