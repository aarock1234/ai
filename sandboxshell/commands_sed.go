package sandboxshell

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// runSed carries out "sed [-n] [-E] [-i[suffix]] [-e script]... [script]
// [file...]": the stream editor with addresses, blocks, branches, the hold
// space and the s, y, a, i, c, d, D, p, P, n, N, g, G, h, H, x, =, q, Q, z,
// b, t and T commands.
func runSed(ctx context.Context, call *commandCall) int {
	options, operands, ok := parseSedOptions(call)
	if !ok {
		return exitFailure
	}

	program, err := parseSedProgram(options.script, options.syntax)
	if err != nil {
		return call.fail("-e expression #1: %v", err)
	}

	if options.inPlace {
		return call.editInPlace(ctx, program, options, operands)
	}

	inputs, ok := call.readInputs(operands)

	var combined strings.Builder
	for _, input := range inputs {
		combined.Write(input.contents)
	}

	output, exitCode := executeSed(ctx, program, combined.String(), options.quiet)
	call.writeOutput(output)

	if !ok && exitCode == 0 {
		return exitUsage
	}

	return exitCode
}

// sedOptions are the flags of sed.
type sedOptions struct {
	script       string
	syntax       patternSyntax
	quiet        bool
	inPlace      bool
	backupSuffix string
}

// parseSedOptions reads sed's flags. The -i flag only takes an attached
// suffix, so it is handled before the generic parser.
func parseSedOptions(call *commandCall) (sedOptions, []string, bool) {
	var options sedOptions
	var args []string

	for _, arg := range call.args {
		suffix, isInPlace := strings.CutPrefix(arg, "-i")
		if !isInPlace {
			suffix, isInPlace = strings.CutPrefix(arg, "--in-place")
			suffix = strings.TrimPrefix(suffix, "=")
		}

		if isInPlace {
			options.inPlace = true
			options.backupSuffix = suffix

			continue
		}

		args = append(args, arg)
	}

	call.args = args

	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-e",
			"-f",
			"--expression",
			"--file",
		},
	})
	if !ok {
		return sedOptions{}, nil, false
	}

	var scripts []string
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-n", "--quiet", "--silent":
			options.quiet = true
		case "-E", "-r", "--regexp-extended":
			options.syntax = extendedPatterns
		case "-e", "--expression":
			scripts = append(scripts, flag.value)
		case "-f", "--file":
			contents, err := call.readOperand(flag.value)
			if err != nil {
				call.reportFile(flag.value, err)

				return sedOptions{}, nil, false
			}

			scripts = append(scripts, strings.TrimSuffix(string(contents), "\n"))
		case "-s", "--separate", "-u", "--unbuffered", "--posix", "-z":
		default:
			call.invalidFlag(flag)

			return sedOptions{}, nil, false
		}
	}

	operands := parsed.operands
	if len(scripts) == 0 {
		if len(operands) == 0 {
			call.report("no script specified")

			return sedOptions{}, nil, false
		}

		scripts = append(scripts, operands[0])
		operands = operands[1:]
	}

	options.script = strings.Join(scripts, "\n")

	return options, operands, true
}

// editInPlace runs the program on every file separately and writes the
// result back, keeping a backup when a suffix was given.
func (c *commandCall) editInPlace(ctx context.Context, program *sedProgram, options sedOptions, operands []string) int {
	if len(operands) == 0 {
		return c.fail("no input files")
	}

	exitCode := 0
	for _, operand := range operands {
		resolved := c.resolve(operand)

		contents, err := c.fs().ReadFile(resolved)
		if err != nil {
			c.report("can't read %s: %s", operand, describeError(err))
			exitCode = exitUsage

			continue
		}

		output, code := executeSed(ctx, program, string(contents), options.quiet)
		exitCode = max(exitCode, code)

		if options.backupSuffix != "" {
			if err := c.fs().WriteFile(resolved+options.backupSuffix, contents, 0o644); err != nil {
				c.reportFile(operand+options.backupSuffix, err)
				exitCode = exitFailure

				continue
			}
		}

		if err := c.fs().WriteFile(resolved, []byte(output), 0o644); err != nil {
			c.reportFile(operand, err)
			exitCode = exitFailure
		}
	}

	return exitCode
}

// sedProgram is a parsed script, flattened into instructions: a block is
// an instruction that jumps past its end when its address does not match.
type sedProgram struct {
	instructions []sedInstruction
}

// sedInstruction is one command with its addresses.
type sedInstruction struct {
	first  *sedAddress
	second *sedAddress
	negate bool

	command byte
	text    string

	// jump is the instruction after a block's end, or a branch target; -1
	// branches to the end of the program.
	jump  int
	label string

	substitution *sedSubstitution
	translation  map[rune]rune
	exitCode     int
}

// sedAddress selects lines by number, as the last line, or by expression.
type sedAddress struct {
	line       int
	last       bool
	expression *regexp.Regexp
}

// sedSubstitution is the s command.
type sedSubstitution struct {
	expression  *regexp.Regexp
	replacement []replacementPart
	global      bool
	occurrence  int
	print       bool
}

// replacementPart is literal text, or a group of the match: 0 is the whole
// match.
type replacementPart struct {
	literal string
	group   int
}

// sedParser parses a sed script.
type sedParser struct {
	script   string
	position int
	syntax   patternSyntax

	program    *sedProgram
	openBlocks []int
	labels     map[string]int
}

// parseSedProgram parses a script.
func parseSedProgram(script string, syntax patternSyntax) (*sedProgram, error) {
	parser := &sedParser{
		script:  script,
		syntax:  syntax,
		program: &sedProgram{},
		labels:  make(map[string]int),
	}

	if err := parser.parse(); err != nil {
		return nil, err
	}

	return parser.program, nil
}

// parse parses every command and resolves branch labels.
func (p *sedParser) parse() error {
	for {
		p.skipSeparators()
		if p.atEnd() {
			break
		}

		if err := p.parseCommand(); err != nil {
			return err
		}
	}

	if len(p.openBlocks) > 0 {
		return errors.New("unmatched `{'")
	}

	return p.resolveLabels()
}

// resolveLabels points the branch instructions at their labels.
func (p *sedParser) resolveLabels() error {
	for index := range p.program.instructions {
		instruction := &p.program.instructions[index]
		if !strings.ContainsRune("btT", rune(instruction.command)) {
			continue
		}

		if instruction.label == "" {
			instruction.jump = -1

			continue
		}

		target, ok := p.labels[instruction.label]
		if !ok {
			return fmt.Errorf("can't find label for jump to `%s'", instruction.label)
		}

		instruction.jump = target
	}

	return nil
}

// parseCommand parses one addressed command.
func (p *sedParser) parseCommand() error {
	var instruction sedInstruction
	var err error

	instruction.first, err = p.parseAddress()
	if err != nil {
		return err
	}

	if instruction.first != nil && p.accept(',') {
		instruction.second, err = p.parseAddress()
		if err != nil {
			return err
		}

		if instruction.second == nil {
			return errors.New("unexpected `,'")
		}
	}

	p.skipSpaces()
	for p.accept('!') {
		instruction.negate = true
		p.skipSpaces()
	}

	if p.atEnd() {
		return errors.New("missing command")
	}

	instruction.command = p.next()

	return p.parseCommandBody(instruction)
}

// parseCommandBody parses the arguments of a command and adds it.
func (p *sedParser) parseCommandBody(instruction sedInstruction) error {
	switch instruction.command {
	case '{':
		p.openBlocks = append(p.openBlocks, len(p.program.instructions))
	case '}':
		return p.closeBlock()
	case ':':
		label := strings.TrimSpace(p.readUntil(";\n"))
		if label == "" {
			return errors.New("\":\" lacks a label")
		}

		p.labels[label] = len(p.program.instructions)

		return nil
	case 'b', 't', 'T':
		instruction.label = strings.TrimSpace(p.readUntil(";\n}"))
	case 's':
		substitution, err := p.parseSubstitution()
		if err != nil {
			return err
		}

		instruction.substitution = substitution
	case 'y':
		translation, err := p.parseTranslation()
		if err != nil {
			return err
		}

		instruction.translation = translation
	case 'a', 'i', 'c':
		instruction.text = p.parseText()
	case 'q', 'Q':
		p.skipSpaces()
		instruction.exitCode, _ = strconv.Atoi(p.readWhile(unicode.IsDigit))
	case 'd', 'D', 'p', 'P', 'n', 'N', 'g', 'G', 'h', 'H', 'x', '=', 'z':
	default:
		return fmt.Errorf("unknown command: `%c'", instruction.command)
	}

	p.program.instructions = append(p.program.instructions, instruction)

	return nil
}

// closeBlock ends the innermost block.
func (p *sedParser) closeBlock() error {
	if len(p.openBlocks) == 0 {
		return errors.New("unexpected `}'")
	}

	opening := p.openBlocks[len(p.openBlocks)-1]
	p.openBlocks = p.openBlocks[:len(p.openBlocks)-1]
	p.program.instructions[opening].jump = len(p.program.instructions)

	return nil
}

// parseAddress parses a line number, $, /regex/ or \cregexc address.
func (p *sedParser) parseAddress() (*sedAddress, error) {
	if p.atEnd() {
		return nil, nil
	}

	switch character := p.script[p.position]; {
	case character >= '0' && character <= '9':
		line, _ := strconv.Atoi(p.readWhile(unicode.IsDigit))

		return &sedAddress{
			line: line,
		}, nil
	case character == '$':
		p.position++

		return &sedAddress{
			last: true,
		}, nil
	case character == '/' || character == '\\':
		p.position++

		delimiter := byte('/')
		if character == '\\' {
			delimiter = p.next()
		}

		expression, err := p.parseExpression(delimiter, false)
		if err != nil {
			return nil, err
		}

		return &sedAddress{
			expression: expression,
		}, nil
	default:
		return nil, nil
	}
}

// parseSubstitution parses "s/regex/replacement/flags".
func (p *sedParser) parseSubstitution() (*sedSubstitution, error) {
	if p.atEnd() {
		return nil, errors.New("unterminated `s' command")
	}

	delimiter := p.next()

	source, err := p.readDelimited(delimiter)
	if err != nil {
		return nil, errors.New("unterminated `s' command")
	}

	replacement, err := p.readDelimited(delimiter)
	if err != nil {
		return nil, errors.New("unterminated `s' command")
	}

	substitution := &sedSubstitution{
		replacement: parseReplacement(replacement),
		occurrence:  1,
	}

	ignoreCase := false
	for !p.atEnd() && !strings.ContainsRune(";\n}", rune(p.script[p.position])) {
		flag := p.next()

		switch {
		case flag == 'g':
			substitution.global = true
		case flag == 'p':
			substitution.print = true
		case flag == 'i' || flag == 'I':
			ignoreCase = true
		case flag >= '0' && flag <= '9':
			p.position--
			substitution.occurrence, _ = strconv.Atoi(p.readWhile(unicode.IsDigit))
		case flag == ' ' || flag == '\t':
		default:
			return nil, fmt.Errorf("unknown option to `s'")
		}
	}

	substitution.expression, err = compileSedExpression(source, p.syntax, ignoreCase)
	if err != nil {
		return nil, err
	}

	return substitution, nil
}

// parseExpression reads a delimited regular expression and compiles it.
func (p *sedParser) parseExpression(delimiter byte, ignoreCase bool) (*regexp.Regexp, error) {
	source, err := p.readDelimited(delimiter)
	if err != nil {
		return nil, errors.New("unterminated address regex")
	}

	return compileSedExpression(source, p.syntax, ignoreCase)
}

// compileSedExpression compiles a POSIX pattern with leftmost-longest
// matching.
func compileSedExpression(source string, syntax patternSyntax, ignoreCase bool) (*regexp.Regexp, error) {
	translated, err := translatePattern(source, syntax)
	if err != nil {
		return nil, err
	}

	if ignoreCase {
		translated = "(?i)" + translated
	}

	expression, err := regexp.Compile(translated)
	if err != nil {
		return nil, fmt.Errorf("invalid regular expression %q: %w", source, err)
	}

	expression.Longest()

	return expression, nil
}

// parseTranslation parses "y/source/target/".
func (p *sedParser) parseTranslation() (map[rune]rune, error) {
	if p.atEnd() {
		return nil, errors.New("unterminated `y' command")
	}

	delimiter := p.next()

	source, sourceErr := p.readDelimited(delimiter)
	target, targetErr := p.readDelimited(delimiter)
	if sourceErr != nil || targetErr != nil {
		return nil, errors.New("unterminated `y' command")
	}

	sourceRunes := []rune(unescapeBackslashes(source))
	targetRunes := []rune(unescapeBackslashes(target))
	if len(sourceRunes) != len(targetRunes) {
		return nil, errors.New("strings for `y' command are different lengths")
	}

	translation := make(map[rune]rune, len(sourceRunes))
	for index, character := range sourceRunes {
		translation[character] = targetRunes[index]
	}

	return translation, nil
}

// parseText parses the text of a, i and c: "a text" or "a\" followed by
// the text on the next line.
func (p *sedParser) parseText() string {
	p.skipSpaces()

	if p.accept('\\') {
		p.accept('\n')
	}

	return unescapeBackslashes(p.readUntil("\n"))
}

// readDelimited reads up to an unescaped delimiter and consumes it. An
// escaped delimiter becomes the plain delimiter; other escapes are kept.
func (p *sedParser) readDelimited(delimiter byte) (string, error) {
	var text strings.Builder

	for !p.atEnd() {
		character := p.next()

		switch {
		case character == delimiter:
			return text.String(), nil
		case character == '\\' && !p.atEnd() && p.script[p.position] == delimiter:
			text.WriteByte(p.next())
		case character == '\\' && !p.atEnd():
			text.WriteByte(character)
			text.WriteByte(p.next())
		default:
			text.WriteByte(character)
		}
	}

	return "", errors.New("unterminated text")
}

// parseReplacement splits replacement text into literals and group
// references: & is the whole match and \1 to \9 are groups.
func parseReplacement(replacement string) []replacementPart {
	var parts []replacementPart
	var literal strings.Builder

	flushLiteral := func() {
		if literal.Len() > 0 {
			parts = append(parts, replacementPart{
				literal: literal.String(),
			})
			literal.Reset()
		}
	}

	for index := 0; index < len(replacement); index++ {
		character := replacement[index]

		switch {
		case character == '&':
			flushLiteral()
			parts = append(parts, replacementPart{
				group: 0,
			})
		case character == '\\' && index+1 < len(replacement):
			index++

			escaped := replacement[index]
			if escaped >= '1' && escaped <= '9' {
				flushLiteral()
				parts = append(parts, replacementPart{
					group: int(escaped - '0'),
				})

				continue
			}

			literal.WriteString(replacementEscape(escaped))
		default:
			literal.WriteByte(character)
		}
	}

	flushLiteral()

	return parts
}

// replacementEscape returns the text of an escaped replacement character.
func replacementEscape(character byte) string {
	switch character {
	case 'n':
		return "\n"
	case 't':
		return "\t"
	default:
		return string(character)
	}
}

// skipSeparators skips whitespace and semicolons between commands.
func (p *sedParser) skipSeparators() {
	for !p.atEnd() && strings.ContainsRune(" \t\n;", rune(p.script[p.position])) {
		p.position++
	}
}

// skipSpaces skips blanks.
func (p *sedParser) skipSpaces() {
	for !p.atEnd() && (p.script[p.position] == ' ' || p.script[p.position] == '\t') {
		p.position++
	}
}

// readUntil reads up to, but not including, one of the stop characters.
func (p *sedParser) readUntil(stops string) string {
	start := p.position
	for !p.atEnd() && !strings.ContainsRune(stops, rune(p.script[p.position])) {
		p.position++
	}

	return p.script[start:p.position]
}

// readWhile reads characters that satisfy accept.
func (p *sedParser) readWhile(accept func(rune) bool) string {
	start := p.position
	for !p.atEnd() && accept(rune(p.script[p.position])) {
		p.position++
	}

	return p.script[start:p.position]
}

// accept consumes the next character when it is the given one.
func (p *sedParser) accept(character byte) bool {
	if p.atEnd() || p.script[p.position] != character {
		return false
	}

	p.position++

	return true
}

// next consumes and returns the next character.
func (p *sedParser) next() byte {
	character := p.script[p.position]
	p.position++

	return character
}

// atEnd reports whether the script is consumed.
func (p *sedParser) atEnd() bool {
	return p.position >= len(p.script)
}
