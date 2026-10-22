package sandboxshell

import (
	"context"
	"strconv"
	"strings"
)

// sedExecution is the state of running a program over input text.
type sedExecution struct {
	program *sedProgram
	quiet   bool

	lines        []string
	nextLine     int
	lineNumber   int
	finalNewline bool

	pattern     string
	hold        string
	substituted bool
	activeRange []bool

	appended []string
	output   strings.Builder

	quit     bool
	exitCode int
}

// sedCycleEnd tells the executor how a cycle ended.
type sedCycleEnd int

// Cycle endings.
const (
	// cyclePrints prints the pattern space unless -n was given.
	cyclePrints sedCycleEnd = iota

	// cycleDeletes starts the next cycle without printing.
	cycleDeletes

	// cycleRestarts restarts the program on the remaining pattern space
	// without reading input, after D.
	cycleRestarts
)

// executeSed runs a program over text and returns the output and the exit
// code of q or Q.
func executeSed(ctx context.Context, program *sedProgram, text string, quiet bool) (string, int) {
	execution := &sedExecution{
		program:      program,
		quiet:        quiet,
		lines:        splitLines([]byte(text)),
		finalNewline: text == "" || strings.HasSuffix(text, "\n"),
		activeRange:  make([]bool, len(program.instructions)),
	}

	restart := false
	for !execution.quit && ctx.Err() == nil {
		if !restart && !execution.readLine() {
			break
		}

		end := execution.runCycle()
		restart = end == cycleRestarts

		if end == cyclePrints && !execution.quiet {
			execution.printPattern(true)
		}

		execution.flushAppended()
	}

	return execution.output.String(), execution.exitCode
}

// readLine replaces the pattern space with the next input line. It reports
// false at the end of the input.
func (e *sedExecution) readLine() bool {
	if e.nextLine >= len(e.lines) {
		return false
	}

	e.pattern = e.lines[e.nextLine]
	e.nextLine++
	e.lineNumber++
	e.substituted = false

	return true
}

// isLastLine reports whether the current line is the last input line.
func (e *sedExecution) isLastLine() bool {
	return e.nextLine >= len(e.lines)
}

// printPattern prints the pattern space. The final line keeps the missing
// newline of an input that did not end with one.
func (e *sedExecution) printPattern(autoPrint bool) {
	e.output.WriteString(e.pattern)

	if !autoPrint || !e.isLastLine() || e.finalNewline {
		e.output.WriteByte('\n')
	}
}

// flushAppended prints the text queued by a and the end of the cycle.
func (e *sedExecution) flushAppended() {
	for _, text := range e.appended {
		e.output.WriteString(text + "\n")
	}

	e.appended = e.appended[:0]
}

// runCycle runs the program once over the pattern space.
func (e *sedExecution) runCycle() sedCycleEnd {
	counter := 0

	for counter < len(e.program.instructions) {
		instruction := &e.program.instructions[counter]

		if !e.selects(counter, instruction) {
			counter = e.skip(counter, instruction)

			continue
		}

		next, end, ended := e.apply(counter, instruction)
		if ended {
			return end
		}

		counter = next
	}

	return cyclePrints
}

// skip returns the instruction after an unselected one: past the block for
// an opening brace.
func (e *sedExecution) skip(counter int, instruction *sedInstruction) int {
	if instruction.command == '{' {
		return instruction.jump
	}

	return counter + 1
}

// selects reports whether an instruction's addresses select the current
// line, tracking the state of two-address ranges.
func (e *sedExecution) selects(counter int, instruction *sedInstruction) bool {
	selected := e.matchesRange(counter, instruction)

	return selected != instruction.negate
}

// matchesRange evaluates the addresses without negation.
func (e *sedExecution) matchesRange(counter int, instruction *sedInstruction) bool {
	switch {
	case instruction.first == nil:
		return true
	case instruction.second == nil:
		return e.matchesAddress(instruction.first)
	case e.activeRange[counter]:
		e.activeRange[counter] = !e.endsRange(instruction.second)

		return true
	case e.matchesAddress(instruction.first):
		second := instruction.second
		endsOnThisLine := second.expression == nil && !second.last && second.line <= e.lineNumber
		endsAtLastLine := second.last && e.isLastLine()
		e.activeRange[counter] = !endsOnThisLine && !endsAtLastLine

		return true
	default:
		return false
	}
}

// endsRange reports whether the current line closes an active range.
func (e *sedExecution) endsRange(address *sedAddress) bool {
	if address.expression != nil {
		return address.expression.MatchString(e.pattern)
	}

	return e.matchesAddress(address) || !address.last && e.lineNumber >= address.line
}

// matchesAddress reports whether a single address selects the line.
func (e *sedExecution) matchesAddress(address *sedAddress) bool {
	switch {
	case address.last:
		return e.isLastLine()
	case address.expression != nil:
		return address.expression.MatchString(e.pattern)
	default:
		return address.line == e.lineNumber
	}
}

// apply runs a selected instruction and returns the next instruction, or
// how the cycle ended.
func (e *sedExecution) apply(counter int, instruction *sedInstruction) (int, sedCycleEnd, bool) {
	switch instruction.command {
	case 'b':
		return e.branch(instruction), cyclePrints, false
	case 't':
		if !e.substituted {
			return counter + 1, cyclePrints, false
		}

		e.substituted = false

		return e.branch(instruction), cyclePrints, false
	case 'T':
		if e.substituted {
			e.substituted = false

			return counter + 1, cyclePrints, false
		}

		return e.branch(instruction), cyclePrints, false
	}

	if end, ended := e.applyCycleCommand(counter, instruction); ended {
		return 0, end, true
	}

	e.applyEditCommand(instruction)

	return counter + 1, cyclePrints, false
}

// branch returns the target of a branch; the end of the program ends the
// cycle normally.
func (e *sedExecution) branch(instruction *sedInstruction) int {
	if instruction.jump < 0 {
		return len(e.program.instructions)
	}

	return instruction.jump
}

// applyCycleCommand runs the commands that can end the cycle and reports
// whether they did.
func (e *sedExecution) applyCycleCommand(counter int, instruction *sedInstruction) (sedCycleEnd, bool) {
	switch instruction.command {
	case 'd':
		return cycleDeletes, true
	case 'D':
		_, after, found := strings.Cut(e.pattern, "\n")
		if !found {
			return cycleDeletes, true
		}

		e.pattern = after

		return cycleRestarts, true
	case 'c':
		if !e.activeRange[counter] {
			e.output.WriteString(instruction.text + "\n")
		}

		return cycleDeletes, true
	case 'q':
		e.quit = true
		e.exitCode = instruction.exitCode

		return cyclePrints, true
	case 'Q':
		e.quit = true
		e.exitCode = instruction.exitCode

		return cycleDeletes, true
	case 'n':
		return e.nextInput()
	case 'N':
		return e.appendInput()
	default:
		return cyclePrints, false
	}
}

// nextInput carries out n: print the pattern space and read the next line,
// or end without printing again when there is none.
func (e *sedExecution) nextInput() (sedCycleEnd, bool) {
	if !e.quiet {
		e.printPattern(true)
	}

	if !e.readLine() {
		e.quit = true

		return cycleDeletes, true
	}

	return cyclePrints, false
}

// appendInput carries out N: append the next line, or end the cycle and
// print when there is none, as GNU sed does.
func (e *sedExecution) appendInput() (sedCycleEnd, bool) {
	pattern := e.pattern
	if !e.readLine() {
		e.quit = true

		return cyclePrints, true
	}

	e.pattern = pattern + "\n" + e.pattern

	return cyclePrints, false
}

// applyEditCommand runs the commands that never end the cycle.
func (e *sedExecution) applyEditCommand(instruction *sedInstruction) {
	switch instruction.command {
	case 's':
		e.substitute(instruction.substitution)
	case 'y':
		e.pattern = strings.Map(func(character rune) rune {
			if replacement, ok := instruction.translation[character]; ok {
				return replacement
			}

			return character
		}, e.pattern)
	case 'p':
		e.printPattern(false)
	case 'P':
		first, _, _ := strings.Cut(e.pattern, "\n")
		e.output.WriteString(first + "\n")
	case '=':
		e.output.WriteString(strconv.Itoa(e.lineNumber) + "\n")
	case 'a':
		e.appended = append(e.appended, instruction.text)
	case 'i':
		e.output.WriteString(instruction.text + "\n")
	case 'g':
		e.pattern = e.hold
	case 'G':
		e.pattern += "\n" + e.hold
	case 'h':
		e.hold = e.pattern
	case 'H':
		e.hold += "\n" + e.pattern
	case 'x':
		e.pattern, e.hold = e.hold, e.pattern
	case 'z':
		e.pattern = ""
	}
}

// substitute carries out s on the pattern space.
func (e *sedExecution) substitute(substitution *sedSubstitution) {
	matches := substitution.expression.FindAllStringSubmatchIndex(e.pattern, -1)

	var result strings.Builder
	previousEnd := 0
	replaced := false

	for number, match := range matches {
		occurrence := number + 1
		selected := occurrence == substitution.occurrence || substitution.global && occurrence > substitution.occurrence
		if !selected {
			continue
		}

		result.WriteString(e.pattern[previousEnd:match[0]])
		result.WriteString(expandReplacement(e.pattern, match, substitution.replacement))
		previousEnd = match[1]
		replaced = true
	}

	if !replaced {
		return
	}

	result.WriteString(e.pattern[previousEnd:])
	e.pattern = result.String()
	e.substituted = true

	if substitution.print {
		e.printPattern(false)
	}
}

// expandReplacement builds the replacement of one match.
func expandReplacement(subject string, match []int, parts []replacementPart) string {
	var expanded strings.Builder

	for _, part := range parts {
		if part.literal != "" {
			expanded.WriteString(part.literal)

			continue
		}

		start, end := 2*part.group, 2*part.group+1
		if end < len(match) && match[start] >= 0 {
			expanded.WriteString(subject[match[start]:match[end]])
		}
	}

	return expanded.String()
}
