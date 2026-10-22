package partialjson

import "strings"

// scanState is a position in the JSON grammar.
type scanState int

const (
	stateRoot scanState = iota
	stateFinish
	stateString
	stateStringEscape
	stateStringUnicodeEscape
	stateLiteral
	stateNumber
	stateObjectStart
	stateObjectKey
	stateObjectAfterKey
	stateObjectBeforeValue
	stateObjectAfterValue
	stateObjectAfterComma
	stateArrayStart
	stateArrayAfterValue
	stateArrayAfterComma
)

// Fix completes a JSON prefix in one linear pass. It drops a trailing
// incomplete token, then closes open strings, literals, arrays and objects.
// Invalid JSON is not detected; the caller validates the result.
func Fix(input string) string {
	f := fixer{
		input: input,
		stack: []scanState{
			stateRoot,
		},
	}

	for index, character := range input {
		f.scan(index, index+len(string(character)), character)
	}

	return f.complete()
}

// fixer scans the input and records where the last complete token ends.
type fixer struct {
	input string
	stack []scanState

	// validEnd is the byte offset just past the last token kept in the output.
	validEnd int

	literalStart        int
	unicodeEscapeDigits int
}

func (f *fixer) current() scanState {
	return f.stack[len(f.stack)-1]
}

func (f *fixer) push(state scanState) {
	f.stack = append(f.stack, state)
}

func (f *fixer) pop() {
	f.stack = f.stack[:len(f.stack)-1]
}

// replace swaps the current state for another one.
func (f *fixer) replace(state scanState) {
	f.pop()
	f.push(state)
}

// scan applies one character, spanning input[start:end], to the current
// state. Characters after the root value is complete are ignored.
func (f *fixer) scan(start, end int, character rune) {
	switch f.current() {
	case stateRoot:
		f.valueStart(start, end, character, stateFinish)

	case stateObjectStart,
		stateObjectAfterComma,
		stateObjectKey,
		stateObjectAfterKey,
		stateObjectBeforeValue,
		stateObjectAfterValue:
		f.scanObject(start, end, character)

	case stateArrayStart, stateArrayAfterValue, stateArrayAfterComma:
		f.scanArray(start, end, character)

	case stateString:
		f.scanString(end, character)

	case stateStringEscape:
		f.scanStringEscape(end, character)

	case stateStringUnicodeEscape:
		f.scanUnicodeEscape(end, character)

	case stateNumber:
		f.scanNumber(end, character)

	case stateLiteral:
		f.scanLiteral(end, character)
	}
}

// scanObject applies a character inside an object, outside its keys and
// values.
func (f *fixer) scanObject(start, end int, character rune) {
	switch f.current() {
	case stateObjectStart:
		switch character {
		case '"':
			f.replace(stateObjectKey)
		case '}':
			f.validEnd = end
			f.pop()
		}

	case stateObjectAfterComma:
		if character == '"' {
			f.replace(stateObjectKey)
		}

	case stateObjectKey:
		if character == '"' {
			f.replace(stateObjectAfterKey)
		}

	case stateObjectAfterKey:
		if character == ':' {
			f.replace(stateObjectBeforeValue)
		}

	case stateObjectBeforeValue:
		f.valueStart(start, end, character, stateObjectAfterValue)

	case stateObjectAfterValue:
		f.afterObjectValue(end, character)
	}
}

// scanArray applies a character inside an array, outside its values.
func (f *fixer) scanArray(start, end int, character rune) {
	switch f.current() {
	case stateArrayStart:
		f.validEnd = end

		if character == ']' {
			f.pop()

			return
		}

		f.valueStart(start, end, character, stateArrayAfterValue)

	case stateArrayAfterValue:
		switch character {
		case ',':
			f.replace(stateArrayAfterComma)
		case ']':
			f.validEnd = end
			f.pop()
		default:
			f.validEnd = end
		}

	case stateArrayAfterComma:
		f.valueStart(start, end, character, stateArrayAfterValue)
	}
}

// valueStart begins a value. The current state becomes next, which applies
// once the value is complete.
func (f *fixer) valueStart(start, end int, character rune, next scanState) {
	switch {
	case character == '"':
		f.validEnd = end
		f.replace(next)
		f.push(stateString)

	case character == 'f' || character == 't' || character == 'n':
		f.validEnd = end
		f.literalStart = start
		f.replace(next)
		f.push(stateLiteral)

	case character == '-':
		f.replace(next)
		f.push(stateNumber)

	case character >= '0' && character <= '9':
		f.validEnd = end
		f.replace(next)
		f.push(stateNumber)

	case character == '{':
		f.validEnd = end
		f.replace(next)
		f.push(stateObjectStart)

	case character == '[':
		f.validEnd = end
		f.replace(next)
		f.push(stateArrayStart)
	}
}

func (f *fixer) afterObjectValue(end int, character rune) {
	switch character {
	case ',':
		f.replace(stateObjectAfterComma)
	case '}':
		f.validEnd = end
		f.pop()
	}
}

func (f *fixer) afterArrayValue(end int, character rune) {
	switch character {
	case ',':
		f.replace(stateArrayAfterComma)
	case ']':
		f.validEnd = end
		f.pop()
	}
}

// afterValue applies the delimiter that ended a number or literal to the
// enclosing object or array.
func (f *fixer) afterValue(end int, character rune) {
	switch f.current() {
	case stateObjectAfterValue:
		f.afterObjectValue(end, character)
	case stateArrayAfterValue:
		f.afterArrayValue(end, character)
	}
}

func (f *fixer) scanString(end int, character rune) {
	switch character {
	case '"':
		f.pop()
		f.validEnd = end
	case '\\':
		f.push(stateStringEscape)
	default:
		f.validEnd = end
	}
}

// scanStringEscape applies the character after a backslash. A unicode
// escape is kept only once its four hex digits are complete.
func (f *fixer) scanStringEscape(end int, character rune) {
	f.pop()

	if character != 'u' {
		f.validEnd = end

		return
	}

	f.unicodeEscapeDigits = 0
	f.push(stateStringUnicodeEscape)
}

func (f *fixer) scanUnicodeEscape(end int, character rune) {
	if !isHexDigit(character) {
		return
	}

	f.unicodeEscapeDigits++
	if f.unicodeEscapeDigits == 4 {
		f.pop()
		f.validEnd = end
	}
}

func (f *fixer) scanNumber(end int, character rune) {
	switch {
	case character >= '0' && character <= '9':
		f.validEnd = end

	case character == 'e' || character == 'E' || character == '-' || character == '.':
		// An exponent or fraction is complete only once a digit follows.

	case character == ',' || character == '}' || character == ']':
		f.pop()
		f.afterValue(end, character)

	default:
		f.pop()
	}
}

func (f *fixer) scanLiteral(end int, character rune) {
	partial := f.input[f.literalStart:end]
	if isLiteralPrefix(partial) {
		f.validEnd = end

		return
	}

	f.pop()
	f.afterValue(end, character)
}

// complete returns the kept prefix followed by the closing tokens of every
// open state, innermost first.
func (f *fixer) complete() string {
	var result strings.Builder
	result.WriteString(f.input[:f.validEnd])

	for index := len(f.stack) - 1; index >= 0; index-- {
		switch f.stack[index] {
		case stateString:
			result.WriteByte('"')

		case stateObjectKey,
			stateObjectAfterKey,
			stateObjectAfterComma,
			stateObjectStart,
			stateObjectBeforeValue,
			stateObjectAfterValue:
			result.WriteByte('}')

		case stateArrayStart, stateArrayAfterComma, stateArrayAfterValue:
			result.WriteByte(']')

		case stateLiteral:
			result.WriteString(literalRemainder(f.input[f.literalStart:]))
		}
	}

	return result.String()
}

var literals = []string{
	"true",
	"false",
	"null",
}

func isLiteralPrefix(partial string) bool {
	for _, literal := range literals {
		if strings.HasPrefix(literal, partial) {
			return true
		}
	}

	return false
}

// literalRemainder returns the characters that complete a partial literal.
func literalRemainder(partial string) string {
	for _, literal := range literals {
		if strings.HasPrefix(literal, partial) {
			return literal[len(partial):]
		}
	}

	return ""
}

func isHexDigit(character rune) bool {
	switch {
	case character >= '0' && character <= '9':
		return true
	case character >= 'a' && character <= 'f':
		return true
	default:
		return character >= 'A' && character <= 'F'
	}
}
