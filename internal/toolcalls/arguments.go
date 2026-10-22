package toolcalls

import (
	"strings"
	"unicode"
)

// argumentStructure classifies streamed tool call arguments.
type argumentStructure int

const (
	// argumentsUndetermined means only whitespace has arrived.
	argumentsUndetermined argumentStructure = iota

	// argumentsStructured means the arguments start with an object or array.
	argumentsStructured

	// argumentsOther means the arguments are a scalar or malformed.
	argumentsOther
)

// argumentState tracks whether streamed tool call arguments contain a
// complete JSON object or array. It inspects structure rather than parsing,
// because a parsable scalar can still be the prefix of a longer value.
type argumentState struct {
	structure argumentStructure
	stack     []rune
	inString  bool
	escaped   bool
	complete  bool
}

// hasCompleteStructuredValue reports whether a whole object or array has
// arrived.
func (s *argumentState) hasCompleteStructuredValue() bool {
	return s.structure == argumentsStructured && s.complete
}

// append consumes the next piece of the arguments.
func (s *argumentState) append(delta string) {
	for _, character := range delta {
		s.consume(character)
	}
}

func (s *argumentState) consume(character rune) {
	switch {
	case s.structure == argumentsUndetermined:
		s.start(character)

	case s.structure != argumentsStructured || s.complete:
		return

	case s.inString:
		s.consumeString(character)

	default:
		s.consumeStructure(character)
	}
}

// start classifies the arguments by their first non-whitespace character.
func (s *argumentState) start(character rune) {
	if unicode.IsSpace(character) {
		return
	}

	if character != '{' && character != '[' {
		s.structure = argumentsOther

		return
	}

	s.structure = argumentsStructured
	s.stack = []rune{
		character,
	}
}

func (s *argumentState) consumeString(character rune) {
	switch {
	case s.escaped:
		s.escaped = false
	case character == '\\':
		s.escaped = true
	case character == '"':
		s.inString = false
	}
}

func (s *argumentState) consumeStructure(character rune) {
	switch character {
	case '"':
		s.inString = true

	case '{', '[':
		s.stack = append(s.stack, character)

	case '}', ']':
		s.close(character)
	}
}

// close pops the matching opening bracket. A mismatched bracket makes the
// arguments malformed.
func (s *argumentState) close(character rune) {
	opening := '{'
	if character == ']' {
		opening = '['
	}

	if len(s.stack) == 0 || s.stack[len(s.stack)-1] != opening {
		s.structure = argumentsOther

		return
	}

	s.stack = s.stack[:len(s.stack)-1]
	if len(s.stack) == 0 {
		s.complete = true
	}
}

// startsWithStructuredValue reports whether value begins, after whitespace,
// with an object or array.
func startsWithStructuredValue(value *string) bool {
	if value == nil {
		return false
	}

	trimmed := strings.TrimLeftFunc(*value, unicode.IsSpace)

	return strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")
}
