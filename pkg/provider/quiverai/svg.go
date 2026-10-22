package quiverai

import (
	"regexp"
	"strings"
	"unicode/utf8"
)

// svgHeadBytes is how much of a file is inspected to recognize SVG.
const svgHeadBytes = 4096

var (
	// svgHeadPattern matches an SVG root element after optional XML
	// declarations, comments and a doctype.
	svgHeadPattern = regexp.MustCompile(`(?i)^(?:(?:<\?xml[\s\S]*?\?>|<!--[\s\S]*?-->|<!DOCTYPE[\s\S]*?>)\s*)*<svg(?:\s|>)`)

	// svgDocumentStartPattern and the end patterns recognize a complete SVG
	// document in reference images.
	svgDocumentStartPattern = regexp.MustCompile(`(?i)^(?:<\?xml[\s\S]*?\?>\s*)?(?:<!--[\s\S]*?-->\s*)?(?:<!DOCTYPE[\s\S]*?>\s*)?<svg[\s>]`)
	svgClosingTagPattern    = regexp.MustCompile(`(?i)</svg>\s*$`)
	svgSelfClosingPattern   = regexp.MustCompile(`(?is)<svg(?:\s[^>]*)?/>\s*$`)

	numericReferencePattern = regexp.MustCompile(`^#\d+$`)
	hexReferencePattern     = regexp.MustCompile(`^#x[\dA-Fa-f]+$`)
	namedReferencePattern   = regexp.MustCompile(`^[A-Z_a-z:\x{0080}-\x{FFFF}][-.0-9A-Z_a-z:\x{00B7}\x{0080}-\x{FFFF}]*$`)
)

// hasSVGHead reports whether data starts like an SVG document.
func hasSVGHead(data []byte) bool {
	head := strings.ToValidUTF8(string(data[:min(len(data), svgHeadBytes)]), "\uFFFD")

	return svgHeadPattern.MatchString(strings.TrimLeft(head, " \t\n\r\f\v\u00a0\ufeff"))
}

// isSVGDocument reports whether data is UTF-8 text holding a complete SVG
// document.
func isSVGDocument(data []byte) bool {
	if !utf8.Valid(data) {
		return false
	}

	normalized := strings.TrimSpace(strings.TrimPrefix(string(data), "\uFEFF"))

	return svgDocumentStartPattern.MatchString(normalized) &&
		(svgClosingTagPattern.MatchString(normalized) || svgSelfClosingPattern.MatchString(normalized))
}

// svgScanner checks that a document is well-formed XML with an svg root
// element.
type svgScanner struct {
	document   []rune
	position   int
	elements   []string
	rootSeen   bool
	rootClosed bool
	doctype    bool
}

// isSVGMarkup reports whether svg is a single well-formed XML document whose
// root element is svg. It checks element nesting, attribute syntax,
// character references, comments, CDATA sections and the doctype.
func isSVGMarkup(svg string) bool {
	scanner := svgScanner{
		document: []rune(strings.TrimPrefix(svg, "\uFEFF")),
	}

	for scanner.position < len(scanner.document) {
		if !scanner.step() {
			return false
		}
	}

	return scanner.rootSeen && scanner.rootClosed && len(scanner.elements) == 0
}

// step consumes the next construct and reports whether it is valid.
func (s *svgScanner) step() bool {
	switch {
	case s.document[s.position] != '<':
		return s.text()
	case s.hasPrefix("<!--"):
		return s.comment()
	case s.hasPrefix("<?"):
		return s.instruction()
	case s.hasPrefix("<![CDATA["):
		return s.cdata()
	case strings.EqualFold(s.slice(s.position, s.position+9), "<!DOCTYPE"):
		return s.doctypeDeclaration()
	case s.hasPrefix("<!"):
		return false
	case s.hasPrefix("</"):
		return s.closingTag()
	default:
		return s.openingTag()
	}
}

func (s *svgScanner) text() bool {
	end := s.indexOf("<", s.position)
	if end == -1 {
		end = len(s.document)
	}

	text := s.slice(s.position, end)
	if len(s.elements) == 0 && strings.TrimSpace(text) != "" {
		return false
	}

	if strings.Contains(text, "]]>") || !hasValidXMLReferences(text) {
		return false
	}

	s.position = end

	return true
}

func (s *svgScanner) comment() bool {
	end := s.indexOf("-->", s.position+4)
	if end == -1 || strings.Contains(s.slice(s.position+4, end), "--") {
		return false
	}

	s.position = end + 3

	return true
}

func (s *svgScanner) instruction() bool {
	end := s.indexOf("?>", s.position+2)
	if end == -1 {
		return false
	}

	s.position = end + 2

	return true
}

func (s *svgScanner) cdata() bool {
	if len(s.elements) == 0 {
		return false
	}

	end := s.indexOf("]]>", s.position+9)
	if end == -1 {
		return false
	}

	s.position = end + 3

	return true
}

func (s *svgScanner) doctypeDeclaration() bool {
	if s.rootSeen || s.doctype || len(s.elements) > 0 || !isXMLWhitespace(s.at(s.position+9)) {
		return false
	}

	name, nameEnd, ok := s.readName(s.skipWhitespace(s.position + 9))
	if !ok || strings.ToLower(name) != "svg" {
		return false
	}

	end := s.doctypeEnd(nameEnd)
	if end == -1 {
		return false
	}

	s.doctype = true
	s.position = end

	return true
}

func (s *svgScanner) closingTag() bool {
	name, nameEnd, ok := s.readName(s.position + 2)
	if !ok {
		return false
	}

	end := s.skipWhitespace(nameEnd)
	if s.at(end) != '>' {
		return false
	}

	if len(s.elements) == 0 || s.elements[len(s.elements)-1] != name {
		return false
	}

	s.elements = s.elements[:len(s.elements)-1]
	if len(s.elements) == 0 {
		s.rootClosed = true
	}

	s.position = end + 1

	return true
}

func (s *svgScanner) openingTag() bool {
	if s.rootClosed {
		return false
	}

	name, nameEnd, ok := s.readName(s.position + 1)
	if !ok {
		return false
	}

	if !s.rootSeen {
		if strings.ToLower(name) != "svg" {
			return false
		}

		s.rootSeen = true
	}

	return s.attributes(name, nameEnd)
}

// attributes consumes the attributes and the end of an opening tag.
func (s *svgScanner) attributes(name string, position int) bool {
	seen := make(map[string]bool)

	for position < len(s.document) {
		beforeWhitespace := position
		position = s.skipWhitespace(position)

		if s.hasPrefixAt("/>", position) {
			if len(s.elements) == 0 {
				s.rootClosed = true
			}

			s.position = position + 2

			return true
		}

		if s.at(position) == '>' {
			s.elements = append(s.elements, name)
			s.position = position + 1

			return true
		}

		if position == beforeWhitespace {
			return false
		}

		next, ok := s.attribute(position, seen)
		if !ok {
			return false
		}

		position = next
	}

	return false
}

// attribute consumes one name="value" pair and returns the position after
// it.
func (s *svgScanner) attribute(position int, seen map[string]bool) (int, bool) {
	name, nameEnd, ok := s.readName(position)
	if !ok || seen[name] {
		return 0, false
	}

	seen[name] = true

	position = s.skipWhitespace(nameEnd)
	if s.at(position) != '=' {
		return 0, false
	}

	position = s.skipWhitespace(position + 1)

	quote := s.at(position)
	if quote != '"' && quote != '\'' {
		return 0, false
	}

	valueEnd := s.indexOf(string(quote), position+1)
	if valueEnd == -1 {
		return 0, false
	}

	value := s.slice(position+1, valueEnd)
	if strings.Contains(value, "<") || !hasValidXMLReferences(value) {
		return 0, false
	}

	return valueEnd + 1, true
}

// readName reads an XML name at position and returns it with the position
// after it.
func (s *svgScanner) readName(position int) (string, int, bool) {
	if !isXMLNameStart(s.at(position)) {
		return "", 0, false
	}

	start := position
	position++

	for isXMLNameCharacter(s.at(position)) {
		position++
	}

	return s.slice(start, position), position, true
}

func (s *svgScanner) skipWhitespace(position int) int {
	for isXMLWhitespace(s.at(position)) {
		position++
	}

	return position
}

// doctypeEnd returns the position after the doctype's closing bracket,
// skipping quoted strings and the internal subset, or -1.
func (s *svgScanner) doctypeEnd(position int) int {
	subsetDepth := 0
	var quote rune

	for ; position < len(s.document); position++ {
		character := s.document[position]

		switch {
		case quote != 0:
			if character == quote {
				quote = 0
			}
		case character == '"' || character == '\'':
			quote = character
		case character == '[':
			subsetDepth++
		case character == ']':
			if subsetDepth == 0 {
				return -1
			}

			subsetDepth--
		case character == '>' && subsetDepth == 0:
			return position + 1
		}
	}

	return -1
}

// at returns the rune at position, or zero past the end.
func (s *svgScanner) at(position int) rune {
	if position < 0 || position >= len(s.document) {
		return 0
	}

	return s.document[position]
}

// slice returns the runes from start to end, clamped to the document.
func (s *svgScanner) slice(start int, end int) string {
	end = min(end, len(s.document))
	if start >= end {
		return ""
	}

	return string(s.document[start:end])
}

func (s *svgScanner) hasPrefix(prefix string) bool {
	return s.hasPrefixAt(prefix, s.position)
}

func (s *svgScanner) hasPrefixAt(prefix string, position int) bool {
	expected := []rune(prefix)

	return s.slice(position, position+len(expected)) == prefix
}

// indexOf returns the rune index of substring at or after from, or -1.
func (s *svgScanner) indexOf(substring string, from int) int {
	needle := []rune(substring)

	for index := max(from, 0); index+len(needle) <= len(s.document); index++ {
		if string(s.document[index:index+len(needle)]) == substring {
			return index
		}
	}

	return -1
}

func isXMLWhitespace(character rune) bool {
	return character == '\t' || character == '\n' || character == '\r' || character == ' '
}

func isXMLNameStart(character rune) bool {
	return character >= 'A' && character <= 'Z' ||
		character >= 'a' && character <= 'z' ||
		character == '_' ||
		character == ':' ||
		character >= 0x80
}

func isXMLNameCharacter(character rune) bool {
	return isXMLNameStart(character) ||
		character >= '0' && character <= '9' ||
		character == '-' ||
		character == '.' ||
		character == 0xB7
}

// hasValidXMLReferences reports whether every "&" starts a numeric,
// hexadecimal or named character reference ending with ";".
func hasValidXMLReferences(value string) bool {
	for {
		start := strings.IndexByte(value, '&')
		if start == -1 {
			return true
		}

		length := strings.IndexByte(value[start+1:], ';')
		if length == -1 {
			return false
		}

		reference := value[start+1 : start+1+length]
		if !numericReferencePattern.MatchString(reference) &&
			!hexReferencePattern.MatchString(reference) &&
			!namedReferencePattern.MatchString(reference) {
			return false
		}

		value = value[start+1+length+1:]
	}
}
