package sandbox

import (
	"fmt"
	"strings"

	"golang.org/x/text/encoding/htmlindex"
)

// DecodeText decodes file bytes with a named text encoding. Empty, "utf-8"
// and "utf8" decode UTF-8; other names are resolved as WHATWG encoding
// labels, such as "utf-16le" or "iso-8859-1". Unknown encodings are an error.
func DecodeText(data []byte, encoding string) (string, error) {
	if isUTF8(encoding) {
		return string(data), nil
	}

	decoder, err := htmlindex.Get(encoding)
	if err != nil {
		return "", fmt.Errorf("unsupported text encoding %q: %w", encoding, err)
	}

	decoded, err := decoder.NewDecoder().Bytes(data)
	if err != nil {
		return "", fmt.Errorf("decode %s text: %w", encoding, err)
	}

	return string(decoded), nil
}

// EncodeText encodes text with a named text encoding, using the same names
// as DecodeText.
func EncodeText(text string, encoding string) ([]byte, error) {
	if isUTF8(encoding) {
		return []byte(text), nil
	}

	encoder, err := htmlindex.Get(encoding)
	if err != nil {
		return nil, fmt.Errorf("unsupported text encoding %q: %w", encoding, err)
	}

	encoded, err := encoder.NewEncoder().Bytes([]byte(text))
	if err != nil {
		return nil, fmt.Errorf("encode %s text: %w", encoding, err)
	}

	return encoded, nil
}

// ExtractLines returns a 1-based inclusive line range of text. It detects the
// line ending ("\r\n", "\n" or "\r", in that priority) and keeps it when
// joining the lines; mixed line endings are not supported. Zero bounds are
// unset: when both are zero the text is returned unchanged, and an end past
// the last line reads through the end.
func ExtractLines(text string, startLine int, endLine int) string {
	if startLine == 0 && endLine == 0 {
		return text
	}

	lineEnding := detectLineEnding(text)
	lines := strings.Split(text, lineEnding)

	start := max(1, startLine) - 1
	end := len(lines)
	if endLine > 0 {
		end = min(len(lines), endLine)
	}

	if start >= end {
		return ""
	}

	return strings.Join(lines[start:end], lineEnding)
}

// isUTF8 reports whether encoding names UTF-8, the default.
func isUTF8(encoding string) bool {
	switch strings.ToLower(encoding) {
	case "", "utf-8", "utf8":
		return true
	default:
		return false
	}
}

// detectLineEnding returns the line ending text uses, defaulting to "\n".
func detectLineEnding(text string) string {
	switch {
	case strings.Contains(text, "\r\n"):
		return "\r\n"
	case strings.Contains(text, "\n"):
		return "\n"
	case strings.Contains(text, "\r"):
		return "\r"
	default:
		return "\n"
	}
}
