// Package shellquote quotes strings for POSIX shells.
package shellquote

import "strings"

// Quote returns value as one POSIX shell word: wrapped in single quotes,
// inside which POSIX shells interpret nothing, with each single quote of
// value closing the quotes, escaped, and reopening them.
func Quote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'\''`) + "'"
}
