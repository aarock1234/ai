// Package urlpath escapes caller and response values placed in URL paths,
// so that an identifier can never change which resource a request names.
package urlpath

import (
	"net/url"
	"strings"
)

// Escape escapes one path segment. Dot segments are escaped twice, since
// URL parsing normalizes both literal and percent-encoded dots.
func Escape(segment string) string {
	switch segment {
	case ".":
		return "%252E"
	case "..":
		return "%252E%252E"
	default:
		return url.PathEscape(segment)
	}
}

// EscapeEach escapes every slash-separated segment of path and keeps the
// slashes between them.
func EscapeEach(path string) string {
	segments := strings.Split(path, "/")
	for index, segment := range segments {
		segments[index] = Escape(segment)
	}

	return strings.Join(segments, "/")
}
