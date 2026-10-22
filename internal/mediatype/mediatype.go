// Package mediatype detects file media types from content and resolves the
// full media types of file parts.
package mediatype

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// id3HeaderBytes is the size of an ID3v2 tag header, which may precede MP3
// audio data.
const id3HeaderBytes = 10

// TopLevel returns the part before the slash, such as "image" for
// "image/png".
func TopLevel(mediaType string) string {
	top, _, _ := strings.Cut(mediaType, "/")

	return top
}

// IsFull reports whether mediaType names a concrete subtype rather than a
// wildcard such as "image/*" or a bare top-level type such as "image".
func IsFull(mediaType string) bool {
	_, subtype, found := strings.Cut(mediaType, "/")

	return found && subtype != "" && subtype != "*"
}

// Detect returns the media type whose signature matches data. When topLevel
// is not empty, only signatures of that top-level type are considered.
func Detect(data []byte, topLevel string) (string, bool) {
	data = skipID3Tag(data)

	for _, signature := range mediaSignatures {
		if topLevel != "" && TopLevel(signature.mediaType) != topLevel {
			continue
		}

		if matches(data, signature.prefix) {
			return signature.mediaType, true
		}
	}

	return "", false
}

// Resolve returns mediaType when it is already full, and otherwise detects
// the full type from data within the declared top-level type.
func Resolve(mediaType string, data []byte) (string, bool) {
	if IsFull(mediaType) {
		return mediaType, true
	}

	return Detect(data, TopLevel(mediaType))
}

// skipID3Tag returns the data after an ID3v2 tag, so that the MP3 frame
// signature that follows it can be matched.
func skipID3Tag(data []byte) []byte {
	if len(data) <= id3HeaderBytes || !bytes.HasPrefix(data, []byte("ID3")) {
		return data
	}

	offset := id3HeaderBytes + id3TagSize(data[6:10])
	if offset > len(data) {
		return nil
	}

	return data[offset:]
}

// id3TagSize decodes the tag size, stored as four 7-bit "syncsafe" bytes.
func id3TagSize(size []byte) int {
	value := 0
	for _, part := range size {
		value = value<<7 | int(part&0x7f)
	}

	return value
}

// matches reports whether data starts with prefix. Negative prefix entries
// match any byte.
func matches(data []byte, prefix []int) bool {
	if len(data) < len(prefix) {
		return false
	}

	for index, expected := range prefix {
		if expected >= 0 && int(data[index]) != expected {
			return false
		}
	}

	return true
}

// ResolveFile returns the full media type of a file part, as providers whose
// APIs demand a concrete subtype require. A full declared type is returned
// unchanged; otherwise the subtype is detected from inline bytes within the
// declared top-level type. Files passed by URL, reference or text, and bytes
// whose type cannot be detected, fail with an error wrapping
// model.ErrUnsupported.
func ResolveFile(mediaType string, data model.FileData) (string, error) {
	if IsFull(mediaType) {
		return mediaType, nil
	}

	inline, ok := data.(model.FileBytes)
	if !ok {
		return "", fmt.Errorf(
			"%w: file of media type %q must specify subtype since it is not passed as inline bytes",
			model.ErrUnsupported,
			mediaType,
		)
	}

	detected, ok := Detect(inline.Data, TopLevel(mediaType))
	if !ok {
		return "", fmt.Errorf(
			"%w: file of media type %q must specify subtype since it could not be auto-detected",
			model.ErrUnsupported,
			mediaType,
		)
	}

	return detected, nil
}
