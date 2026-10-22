package amazonbedrock

import (
	"encoding/base64"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// imageFormats are the image formats Converse accepts, by media type.
var imageFormats = map[string]string{
	"image/jpeg": "jpeg",
	"image/png":  "png",
	"image/gif":  "gif",
	"image/webp": "webp",
}

// documentFormats are the document formats Converse accepts, by media type.
var documentFormats = map[string]string{
	"application/pdf":    "pdf",
	"text/csv":           "csv",
	"application/msword": "doc",
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
	"application/vnd.ms-excel": "xls",
	"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
	"text/html":     "html",
	"text/plain":    "txt",
	"text/markdown": "md",
}

// videoFormats are the video formats Converse accepts, by media type.
var videoFormats = map[string]string{
	"video/x-matroska": "mkv",
	"video/quicktime":  "mov",
	"video/mp4":        "mp4",
	"video/webm":       "webm",
	"video/x-flv":      "flv",
	"video/mpeg":       "mpeg",
	"video/mpg":        "mpg",
	"video/wmv":        "wmv",
	"video/x-ms-wmv":   "wmv",
	"video/3gpp":       "three_gp",
}

var (
	// documentNameWhitespace matches whitespace runs, including the Unicode
	// spaces JavaScript's \s matches.
	documentNameWhitespace = regexp.MustCompile(`[\s\v\p{Zs}\x{FEFF}\x{2028}\x{2029}]+`)

	// documentNameDisallowed matches characters Converse rejects in
	// document names.
	documentNameDisallowed = regexp.MustCompile(`[^a-zA-Z0-9 ()\[\]-]`)

	// toolNameDisallowed matches characters Converse rejects in tool names.
	toolNameDisallowed = regexp.MustCompile(`[^a-zA-Z0-9_-]`)
)

// maxDocumentNameLength is the longest document name Converse accepts.
const maxDocumentNameLength = 200

func imageFormat(mediaType string) (string, error) {
	return formatOf(imageFormats, "image", mediaType)
}

func documentFormat(mediaType string) (string, error) {
	return formatOf(documentFormats, "file", mediaType)
}

func videoFormat(mediaType string) (string, error) {
	return formatOf(videoFormats, "video", mediaType)
}

// formatOf returns the Converse format of mediaType from formats, or an
// unsupported error listing the accepted media types.
func formatOf(formats map[string]string, kind string, mediaType string) (string, error) {
	if format, found := formats[mediaType]; found {
		return format, nil
	}

	accepted := slices.Sorted(maps.Keys(formats))

	return "", fmt.Errorf(
		"%w: unsupported %s mime type: %s, expected one of: %s",
		model.ErrUnsupported,
		kind,
		mediaType,
		strings.Join(accepted, ", "),
	)
}

// mediaSourceOf returns the source of inline bytes or an S3 URL. Other data
// is reported as the unsupported functionality.
func mediaSourceOf(data model.FileData, functionality string) (mediaSource, error) {
	switch data := data.(type) {
	case model.FileBytes:
		return mediaSource{
			Bytes: base64.StdEncoding.EncodeToString(data.Data),
		}, nil

	case model.FileURL:
		if !isS3URL(data) {
			return mediaSource{}, fmt.Errorf("%w: %s", model.ErrUnsupported, functionality)
		}

		return mediaSource{
			S3Location: &s3Location{
				URI: data.URL.String(),
			},
		}, nil

	default:
		return mediaSource{}, fmt.Errorf("%w: %s", model.ErrUnsupported, functionality)
	}
}

func isS3URL(data model.FileURL) bool {
	return data.URL != nil && data.URL.Scheme == "s3"
}

// sanitizeDocumentName keeps the characters Converse accepts in a document
// name, without the file extension.
func sanitizeDocumentName(filename string) string {
	name, _, _ := strings.Cut(filename, ".")
	name = documentNameWhitespace.ReplaceAllString(name, " ")
	name = documentNameDisallowed.ReplaceAllString(name, "")
	name = strings.TrimSpace(name)

	if len(name) > maxDocumentNameLength {
		name = name[:maxDocumentNameLength]
	}

	return strings.TrimSpace(name)
}

// sanitizeToolName keeps the characters Converse accepts in a tool name.
func sanitizeToolName(name string) string {
	sanitized := toolNameDisallowed.ReplaceAllString(name, "")
	if sanitized == "" {
		return "_"
	}

	return sanitized
}
