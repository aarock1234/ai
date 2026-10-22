// Package dataurl encodes and decodes data URLs of the form
// "data:[<media type>][;base64],<data>", and converts inline image and video
// files to URLs that providers accept in place of a fetchable URL.
package dataurl

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net/url"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	scheme       = "data:"
	base64Marker = ";base64"
)

// Is reports whether location has the data scheme, ignoring case.
func Is(location string) bool {
	return hasPrefixFold(location, scheme)
}

// Parse returns the media type and decoded content of a data URL of the
// form "data:[<media type>][;base64],<data>". The scheme and the base64
// marker are matched without regard to case. The media type is empty when
// the URL omits it.
func Parse(location string) (string, []byte, error) {
	if !Is(location) {
		return "", nil, errors.New("not a data URL")
	}

	content := location[len(scheme):]

	header, payload, found := strings.Cut(content, ",")
	if !found {
		return "", nil, errors.New("data URL has no comma separator")
	}

	mediaType, _, _ := strings.Cut(header, ";")
	if !hasSuffixFold(header, base64Marker) {
		decoded, err := url.PathUnescape(payload)
		if err != nil {
			return "", nil, fmt.Errorf("decode data URL text: %w", err)
		}

		return mediaType, []byte(decoded), nil
	}

	data, err := DecodeBase64(payload)
	if err != nil {
		return "", nil, fmt.Errorf("decode data URL base64: %w", err)
	}

	return mediaType, data, nil
}

func hasPrefixFold(value string, prefix string) bool {
	return len(value) >= len(prefix) && strings.EqualFold(value[:len(prefix)], prefix)
}

func hasSuffixFold(value string, suffix string) bool {
	return len(value) >= len(suffix) && strings.EqualFold(value[len(value)-len(suffix):], suffix)
}

// DecodeBase64 accepts standard and URL-safe base64, with or without
// padding.
func DecodeBase64(encoded string) ([]byte, error) {
	normalized := strings.NewReplacer("-", "+", "_", "/").Replace(encoded)
	normalized = strings.TrimRight(normalized, "=")

	return base64.RawStdEncoding.DecodeString(normalized)
}

// Encode returns data as a base64 data URL of the form
// "data:<media type>;base64,<data>".
func Encode(mediaType string, data []byte) string {
	return scheme + mediaType + base64Marker + "," + base64.StdEncoding.EncodeToString(data)
}

// ImageFileURL returns the URL of a URL image, or inline image bytes as a
// data URL, like upstream convertImageModelFileToDataUri.
func ImageFileURL(file model.ImageFile) (string, error) {
	switch typed := file.(type) {
	case model.ImageURL:
		return typed.URL, nil

	case model.ImageBytes:
		return Encode(typed.MediaType, typed.Data), nil

	default:
		return "", fmt.Errorf("%w: image file %T", model.ErrUnsupported, file)
	}
}

// VideoFileURL returns the URL of a URL video input, or inline bytes as a
// data URL.
func VideoFileURL(file model.VideoFile) (string, error) {
	switch typed := file.(type) {
	case model.VideoURL:
		return typed.URL, nil

	case model.VideoBytes:
		return Encode(typed.MediaType, typed.Data), nil

	default:
		return "", fmt.Errorf("%w: video file %T", model.ErrUnsupported, file)
	}
}
