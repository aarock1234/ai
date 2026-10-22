package quiverai

import (
	"encoding/base64"
	"fmt"
	"net/url"
	"regexp"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	maxReferenceBase64Length = 16_777_216
	maxReferenceBytes        = 12_582_912
)

var (
	// schemePattern recognizes absolute URLs such as "https://...".
	schemePattern = regexp.MustCompile(`(?i)^[a-z][a-z\d+.-]*://`)

	// base64DataURLPattern captures the media type and data of a base64
	// data URL.
	base64DataURLPattern = regexp.MustCompile(`(?s)^data:([^;,]+);base64,(.+)$`)
)

// referenceMediaTypes are the accepted reference image types.
var referenceMediaTypes = []string{
	"image/gif",
	"image/jpeg",
	"image/png",
	"image/svg+xml",
	"image/webp",
}

// ImageReference is a reference image: a URL or base64 data without a data
// URL prefix. Exactly one field is set.
type ImageReference struct {
	URL    string `json:"url,omitempty"`
	Base64 string `json:"base64,omitempty"`
}

// PrepareImageReference converts an http or https URL, a base64 data URL or
// plain base64 data into an ImageReference. Data must be a PNG, JPEG, WebP,
// GIF or SVG image of at most 12 MiB.
func PrepareImageReference(input string) (ImageReference, error) {
	if schemePattern.MatchString(input) {
		validated, err := validateImageURL(input)
		if err != nil {
			return ImageReference{}, err
		}

		return ImageReference{
			URL: validated,
		}, nil
	}

	if strings.HasPrefix(input, "data:") {
		match := base64DataURLPattern.FindStringSubmatch(input)
		if match == nil || !slices.Contains(referenceMediaTypes, match[1]) {
			return ImageReference{}, fmt.Errorf(
				"%w: input: QuiverAI reference image data URLs must use base64 encoding and a supported image media type",
				model.ErrInvalidArgument,
			)
		}

		input = match[2]
	}

	if err := validateReferenceBase64(input, "input"); err != nil {
		return ImageReference{}, err
	}

	return ImageReference{
		Base64: input,
	}, nil
}

// PrepareImageReferenceBytes encodes image data as an ImageReference. The
// data must be a PNG, JPEG, WebP, GIF or SVG image of at most 12 MiB.
func PrepareImageReferenceBytes(data []byte) (ImageReference, error) {
	if err := validateReferenceBytes(data, "input"); err != nil {
		return ImageReference{}, err
	}

	return ImageReference{
		Base64: base64.StdEncoding.EncodeToString(data),
	}, nil
}

// validateImageURL returns the normalized URL when it is an http or https
// URL.
func validateImageURL(location string) (string, error) {
	parsed, err := url.Parse(location)
	if err != nil || parsed.Scheme == "" {
		return "", fmt.Errorf("%w: url: QuiverAI image URLs must be valid HTTP or HTTPS URLs", model.ErrInvalidArgument)
	}

	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("%w: url: QuiverAI image URLs must use HTTP or HTTPS", model.ErrInvalidArgument)
	}

	return parsed.String(), nil
}

// validateReferenceBase64 checks the length, encoding and content of base64
// reference data.
func validateReferenceBase64(encoded string, argument string) error {
	if encoded == "" || len(encoded) > maxReferenceBase64Length {
		return fmt.Errorf(
			"%w: %s: QuiverAI reference images must contain 1-%d base64 characters",
			model.ErrInvalidArgument,
			argument,
			maxReferenceBase64Length,
		)
	}

	data, err := dataurl.DecodeBase64(encoded)
	if err != nil {
		return fmt.Errorf("%w: %s: QuiverAI reference image data must be valid base64: %w", model.ErrInvalidArgument, argument, err)
	}

	return validateReferenceBytes(data, argument)
}

// validateReferenceBytes checks the size and image type of reference data.
func validateReferenceBytes(data []byte, argument string) error {
	if len(data) == 0 || len(data) > maxReferenceBytes {
		return fmt.Errorf(
			"%w: %s: QuiverAI reference images must decode to 1-%d bytes",
			model.ErrInvalidArgument,
			argument,
			maxReferenceBytes,
		)
	}

	mediaType := "image/svg+xml"
	if !isSVGDocument(data) {
		mediaType, _ = mediatype.Detect(data, "image")
	}

	if !slices.Contains(referenceMediaTypes, mediaType) {
		return fmt.Errorf(
			"%w: %s: QuiverAI reference images must be PNG, JPEG, WebP, GIF, or SVG data",
			model.ErrInvalidArgument,
			argument,
		)
	}

	return nil
}
