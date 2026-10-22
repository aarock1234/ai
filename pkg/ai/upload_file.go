package ai

import (
	"context"
	"fmt"
	"net/http"
	"slices"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// textSniffBytes is how much of a file is inspected to decide whether
// content without a detectable signature is text.
const textSniffBytes = 512

// UploadFileOptions configures UploadFile.
type UploadFileOptions struct {
	// API is the files interface, such as the result of a provider's Files
	// method or Registry.Files.
	API model.Files

	// Data is model.FileBytes, model.FileText or model.FileStream. The
	// caller owns a stream's reader and closes it after the call.
	Data model.UploadFileData

	// MediaType is optional. Without it, the type is detected from the
	// bytes, falling back to "text/plain" for text and
	// "application/octet-stream" for streams and other content.
	MediaType string

	// Filename is optional. Multipart uploads send "blob" without it.
	Filename string

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// LogWarnings receives the warnings of the upload. Nil discards them;
	// they are always returned on the result.
	LogWarnings LogWarningsFunc
}

// UploadFileResult is the outcome of UploadFile. Pass its
// ProviderReference to prompts as model.FileReference{IDs: reference}.
type UploadFileResult = model.UploadFileResult

// UploadFile uploads a file through a provider's files interface, so that
// prompts can reference it instead of sending its content.
func UploadFile(ctx context.Context, options UploadFileOptions) (*UploadFileResult, error) {
	if options.API == nil {
		return nil, &InvalidArgumentError{
			Parameter: "api",
			Message:   "api is required; the provider must offer a files interface",
		}
	}

	if options.Data == nil {
		return nil, &InvalidArgumentError{
			Parameter: "data",
			Message:   "data is required",
		}
	}

	mediaType := options.MediaType
	if mediaType == "" {
		mediaType = detectUploadMediaType(options.Data)
	}

	result, err := options.API.UploadFile(ctx, model.UploadFileOptions{
		Data:            options.Data,
		MediaType:       mediaType,
		Filename:        options.Filename,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	})
	if err != nil {
		return nil, err
	}

	if result == nil {
		return nil, fmt.Errorf("%w: files interface returned no result", model.ErrInvalidResponse)
	}

	options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: result.Warnings,
		Provider: options.API.Provider(),
	})

	return result, nil
}

// detectUploadMediaType detects the media type of bytes from their
// signature, or as text when they look like text. Streams cannot be
// inspected without consuming them.
func detectUploadMediaType(data model.UploadFileData) string {
	switch typed := data.(type) {
	case model.FileText:
		return "text/plain"

	case model.FileBytes:
		if detected, ok := mediatype.Detect(typed.Data, ""); ok {
			return detected
		}

		if isLikelyText(typed.Data) {
			return "text/plain"
		}

		return genericBinaryMediaType

	default:
		return genericBinaryMediaType
	}
}

// isLikelyText reports whether the first bytes contain no NUL and no
// control characters other than tab, line feed and carriage return.
func isLikelyText(data []byte) bool {
	sample := data[:min(len(data), textSniffBytes)]
	if len(sample) == 0 {
		return false
	}

	return !slices.ContainsFunc(sample, isTextControl)
}

// isTextControl reports whether a byte is NUL or a control character that
// text does not contain: any below space except tab, line feed and carriage
// return.
func isTextControl(value byte) bool {
	switch value {
	case '\t', '\n', '\r':
		return false
	default:
		return value < 0x20
	}
}
