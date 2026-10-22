package ai

import (
	"context"
	"net/url"

	"github.com/aarock1234/ai/internal/download"
)

// DefaultMaxDownloadBytes is the default download size limit: 2 GiB.
const DefaultMaxDownloadBytes = download.DefaultMaxBytes

// DownloadError reports a failed file download.
type DownloadError = download.Error

// DownloadedFile is the content of a downloaded URL.
type DownloadedFile = download.File

// Downloader fetches the content of one URL. It returns a *DownloadError
// when the download fails.
type Downloader func(ctx context.Context, fileURL *url.URL) (*DownloadedFile, error)

// DownloadOptions configures NewDownloader.
type DownloadOptions struct {
	// MaxBytes limits the size of a download. Zero selects
	// DefaultMaxDownloadBytes.
	MaxBytes int64
}

// DownloadRequest is one file URL found in a prompt.
type DownloadRequest struct {
	URL *url.URL

	// IsURLSupportedByModel reports whether the model downloads the URL
	// itself, according to its SupportedURLs.
	IsURLSupportedByModel bool
}

// DownloadFunction resolves the file URLs of a prompt. It returns one entry
// per request, in request order: downloaded content, or nil to pass the URL
// to the model unchanged. The default downloads every URL the model does not
// support and passes the others through.
type DownloadFunction func(ctx context.Context, requests []DownloadRequest) ([]*DownloadedFile, error)

// NewDownloader returns a Downloader that guards against server-side request
// forgery: it accepts only http, https and data URLs, rejects local and
// private hosts, validates every redirect before following it, and checks
// the resolved address of every connection. Data URLs are decoded without a
// network request.
func NewDownloader(options DownloadOptions) Downloader {
	return func(ctx context.Context, fileURL *url.URL) (*DownloadedFile, error) {
		if fileURL == nil {
			return nil, &DownloadError{
				Message: "download URL is missing",
			}
		}

		return download.Fetch(ctx, download.Request{
			URL:      fileURL.String(),
			MaxBytes: options.MaxBytes,
		})
	}
}

// defaultDownloadFunction downloads the URLs that the model does not support,
// concurrently, and passes the supported ones through.
func defaultDownloadFunction(downloader Downloader) DownloadFunction {
	return func(ctx context.Context, requests []DownloadRequest) ([]*DownloadedFile, error) {
		return mapConcurrently(ctx, requests, 0, func(ctx context.Context, request DownloadRequest) (*DownloadedFile, error) {
			if request.IsURLSupportedByModel {
				return nil, nil
			}

			return downloader(ctx, request.URL)
		})
	}
}
