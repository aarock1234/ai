package model

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ProviderReference maps provider names to the provider-specific identifier
// of one uploaded file or skill, such as {"openai": "file-abc123"}. Prompts
// reference an uploaded file with FileReference{IDs: reference}.
type ProviderReference map[string]string

// Files uploads files to a provider so that later calls can reference them
// instead of sending their content again.
//
// Only UploadFile is required. Implementations signal the optional
// operations by also implementing FileMetadataReader, FileDownloader and
// FileDeleter.
type Files interface {
	// Provider returns the provider identifier, such as "openai.files".
	Provider() string

	// UploadFile uploads one file and returns its provider reference.
	UploadFile(ctx context.Context, options UploadFileOptions) (*UploadFileResult, error)
}

// FileMetadataReader is implemented by files interfaces that read the
// metadata of uploaded files.
type FileMetadataReader interface {
	GetFileMetadata(ctx context.Context, options FileOptions) (*FileMetadataResult, error)
}

// FileDownloader is implemented by files interfaces that download the
// content of uploaded files.
type FileDownloader interface {
	DownloadFile(ctx context.Context, options FileOptions) (*DownloadFileResult, error)
}

// FileDeleter is implemented by files interfaces that delete uploaded files.
type FileDeleter interface {
	DeleteFile(ctx context.Context, options FileOptions) (*DeleteFileResult, error)
}

// FilesProvider is implemented by providers that offer a files interface.
// It is separate from Provider because most providers have none.
type FilesProvider interface {
	Files() Files
}

// UploadFileData is the content of an upload: FileBytes, FileText or
// FileStream.
type UploadFileData interface {
	isUploadFileData()
}

// InlineFileData is upload content held in memory: FileBytes or FileText.
type InlineFileData interface {
	UploadFileData

	isInlineFileData()
}

// FileStream is upload content read from Reader while the request is sent,
// without buffering the whole file. Providers that cannot stream uploads
// return an error wrapping ErrUnsupported without reading it. The caller
// owns Reader: it is read at most once and never closed by the provider.
type FileStream struct {
	Reader io.Reader
}

func (FileBytes) isUploadFileData()  {}
func (FileText) isUploadFileData()   {}
func (FileStream) isUploadFileData() {}

func (FileBytes) isInlineFileData() {}
func (FileText) isInlineFileData()  {}

// InlineFileBytes returns the bytes of inline upload content: the data of
// FileBytes or the UTF-8 text of FileText. A FileStream, or any other
// content, returns an error wrapping ErrUnsupported.
func InlineFileBytes(data UploadFileData) ([]byte, error) {
	switch typed := data.(type) {
	case FileBytes:
		return typed.Data, nil

	case FileText:
		return []byte(typed.Text), nil

	case FileStream:
		return nil, fmt.Errorf("%w: streaming file upload", ErrUnsupported)

	default:
		return nil, fmt.Errorf("%w: file upload data %T", ErrUnsupported, data)
	}
}

// UploadFileOptions are the inputs of one upload.
type UploadFileOptions struct {
	Data UploadFileData

	// MediaType is the IANA media type of the file, such as
	// "application/pdf".
	MediaType string

	// Filename is optional. Multipart uploads send "blob" without it.
	Filename string

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// UploadFileResult is the outcome of one upload. Empty fields were not
// reported by the provider.
type UploadFileResult struct {
	// ProviderReference holds the provider's identifier under its canonical
	// name, such as "openai", which may differ from Provider().
	ProviderReference ProviderReference

	MediaType string
	Filename  string

	// ByteSize is nil when the provider did not report the size.
	ByteSize *int64

	CreatedAt time.Time

	// ExpiresAt is when the provider deletes the file.
	ExpiresAt time.Time

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
}

// FileOptions identify one uploaded file for a metadata read, download or
// deletion.
type FileOptions struct {
	// File is the reference returned by UploadFile.
	File ProviderReference

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// FileMetadataResult is the metadata of an uploaded file. Empty fields were
// not reported by the provider.
type FileMetadataResult struct {
	// ProviderReference holds only this provider's entry; do not replace a
	// merged multi-provider reference with it.
	ProviderReference ProviderReference

	Filename  string
	MediaType string

	// ByteSize is nil when the provider did not report the size.
	ByteSize *int64

	CreatedAt time.Time
	ExpiresAt time.Time

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
}

// DownloadFileResult is the content of an uploaded file.
type DownloadFileResult struct {
	// Content streams the file. The caller must close it.
	Content io.ReadCloser

	// MediaType is empty when the provider did not report it.
	MediaType string

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
}

// DeleteFileResult is the outcome of a deletion.
type DeleteFileResult struct {
	// ProviderReference holds only this provider's entry.
	ProviderReference ProviderReference

	// Deleted reports whether the provider confirmed the deletion.
	Deleted bool

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
}
