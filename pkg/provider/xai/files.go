package xai

import (
	"context"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// filesPath is the Files API endpoint.
const filesPath = "/files"

// defaultUploadName is the multipart filename of uploads without one.
const defaultUploadName = "blob"

// Expiry limits of uploaded files, in seconds.
const (
	minFileExpiresAfter = 3600
	maxFileExpiresAfter = 2592000
)

var (
	_ model.Files              = (*Files)(nil)
	_ model.FileMetadataReader = (*Files)(nil)
	_ model.FileDownloader     = (*Files)(nil)
	_ model.FileDeleter        = (*Files)(nil)
	_ model.FilesProvider      = (*Provider)(nil)
)

// FilesOptions are the upload settings read from the "xai" provider options
// namespace.
type FilesOptions struct {
	// TeamID uploads the file for a team.
	TeamID *string `json:"teamId,omitempty"`

	// FilePath is accepted for compatibility and not sent.
	FilePath *string `json:"filePath,omitempty"`

	// ExpiresAfter is the lifetime in seconds from the upload, between 3600
	// and 2592000. Without it, the file is kept until it is deleted.
	ExpiresAfter *int `json:"expiresAfter,omitempty"`
}

// FileMetadata is the provider metadata of an uploaded file, stored under
// the "xai" namespace. Fields xAI did not report are omitted.
type FileMetadata struct {
	Filename  string `json:"filename,omitempty"`
	Purpose   string `json:"purpose,omitempty"`
	Bytes     *int64 `json:"bytes,omitempty"`
	CreatedAt *int64 `json:"createdAt,omitempty"`
	Status    string `json:"status,omitempty"`
	ExpiresAt *int64 `json:"expiresAt,omitempty"`
}

// Files uploads, inspects, downloads and deletes files with the xAI Files
// API. Uploads accept model.FileStream content without buffering it.
type Files struct {
	provider *Provider
}

// fileResponse is a file object.
type fileResponse struct {
	ID        string  `json:"id"`
	Object    *string `json:"object"`
	Bytes     *int64  `json:"bytes"`
	CreatedAt *int64  `json:"created_at"`
	ExpiresAt *int64  `json:"expires_at"`
	Filename  *string `json:"filename"`
	Purpose   *string `json:"purpose"`
	Status    *string `json:"status"`
}

// fileDeleteResponse is the outcome of a deletion.
type fileDeleteResponse struct {
	ID      string  `json:"id"`
	Object  *string `json:"object"`
	Deleted *bool   `json:"deleted"`
}

// Files returns the Files API.
func (p *Provider) Files() model.Files {
	return &Files{
		provider: p,
	}
}

// Provider returns the provider identifier, "xai.files" by default.
func (f *Files) Provider() string {
	return f.provider.name + ".files"
}

// UploadFile uploads a file. Stream content is sent as it is read.
func (f *Files) UploadFile(ctx context.Context, options model.UploadFileOptions) (*model.UploadFileResult, error) {
	filesOptions, err := readFilesOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	request, err := f.provider.apiRequest(filesPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := sendUpload(ctx, request, options, filesOptions)
	if err != nil {
		return nil, err
	}

	file, err := decodeFileResponse(response.Body)
	if err != nil {
		return nil, err
	}

	return &model.UploadFileResult{
		ProviderReference: model.ProviderReference{
			Namespace: file.ID,
		},
		MediaType: options.MediaType,
		Filename:  stringOr(file.Filename, options.Filename),
		ByteSize:  file.Bytes,
		CreatedAt: unixTime(file.CreatedAt),
		ExpiresAt: unixTime(file.ExpiresAt),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: file.metadata(),
		},
	}, nil
}

// GetFileMetadata reads the metadata of an uploaded file.
func (f *Files) GetFileMetadata(ctx context.Context, options model.FileOptions) (*model.FileMetadataResult, error) {
	request, err := f.fileRequest(options, "")
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, err
	}

	file, err := decodeFileResponse(response.Body)
	if err != nil {
		return nil, err
	}

	return &model.FileMetadataResult{
		ProviderReference: model.ProviderReference{
			Namespace: file.ID,
		},
		Filename:  stringOr(file.Filename, ""),
		ByteSize:  file.Bytes,
		CreatedAt: unixTime(file.CreatedAt),
		ExpiresAt: unixTime(file.ExpiresAt),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: file.metadata(),
		},
	}, nil
}

// DownloadFile streams the content of an uploaded file. The caller closes
// the content.
func (f *Files) DownloadFile(ctx context.Context, options model.FileOptions) (*model.DownloadFileResult, error) {
	request, err := f.fileRequest(options, "/content")
	if err != nil {
		return nil, err
	}

	stream, err := apicall.GetStream(ctx, request)
	if err != nil {
		return nil, err
	}

	return &model.DownloadFileResult{
		Content:   stream.Body,
		MediaType: contentMediaType(stream.Headers.Get("Content-Type")),
	}, nil
}

// DeleteFile deletes an uploaded file.
func (f *Files) DeleteFile(ctx context.Context, options model.FileOptions) (*model.DeleteFileResult, error) {
	request, err := f.fileRequest(options, "")
	if err != nil {
		return nil, err
	}

	response, err := apicall.Delete(ctx, request)
	if err != nil {
		return nil, err
	}

	var deleted fileDeleteResponse
	if err := apicall.DecodeJSON(response.Body, &deleted); err != nil {
		return nil, err
	}

	if deleted.ID == "" || deleted.Deleted == nil {
		return nil, fmt.Errorf("%w: file deletion response has no id or deleted flag", model.ErrInvalidResponse)
	}

	return &model.DeleteFileResult{
		ProviderReference: model.ProviderReference{
			Namespace: deleted.ID,
		},
		Deleted: *deleted.Deleted,
	}, nil
}

// fileRequest prepares a call to the file's endpoint, followed by suffix.
func (f *Files) fileRequest(options model.FileOptions, suffix string) (apicall.Request, error) {
	fileID := options.File[Namespace]
	if strings.TrimSpace(fileID) == "" {
		return apicall.Request{}, fmt.Errorf("%w: file: file reference is missing an 'xai' file id", model.ErrInvalidArgument)
	}

	return f.provider.apiRequest(filesPath+"/"+encodePathSegment(fileID)+suffix, options.Headers)
}

// readFilesOptions decodes and validates the upload options.
func readFilesOptions(options model.ProviderOptions) (FilesOptions, error) {
	filesOptions, _, err := model.ProviderValue[FilesOptions](options, Namespace)
	if err != nil {
		return FilesOptions{}, err
	}

	expiresAfter := filesOptions.ExpiresAfter
	if expiresAfter != nil && (*expiresAfter < minFileExpiresAfter || *expiresAfter > maxFileExpiresAfter) {
		return FilesOptions{}, fmt.Errorf(
			"%w: xai provider options: expiresAfter must be between %d and %d",
			model.ErrInvalidArgument,
			minFileExpiresAfter,
			maxFileExpiresAfter,
		)
	}

	return filesOptions, nil
}

// sendUpload posts the multipart upload, streaming stream content. xAI
// rejects uploads whose fields follow the file, so the file is last.
func sendUpload(
	ctx context.Context,
	request apicall.Request,
	options model.UploadFileOptions,
	filesOptions FilesOptions,
) (*apicall.Response, error) {
	filename := options.Filename
	if filename == "" {
		filename = defaultUploadName
	}

	fields := uploadFields(filesOptions)

	if stream, ok := options.Data.(model.FileStream); ok {
		return apicall.PostFormStream(ctx, request, fields, apicall.FormStream{
			Name:      "file",
			Filename:  filename,
			MediaType: options.MediaType,
			Reader:    stream.Reader,
		})
	}

	content, err := model.InlineFileBytes(options.Data)
	if err != nil {
		return nil, err
	}

	file := apicall.FormField{
		Name:      "file",
		Filename:  filename,
		MediaType: options.MediaType,
		Data:      content,
	}

	return apicall.PostForm(ctx, request, append(fields, file))
}

// uploadFields returns the optional expiry and team.
func uploadFields(options FilesOptions) []apicall.FormField {
	var fields []apicall.FormField

	if options.ExpiresAfter != nil {
		fields = append(fields, apicall.FormField{
			Name:  "expires_after",
			Value: strconv.Itoa(*options.ExpiresAfter),
		})
	}

	if options.TeamID != nil {
		fields = append(fields, apicall.FormField{
			Name:  "team_id",
			Value: *options.TeamID,
		})
	}

	return fields
}

func decodeFileResponse(body []byte) (fileResponse, error) {
	var file fileResponse
	if err := apicall.DecodeJSON(body, &file); err != nil {
		return fileResponse{}, err
	}

	if file.ID == "" {
		return fileResponse{}, fmt.Errorf("%w: file response has no id", model.ErrInvalidResponse)
	}

	return file, nil
}

// metadata reports the fields xAI returned.
func (r fileResponse) metadata() FileMetadata {
	return FileMetadata{
		Filename:  stringOr(r.Filename, ""),
		Purpose:   stringOr(r.Purpose, ""),
		Bytes:     r.Bytes,
		CreatedAt: r.CreatedAt,
		Status:    stringOr(r.Status, ""),
		ExpiresAt: r.ExpiresAt,
	}
}

// encodePathSegment escapes a path segment. Dot segments are escaped twice,
// since URL parsing normalizes both literal and percent-encoded dots.
func encodePathSegment(value string) string {
	switch value {
	case ".":
		return "%252E"
	case "..":
		return "%252E%252E"
	default:
		return url.PathEscape(value)
	}
}

// contentMediaType returns the media type of a Content-Type header without
// its parameters.
func contentMediaType(contentType string) string {
	essence, _, _ := strings.Cut(contentType, ";")

	return strings.TrimSpace(essence)
}

// unixTime converts optional Unix seconds, returning the zero time when
// absent.
func unixTime(seconds *int64) time.Time {
	if seconds == nil {
		return time.Time{}
	}

	return time.Unix(*seconds, 0).UTC()
}

// stringOr returns the value, or fallback when it is nil or empty.
func stringOr(value *string, fallback string) string {
	if value == nil || *value == "" {
		return fallback
	}

	return *value
}
