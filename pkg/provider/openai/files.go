package openai

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/urlpath"
	"github.com/aarock1234/ai/pkg/model"
)

// Upload form values.
const (
	defaultFilePurpose = "assistants"
	defaultUploadName  = "blob"
	expiryAnchorCreate = "created_at"
)

var (
	_ model.Files              = (*Files)(nil)
	_ model.FileMetadataReader = (*Files)(nil)
	_ model.FileDownloader     = (*Files)(nil)
	_ model.FileDeleter        = (*Files)(nil)
	_ model.FilesProvider      = (*Provider)(nil)
)

// FilesOptions are the upload settings read from the "openai" provider
// options namespace.
type FilesOptions struct {
	// Purpose is the intended use of the file. The default is "assistants".
	Purpose string `json:"purpose,omitempty"`

	// ExpiresAfter is the number of seconds after creation before the file
	// expires.
	ExpiresAfter *int `json:"expiresAfter,omitempty"`
}

// FileMetadata is the provider metadata of an uploaded file, stored under
// the "openai" namespace. Fields OpenAI did not report are omitted.
type FileMetadata struct {
	Filename  string `json:"filename,omitempty"`
	Purpose   string `json:"purpose,omitempty"`
	Bytes     *int64 `json:"bytes,omitempty"`
	CreatedAt *int64 `json:"createdAt,omitempty"`
	Status    string `json:"status,omitempty"`
	ExpiresAt *int64 `json:"expiresAt,omitempty"`
}

// Files uploads, inspects, downloads and deletes files with the OpenAI
// Files API. Uploads accept model.FileStream content without buffering it.
type Files struct {
	config config
}

// fileResponse is a file object.
type fileResponse struct {
	ID        string  `json:"id"`
	Object    *string `json:"object"`
	Bytes     *int64  `json:"bytes"`
	CreatedAt *int64  `json:"created_at"`
	Filename  *string `json:"filename"`
	Purpose   *string `json:"purpose"`
	Status    *string `json:"status"`
	ExpiresAt *int64  `json:"expires_at"`
}

// fileDeleteResponse is the outcome of a deletion.
type fileDeleteResponse struct {
	ID      string  `json:"id"`
	Object  *string `json:"object"`
	Deleted *bool   `json:"deleted"`
}

// Provider returns the provider identifier, such as "openai.files".
func (f *Files) Provider() string {
	return f.config.modelProvider("files")
}

// UploadFile uploads a file. Stream content is sent as it is read.
func (f *Files) UploadFile(ctx context.Context, options model.UploadFileOptions) (*model.UploadFileResult, error) {
	filesOptions, _, err := model.ProviderValue[FilesOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	request, err := f.config.request(ctx, "/files", options.Headers)
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

	filename := valueOr(file.Filename, options.Filename)

	return &model.UploadFileResult{
		ProviderReference: model.ProviderReference{
			Namespace: file.ID,
		},
		MediaType: options.MediaType,
		Filename:  filename,
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
	request, err := f.fileRequest(ctx, options, "")
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
		Filename:  valueOr(file.Filename, ""),
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
	request, err := f.fileRequest(ctx, options, "/content")
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
	request, err := f.fileRequest(ctx, options, "")
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
func (f *Files) fileRequest(ctx context.Context, options model.FileOptions, suffix string) (apicall.Request, error) {
	fileID := options.File[Namespace]
	if strings.TrimSpace(fileID) == "" {
		return apicall.Request{}, fmt.Errorf("%w: file: file reference is missing an 'openai' file id", model.ErrInvalidArgument)
	}

	return f.config.request(ctx, "/files/"+urlpath.Escape(fileID)+suffix, options.Headers)
}

// sendUpload posts the multipart upload, streaming stream content.
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

	return apicall.PostForm(ctx, request, append([]apicall.FormField{file}, fields...))
}

// uploadFields returns the purpose and the optional expiry, anchored at the
// creation time.
func uploadFields(options FilesOptions) []apicall.FormField {
	purpose := options.Purpose
	if purpose == "" {
		purpose = defaultFilePurpose
	}

	fields := []apicall.FormField{
		{
			Name:  "purpose",
			Value: purpose,
		},
	}

	if options.ExpiresAfter == nil {
		return fields
	}

	return append(fields,
		apicall.FormField{
			Name:  "expires_after[anchor]",
			Value: expiryAnchorCreate,
		},
		apicall.FormField{
			Name:  "expires_after[seconds]",
			Value: strconv.Itoa(*options.ExpiresAfter),
		},
	)
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

// metadata reports the fields OpenAI returned.
func (r fileResponse) metadata() FileMetadata {
	return FileMetadata{
		Filename:  valueOr(r.Filename, ""),
		Purpose:   valueOr(r.Purpose, ""),
		Bytes:     r.Bytes,
		CreatedAt: r.CreatedAt,
		Status:    valueOr(r.Status, ""),
		ExpiresAt: r.ExpiresAt,
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

func valueOr(value *string, fallback string) string {
	if value == nil || *value == "" {
		return fallback
	}

	return *value
}
