package google

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/internal/urlpath"
	"github.com/aarock1234/ai/pkg/model"
)

// Default processing wait of uploads.
const (
	defaultFilePollInterval = 2 * time.Second
	defaultFilePollTimeout  = 5 * time.Minute
)

// File states reported while and after a file is processed.
const (
	fileStateProcessing = "PROCESSING"
	fileStateFailed     = "FAILED"
)

// fileNamePattern matches file resource names, such as "files/abc-123".
var fileNamePattern = regexp.MustCompile(`^files/([^/]+)$`)

// ErrFileUploadFailed reports an upload without an upload URL, a file whose
// processing failed, or processing that did not finish in time.
var ErrFileUploadFailed = errors.New("google file upload failed")

var (
	_ model.Files         = (*Files)(nil)
	_ model.FilesProvider = (*Provider)(nil)
)

// FilesOptions are the upload settings read from the "google" provider
// options namespace.
type FilesOptions struct {
	// DisplayName is the file's human-readable name.
	DisplayName *string `json:"displayName,omitempty"`

	// PollIntervalMs is the time between processing checks, 2000 by
	// default.
	PollIntervalMs *float64 `json:"pollIntervalMs,omitempty"`

	// PollTimeoutMs is the longest wait for processing, 300000 by default.
	PollTimeoutMs *float64 `json:"pollTimeoutMs,omitempty"`
}

// FileMetadata is the provider metadata of an uploaded file, stored under
// the "google" namespace.
type FileMetadata struct {
	Name        string  `json:"name"`
	DisplayName *string `json:"displayName,omitempty"`
	MimeType    string  `json:"mimeType"`
	SizeBytes   *string `json:"sizeBytes,omitempty"`
	State       string  `json:"state"`
	URI         string  `json:"uri"`

	CreateTime     *string `json:"createTime,omitempty"`
	UpdateTime     *string `json:"updateTime,omitempty"`
	ExpirationTime *string `json:"expirationTime,omitempty"`
	SHA256Hash     *string `json:"sha256Hash,omitempty"`
}

// Files uploads files with the Gemini API resumable upload protocol and
// waits until they are processed. The file URI is the reference.
type Files struct {
	provider *Provider
}

// fileResource is a Gemini file.
type fileResource struct {
	Name           string  `json:"name"`
	DisplayName    *string `json:"displayName"`
	MimeType       string  `json:"mimeType"`
	SizeBytes      *string `json:"sizeBytes"`
	CreateTime     *string `json:"createTime"`
	UpdateTime     *string `json:"updateTime"`
	ExpirationTime *string `json:"expirationTime"`
	SHA256Hash     *string `json:"sha256Hash"`
	URI            string  `json:"uri"`
	State          string  `json:"state"`
}

// uploadResponse is the response of the finalized upload.
type uploadResponse struct {
	File *fileResource `json:"file"`
}

// Provider returns the provider identifier, the provider name, such as
// "google.generative-ai".
func (f *Files) Provider() string {
	return f.provider.name
}

// UploadFile uploads inline file content and waits while it is processed.
// Gemini ignores filenames, which produce an unsupported warning.
func (f *Files) UploadFile(ctx context.Context, options model.UploadFileOptions) (*model.UploadFileResult, error) {
	filesOptions, err := readFilesOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	var warnings []model.Warning
	if options.Filename != "" {
		warnings = append(warnings, model.UnsupportedWarning("filename", ""))
	}

	content, err := model.InlineFileBytes(options.Data)
	if err != nil {
		return nil, err
	}

	file, err := f.upload(ctx, options, filesOptions, content)
	if err != nil {
		return nil, err
	}

	file, err = f.waitWhileProcessing(ctx, file, filesOptions, options.Headers)
	if err != nil {
		return nil, err
	}

	return &model.UploadFileResult{
		ProviderReference: model.ProviderReference{
			Namespace: file.URI,
		},
		MediaType: file.MimeType,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: file.metadata(),
		},
		Warnings: warnings,
	}, nil
}

// upload sends the content with the resumable upload protocol and checks
// the uploaded file.
func (f *Files) upload(
	ctx context.Context,
	options model.UploadFileOptions,
	filesOptions FilesOptions,
	content []byte,
) (fileResource, error) {
	uploaded, err := uploadResumable[uploadResponse](ctx, f.provider, resumableUpload{
		DisplayName: filesOptions.DisplayName,
		MediaType:   options.MediaType,
		Content:     content,
		Headers:     options.Headers,
	})
	if err != nil {
		return fileResource{}, err
	}

	if uploaded.File == nil {
		return fileResource{}, fmt.Errorf("%w: upload response has no file", model.ErrInvalidResponse)
	}

	if err := uploaded.File.validate(); err != nil {
		return fileResource{}, err
	}

	return *uploaded.File, nil
}

// waitWhileProcessing checks the file until it leaves the processing state
// or the poll timeout passes, and rejects files whose processing failed.
func (f *Files) waitWhileProcessing(
	ctx context.Context,
	file fileResource,
	options FilesOptions,
	headers http.Header,
) (fileResource, error) {
	interval := durationFromMilliseconds(options.PollIntervalMs, defaultFilePollInterval)
	timeout := durationFromMilliseconds(options.PollTimeoutMs, defaultFilePollTimeout)
	started := time.Now()

	for file.State == fileStateProcessing {
		if time.Since(started) > timeout {
			return fileResource{}, fmt.Errorf("%w: file processing timed out after %s", ErrFileUploadFailed, timeout)
		}

		if err := poll.Wait(ctx, interval); err != nil {
			return fileResource{}, context.Cause(ctx)
		}

		current, err := f.fileStatus(ctx, file.Name, headers)
		if err != nil {
			return fileResource{}, err
		}

		file = current
	}

	if file.State == fileStateFailed {
		return fileResource{}, fmt.Errorf("%w: file processing failed for %s", ErrFileUploadFailed, file.Name)
	}

	return file, nil
}

// fileStatus reads a file resource.
func (f *Files) fileStatus(ctx context.Context, name string, headers http.Header) (fileResource, error) {
	request, err := f.provider.request(ctx, filePath(name), headers)
	if err != nil {
		return fileResource{}, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return fileResource{}, err
	}

	var file fileResource
	if err := apicall.DecodeJSON(response.Body, &file); err != nil {
		return fileResource{}, err
	}

	if err := file.validate(); err != nil {
		return fileResource{}, err
	}

	return file, nil
}

// validate checks that the file has every required field.
func (r fileResource) validate() error {
	if r.Name == "" || r.URI == "" || r.State == "" || r.MimeType == "" {
		return fmt.Errorf("%w: file response has no name, uri, state or mimeType", model.ErrInvalidResponse)
	}

	return nil
}

// filePath escapes the ID of a "files/{id}" resource name, or the whole
// name otherwise.
func filePath(name string) string {
	if match := fileNamePattern.FindStringSubmatch(name); match != nil {
		return "files/" + urlpath.Escape(match[1])
	}

	return urlpath.Escape(name)
}

// readFilesOptions reads and validates the call's Google upload options.
func readFilesOptions(options model.ProviderOptions) (FilesOptions, error) {
	filesOptions, _, err := model.ProviderValue[FilesOptions](options, Namespace)
	if err != nil {
		return FilesOptions{}, err
	}

	if filesOptions.PollIntervalMs != nil && *filesOptions.PollIntervalMs <= 0 {
		return FilesOptions{}, fmt.Errorf("%w: google pollIntervalMs must be positive", model.ErrInvalidArgument)
	}

	if filesOptions.PollTimeoutMs != nil && *filesOptions.PollTimeoutMs <= 0 {
		return FilesOptions{}, fmt.Errorf("%w: google pollTimeoutMs must be positive", model.ErrInvalidArgument)
	}

	return filesOptions, nil
}

// metadata reports the file fields under their Gemini names.
func (r fileResource) metadata() FileMetadata {
	return FileMetadata{
		Name:           r.Name,
		DisplayName:    r.DisplayName,
		MimeType:       r.MimeType,
		SizeBytes:      r.SizeBytes,
		State:          r.State,
		URI:            r.URI,
		CreateTime:     r.CreateTime,
		UpdateTime:     r.UpdateTime,
		ExpirationTime: r.ExpirationTime,
		SHA256Hash:     r.SHA256Hash,
	}
}

func durationFromMilliseconds(milliseconds *float64, fallback time.Duration) time.Duration {
	if milliseconds == nil {
		return fallback
	}

	return time.Duration(*milliseconds * float64(time.Millisecond))
}
