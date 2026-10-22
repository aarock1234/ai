package deepseek

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// filesPath is the file upload endpoint.
const filesPath = "/files"

// Upload limits and fixed form and response values.
const (
	maxFileBytes       = 64 << 20
	maxFilenameLength  = 512
	userDataPurpose    = "user_data"
	fileObject         = "file"
	defaultUploadName  = "blob"
	expiryAnchorCreate = "created_at"
)

// genericMediaTypes are media types that do not identify the image format.
var genericMediaTypes = map[string]bool{
	"":                         true,
	"application/binary":       true,
	"application/octet-stream": true,
	"binary/octet-stream":      true,
	"image":                    true,
	"image/*":                  true,
}

// supportedExtensions are the image filename extensions DeepSeek accepts.
var supportedExtensions = map[string]bool{
	"gif":  true,
	"jpeg": true,
	"jpg":  true,
	"png":  true,
	"webp": true,
}

var (
	_ model.Files         = (*Files)(nil)
	_ model.FilesProvider = (*Provider)(nil)
)

// Files uploads JPEG, PNG, GIF and WebP images of up to 64 MiB, which chat
// prompts reference with model.FileReference under the "deepseek" key.
type Files struct {
	provider *Provider
}

// Provider returns the provider identifier, "deepseek.files" by default.
func (f *Files) Provider() string {
	return f.provider.name + ".files"
}

// FileMetadata is the provider metadata of an uploaded file, stored under
// the "deepseek" namespace. Fields DeepSeek did not report are omitted.
type FileMetadata struct {
	Object    string `json:"object,omitempty"`
	Filename  string `json:"filename,omitempty"`
	Purpose   string `json:"purpose,omitempty"`
	Bytes     *int64 `json:"bytes,omitempty"`
	CreatedAt *int64 `json:"createdAt,omitempty"`
	ExpiresAt *int64 `json:"expiresAt,omitempty"`
}

// fileUpload is the validated content of one upload.
type fileUpload struct {
	content   []byte
	mediaType string
	filename  string
}

// fileResponse is the uploaded file object.
type fileResponse struct {
	ID        *string `json:"id"`
	Object    *string `json:"object"`
	Bytes     *int64  `json:"bytes"`
	CreatedAt *int64  `json:"created_at"`
	Filename  *string `json:"filename"`
	Purpose   *string `json:"purpose"`
	ExpiresAt *int64  `json:"expires_at"`
}

// UploadFile validates and uploads an image for use in chat prompts. A
// generic media type, such as "application/octet-stream", requires
// detectable content or a filename with an image extension. Filenames have
// at most 512 characters. Streams are not supported.
func (f *Files) UploadFile(ctx context.Context, options model.UploadFileOptions) (*model.UploadFileResult, error) {
	filesOptions, err := readFilesOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	content, err := model.InlineFileBytes(options.Data)
	if err != nil {
		return nil, err
	}

	upload := fileUpload{
		content:   content,
		mediaType: options.MediaType,
		filename:  options.Filename,
	}

	if err := validateUpload(upload); err != nil {
		return nil, err
	}

	headers, err := f.provider.requestHeaders()
	if err != nil {
		return nil, err
	}

	request := apicall.Request{
		Client:     f.provider.httpClient,
		URL:        f.provider.url(filesPath),
		Headers:    apicall.MergeHeaders(headers, options.Headers),
		ParseError: parseError,
	}

	response, err := apicall.PostForm(ctx, request, uploadFields(upload, filesOptions))
	if err != nil {
		return nil, err
	}

	var decoded fileResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if err := decoded.validate(); err != nil {
		return nil, err
	}

	filename := valueOrEmpty(decoded.Filename)
	if filename == "" {
		filename = upload.filename
	}

	return &model.UploadFileResult{
		ProviderReference: model.ProviderReference{
			Namespace: *decoded.ID,
		},
		Filename:  filename,
		MediaType: upload.mediaType,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: decoded.metadata(),
		},
	}, nil
}

// uploadFields returns the multipart fields of an upload. An expiry is sent
// as bracketed fields anchored at the creation time.
func uploadFields(upload fileUpload, options FilesOptions) []apicall.FormField {
	filename := upload.filename
	if filename == "" {
		filename = defaultUploadName
	}

	fields := []apicall.FormField{
		{
			Name:      "file",
			Filename:  filename,
			MediaType: upload.mediaType,
			Data:      upload.content,
		},
		{
			Name:  "purpose",
			Value: userDataPurpose,
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

// validateUpload checks the size, filename and image format before
// uploading. Detected content takes precedence over the declared type.
func validateUpload(upload fileUpload) error {
	if len(upload.content) > maxFileBytes {
		return fmt.Errorf(
			"%w: data: deepseek file uploads must not exceed 64 MiB (%d bytes). received %d bytes",
			model.ErrInvalidArgument,
			maxFileBytes,
			len(upload.content),
		)
	}

	if length := utf8.RuneCountInString(upload.filename); length > maxFilenameLength {
		return fmt.Errorf(
			"%w: filename: deepseek filenames must not exceed %d characters. received %d characters",
			model.ErrInvalidArgument,
			maxFilenameLength,
			length,
		)
	}

	detected, isDetected := mediatype.Detect(upload.content, "")
	if isDetected && !supportedImageTypes[detected] {
		return fmt.Errorf(
			"%w: data: deepseek file uploads support JPEG, PNG, GIF, and WebP images. detected unsupported file content type %q",
			model.ErrInvalidArgument,
			detected,
		)
	}

	mediaType := normalizeMediaType(upload.mediaType)
	if supportedImageTypes[mediaType] {
		return nil
	}

	if !genericMediaTypes[mediaType] {
		return fmt.Errorf(
			"%w: mediaType: deepseek file uploads support JPEG, PNG, GIF, and WebP images. received unsupported media type %q",
			model.ErrInvalidArgument,
			upload.mediaType,
		)
	}

	if isDetected || hasSupportedExtension(upload.filename) {
		return nil
	}

	return fmt.Errorf(
		"%w: mediaType: deepseek file uploads support JPEG, PNG, GIF, and WebP images. "+
			"provide a supported media type or a filename ending in .jpg, .jpeg, .png, .gif, or .webp. received %q",
		model.ErrInvalidArgument,
		upload.mediaType,
	)
}

// normalizeMediaType drops parameters and lowercases a media type.
func normalizeMediaType(mediaType string) string {
	essence, _, _ := strings.Cut(mediaType, ";")

	return strings.ToLower(strings.TrimSpace(essence))
}

func hasSupportedExtension(filename string) bool {
	separator := strings.LastIndex(filename, ".")
	if separator < 0 {
		return false
	}

	return supportedExtensions[strings.ToLower(filename[separator+1:])]
}

// validate checks the required ID and the values of the reported fields.
func (r fileResponse) validate() error {
	if r.ID == nil {
		return fmt.Errorf("%w: file response has no id", model.ErrInvalidResponse)
	}

	if err := expectLiteral("object", r.Object, fileObject); err != nil {
		return err
	}

	if err := expectLiteral("purpose", r.Purpose, userDataPurpose); err != nil {
		return err
	}

	for name, value := range map[string]*int64{
		"bytes":      r.Bytes,
		"created_at": r.CreatedAt,
		"expires_at": r.ExpiresAt,
	} {
		if value != nil && *value < 0 {
			return fmt.Errorf("%w: file response %s must not be negative", model.ErrInvalidResponse, name)
		}
	}

	return nil
}

// metadata reports the fields DeepSeek returned.
func (r fileResponse) metadata() FileMetadata {
	return FileMetadata{
		Object:    valueOrEmpty(r.Object),
		Filename:  valueOrEmpty(r.Filename),
		Purpose:   valueOrEmpty(r.Purpose),
		Bytes:     r.Bytes,
		CreatedAt: r.CreatedAt,
		ExpiresAt: r.ExpiresAt,
	}
}
