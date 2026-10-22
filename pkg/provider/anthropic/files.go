package anthropic

import (
	"context"
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// filesBeta enables the Files API.
const filesBeta = "files-api-2025-04-14"

// defaultUploadName is the multipart filename of uploads without one.
const defaultUploadName = "blob"

var (
	_ model.Files         = (*Files)(nil)
	_ model.FilesProvider = (*Provider)(nil)
)

// FileMetadata is the provider metadata of an uploaded file, stored under
// the "anthropic" namespace.
type FileMetadata struct {
	Filename  string `json:"filename"`
	MimeType  string `json:"mimeType"`
	SizeBytes int64  `json:"sizeBytes"`

	// CreatedAt is the RFC 3339 creation time.
	CreatedAt string `json:"createdAt"`

	// Downloadable is nil when not reported.
	Downloadable *bool `json:"downloadable,omitempty"`
}

// Files uploads files with the Anthropic Files API.
type Files struct {
	provider *Provider
}

// fileResponse is an uploaded file object.
type fileResponse struct {
	ID           string  `json:"id"`
	Type         string  `json:"type"`
	Filename     *string `json:"filename"`
	MimeType     *string `json:"mime_type"`
	SizeBytes    *int64  `json:"size_bytes"`
	CreatedAt    *string `json:"created_at"`
	Downloadable *bool   `json:"downloadable"`
}

// Provider returns the provider identifier, the provider name, such as
// "anthropic.messages".
func (f *Files) Provider() string {
	return f.provider.name
}

// UploadFile uploads inline file content. Streams are not supported.
func (f *Files) UploadFile(ctx context.Context, options model.UploadFileOptions) (*model.UploadFileResult, error) {
	content, err := model.InlineFileBytes(options.Data)
	if err != nil {
		return nil, err
	}

	request, err := f.provider.betaRequest(ctx, "/files", filesBeta, options.Headers)
	if err != nil {
		return nil, err
	}

	filename := options.Filename
	if filename == "" {
		filename = defaultUploadName
	}

	response, err := apicall.PostForm(ctx, request, []apicall.FormField{
		{
			Name:      "file",
			Filename:  filename,
			MediaType: options.MediaType,
			Data:      content,
		},
	})
	if err != nil {
		return nil, err
	}

	var file fileResponse
	if err := apicall.DecodeJSON(response.Body, &file); err != nil {
		return nil, err
	}

	if err := file.validate(); err != nil {
		return nil, err
	}

	return &model.UploadFileResult{
		ProviderReference: model.ProviderReference{
			Namespace: file.ID,
		},
		MediaType: *file.MimeType,
		Filename:  *file.Filename,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: FileMetadata{
				Filename:     *file.Filename,
				MimeType:     *file.MimeType,
				SizeBytes:    *file.SizeBytes,
				CreatedAt:    *file.CreatedAt,
				Downloadable: file.Downloadable,
			},
		},
	}, nil
}

// validate checks that the file object has every required field.
func (r fileResponse) validate() error {
	isComplete := r.ID != "" &&
		r.Type == "file" &&
		r.Filename != nil &&
		r.MimeType != nil &&
		r.SizeBytes != nil &&
		r.CreatedAt != nil

	if !isComplete {
		return fmt.Errorf(
			"%w: file response must be a file object with id, filename, mime_type, size_bytes and created_at",
			model.ErrInvalidResponse,
		)
	}

	return nil
}

// betaRequest prepares a call to path below the base URL with the
// configured headers, a beta header and the call headers.
func (p *Provider) betaRequest(ctx context.Context, path string, beta string, callHeaders http.Header) (apicall.Request, error) {
	baseURL, err := p.resolveBaseURL()
	if err != nil {
		return apicall.Request{}, err
	}

	configured, err := p.configuredHeaders(ctx)
	if err != nil {
		return apicall.Request{}, err
	}

	betaHeader := http.Header{
		"Anthropic-Beta": {
			beta,
		},
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        baseURL + path,
		Headers:    apicall.MergeHeaders(configured, betaHeader, callHeaders),
		ParseError: parseErrorMessage,
	}, nil
}
