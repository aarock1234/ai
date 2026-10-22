package google

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// apiVersionSuffix ends the default base URL; uploads and downloads use the
// same version below their /upload and /download prefixes.
const apiVersionSuffix = "/v1beta"

// resumableUpload is one file sent with the Gemini API resumable upload
// protocol.
type resumableUpload struct {
	// DisplayName is the file's human-readable name, if any.
	DisplayName *string

	// MediaType is the declared media type of the file.
	MediaType string

	Content []byte

	// ContentType is sent with the content; empty sends none.
	ContentType string

	// Headers are the call headers of the start request.
	Headers http.Header
}

// uploadStartRequest is the body that starts a resumable upload.
type uploadStartRequest struct {
	File uploadStartFile `json:"file"`
}

type uploadStartFile struct {
	DisplayName *string `json:"display_name,omitempty"`
}

// uploadResumable starts a resumable upload, sends the content and decodes
// the finalized upload response.
func uploadResumable[Response any](ctx context.Context, p *Provider, upload resumableUpload) (Response, error) {
	var uploaded Response

	uploadURL, err := p.startResumableUpload(ctx, upload)
	if err != nil {
		return uploaded, err
	}

	request := apicall.Request{
		Client: p.httpClient,
		URL:    uploadURL,
		Headers: http.Header{
			"X-Goog-Upload-Offset": {
				"0",
			},
			"X-Goog-Upload-Command": {
				"upload, finalize",
			},
		},
		ParseError: parseErrorMessage,
	}

	response, err := apicall.PostBody(ctx, request, upload.Content, upload.ContentType)
	if err != nil {
		return uploaded, err
	}

	if err := apicall.DecodeJSON(response.Body, &uploaded); err != nil {
		return uploaded, err
	}

	return uploaded, nil
}

// startResumableUpload starts an upload and returns its upload URL, which
// carries its own authorization.
func (p *Provider) startResumableUpload(ctx context.Context, upload resumableUpload) (string, error) {
	request, err := p.originRequest(ctx, "/upload"+apiVersionSuffix+"/files", upload.Headers)
	if err != nil {
		return "", err
	}

	request.Headers = apicall.MergeHeaders(request.Headers, http.Header{
		"X-Goog-Upload-Protocol": {
			"resumable",
		},
		"X-Goog-Upload-Command": {
			"start",
		},
		"X-Goog-Upload-Header-Content-Length": {
			strconv.Itoa(len(upload.Content)),
		},
		"X-Goog-Upload-Header-Content-Type": {
			upload.MediaType,
		},
	})

	response, err := apicall.PostJSON(ctx, request, uploadStartRequest{
		File: uploadStartFile{
			DisplayName: upload.DisplayName,
		},
	})
	if err != nil {
		return "", err
	}

	uploadURL := response.Headers.Get("X-Goog-Upload-Url")
	if uploadURL == "" {
		return "", fmt.Errorf(
			"%w: %w: no upload URL returned from initiation request",
			ErrFileUploadFailed,
			model.ErrInvalidResponse,
		)
	}

	return uploadURL, nil
}

// originRequest describes an authenticated call to path below the API
// origin, the base URL without its version, such as "/upload/v1beta/files".
func (p *Provider) originRequest(ctx context.Context, path string, headers http.Header) (apicall.Request, error) {
	request, err := p.request(ctx, "", headers)
	if err != nil {
		return apicall.Request{}, err
	}

	baseURL, err := p.currentBaseURL()
	if err != nil {
		return apicall.Request{}, err
	}

	request.URL = strings.TrimSuffix(baseURL, apiVersionSuffix) + path

	return request, nil
}
