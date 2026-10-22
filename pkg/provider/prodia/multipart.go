package prodia

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// jobPart is one part of a multipart job response.
type jobPart struct {
	name               string
	contentType        string
	contentDisposition string
	body               []byte
}

// readJobParts splits a multipart job response into its parts.
func readJobParts(headers http.Header, body []byte) ([]jobPart, error) {
	contentType := headers.Get("Content-Type")

	_, parameters, err := mime.ParseMediaType(contentType)
	if err != nil || parameters["boundary"] == "" {
		return nil, fmt.Errorf("%w: Prodia response missing multipart boundary in content-type: %s", model.ErrInvalidResponse, contentType)
	}

	reader := multipart.NewReader(bytes.NewReader(body), parameters["boundary"])

	var parts []jobPart
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			return parts, nil
		}

		if err != nil {
			return nil, fmt.Errorf("%w: read multipart response: %w", model.ErrInvalidResponse, err)
		}

		data, err := io.ReadAll(part)
		if err != nil {
			return nil, fmt.Errorf("%w: read multipart response: %w", model.ErrInvalidResponse, err)
		}

		parts = append(parts, jobPart{
			name:               part.FormName(),
			contentType:        part.Header.Get("Content-Type"),
			contentDisposition: part.Header.Get("Content-Disposition"),
			body:               data,
		})
	}
}

// decodeJob decodes the job record part.
func decodeJob(part jobPart) (job, error) {
	var record job
	if err := apicall.DecodeJSON(part.body, &record); err != nil {
		return job{}, fmt.Errorf("decode job part: %w", err)
	}

	return record, nil
}

// isTextOutput reports whether an output part holds text.
func isTextOutput(part jobPart) bool {
	return strings.HasPrefix(part.contentType, "text/") || strings.Contains(part.contentDisposition, ".txt")
}

// isImageOutput reports whether a part holds an image.
func isImageOutput(part jobPart) bool {
	return strings.HasPrefix(part.contentType, "image/")
}
