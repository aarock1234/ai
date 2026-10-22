package apicall

import (
	"context"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"sync"

	"github.com/aarock1234/ai/pkg/model"
)

// FormStream is the streamed file part of PostFormStream.
type FormStream struct {
	Name      string
	Filename  string
	MediaType string

	// Reader is read once while the request is sent. It is not closed.
	Reader io.Reader
}

// PostFormStream sends fields as multipart/form-data followed by file, whose
// content is copied from its reader while the request is sent instead of
// being buffered, and buffers the successful response.
func PostFormStream(ctx context.Context, request Request, fields []FormField, file FormStream) (*Response, error) {
	bodyReader, bodyWriter := io.Pipe()
	writer := multipart.NewWriter(bodyWriter)

	var encoding sync.WaitGroup
	encoding.Go(func() {
		bodyWriter.CloseWithError(writeStreamedForm(writer, fields, file))
	})

	// Closing the reader stops an encoder that the request abandoned.
	defer encoding.Wait()
	defer closeBody(bodyReader)

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, request.URL, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("%w: build request: %w", model.ErrInvalidArgument, err)
	}

	httpRequest.Header = MergeHeaders(request.Headers)
	httpRequest.Header.Set("Content-Type", writer.FormDataContentType())

	response, err := do(request, httpRequest)
	if err != nil {
		return nil, err
	}

	return readResponse(response, nil)
}

// writeStreamedForm encodes the fields and the streamed file, then closes
// the multipart writer.
func writeStreamedForm(writer *multipart.Writer, fields []FormField, file FormStream) error {
	for _, field := range fields {
		if err := writeFormField(writer, field); err != nil {
			return fmt.Errorf("encode form field %q: %w", field.Name, err)
		}
	}

	part, err := writer.CreatePart(filePartHeader(file.Name, file.Filename, file.MediaType))
	if err != nil {
		return fmt.Errorf("encode form field %q: %w", file.Name, err)
	}

	if _, err := io.Copy(part, file.Reader); err != nil {
		return fmt.Errorf("stream form field %q: %w", file.Name, err)
	}

	return writer.Close()
}
