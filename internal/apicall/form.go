package apicall

import (
	"bytes"
	"context"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strings"
)

// quoteEscaper escapes quoted Content-Disposition parameters the same way as
// mime/multipart.
var quoteEscaper = strings.NewReplacer(`\`, `\\`, `"`, `\"`)

// FormField is one multipart field: a value, or a file when Filename is set.
type FormField struct {
	Name  string
	Value string

	Filename  string
	MediaType string
	Data      []byte
}

// PostForm sends fields as multipart/form-data, in order, and buffers the
// successful response.
func PostForm(ctx context.Context, request Request, fields []FormField) (*Response, error) {
	body, contentType, err := encodeForm(fields)
	if err != nil {
		return nil, err
	}

	response, err := send(ctx, request, http.MethodPost, body, contentType)
	if err != nil {
		return nil, err
	}

	return readResponse(response, nil)
}

func encodeForm(fields []FormField) ([]byte, string, error) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	for _, field := range fields {
		if err := writeFormField(writer, field); err != nil {
			return nil, "", fmt.Errorf("encode form field %q: %w", field.Name, err)
		}
	}

	if err := writer.Close(); err != nil {
		return nil, "", fmt.Errorf("encode form: %w", err)
	}

	return body.Bytes(), writer.FormDataContentType(), nil
}

func writeFormField(writer *multipart.Writer, field FormField) error {
	if field.Filename == "" {
		return writer.WriteField(field.Name, field.Value)
	}

	part, err := writer.CreatePart(filePartHeader(field.Name, field.Filename, field.MediaType))
	if err != nil {
		return err
	}

	_, err = part.Write(field.Data)

	return err
}

// filePartHeader returns the header of a file part. The media type is left
// out when it is empty.
func filePartHeader(name string, filename string, mediaType string) textproto.MIMEHeader {
	disposition := fmt.Sprintf(
		`form-data; name="%s"; filename="%s"`,
		quoteEscaper.Replace(name),
		quoteEscaper.Replace(filename),
	)

	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", disposition)

	if mediaType != "" {
		header.Set("Content-Type", mediaType)
	}

	return header
}
