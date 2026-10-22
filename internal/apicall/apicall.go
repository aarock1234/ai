// Package apicall sends HTTP requests to provider APIs and converts failed
// responses into model.APICallError values. Provider packages own their wire
// formats; this package owns the exchange.
package apicall

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/aarock1234/ai/pkg/model"
)

// maxResponseBytes bounds a buffered response body. Generated images and audio
// arrive inline, so the limit is generous.
const maxResponseBytes = 256 << 20

// ErrorParser extracts the provider's error message from a failed response
// body. It returns an empty string when the body has no recognizable message.
type ErrorParser func(body []byte) string

// Request describes one call to a provider endpoint.
type Request struct {
	// Client sends the request. Nil uses http.DefaultClient.
	Client *http.Client

	URL     string
	Headers http.Header

	// ParseError reads failed response bodies. Nil uses the HTTP status text.
	ParseError ErrorParser
}

// Response is a buffered successful response.
type Response struct {
	Headers http.Header
	Body    []byte

	// RequestBody is the encoded request, for result diagnostics.
	RequestBody []byte
}

// Stream is a successful response whose body the caller reads and closes.
type Stream struct {
	Headers     http.Header
	Body        io.ReadCloser
	RequestBody []byte
}

// PostJSON sends body as JSON and buffers the successful response.
func PostJSON(ctx context.Context, request Request, body any) (*Response, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	response, err := send(ctx, request, http.MethodPost, encoded, "application/json")
	if err != nil {
		return nil, err
	}

	return readResponse(response, encoded)
}

// PostJSONStream sends body as JSON and returns the open response body.
func PostJSONStream(ctx context.Context, request Request, body any) (*Stream, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	response, err := send(ctx, request, http.MethodPost, encoded, "application/json")
	if err != nil {
		return nil, err
	}

	return &Stream{
		Headers:     response.Header,
		Body:        response.Body,
		RequestBody: encoded,
	}, nil
}

// PostBody sends body unchanged with the given content type, such as raw
// audio or an SSML document, and buffers the successful response.
func PostBody(ctx context.Context, request Request, body []byte, contentType string) (*Response, error) {
	response, err := send(ctx, request, http.MethodPost, body, contentType)
	if err != nil {
		return nil, err
	}

	return readResponse(response, body)
}

// Get fetches a resource and buffers the successful response.
func Get(ctx context.Context, request Request) (*Response, error) {
	response, err := send(ctx, request, http.MethodGet, nil, "")
	if err != nil {
		return nil, err
	}

	return readResponse(response, nil)
}

// GetStream fetches a resource and returns the open successful response
// body, which the caller reads and closes.
func GetStream(ctx context.Context, request Request) (*Stream, error) {
	response, err := send(ctx, request, http.MethodGet, nil, "")
	if err != nil {
		return nil, err
	}

	return &Stream{
		Headers: response.Header,
		Body:    response.Body,
	}, nil
}

// Delete deletes a resource and buffers the successful response.
func Delete(ctx context.Context, request Request) (*Response, error) {
	response, err := send(ctx, request, http.MethodDelete, nil, "")
	if err != nil {
		return nil, err
	}

	return readResponse(response, nil)
}

// DecodeJSON decodes a provider response body into target.
func DecodeJSON(body []byte, target any) error {
	if len(bytes.TrimSpace(body)) == 0 {
		return model.ErrEmptyResponseBody
	}

	if err := json.Unmarshal(body, target); err != nil {
		return fmt.Errorf("%w: %w", model.ErrInvalidResponse, err)
	}

	return nil
}

// MergeHeaders combines header sets. A key in a later set replaces all values
// of that key from earlier sets; multiple values within one set are kept.
func MergeHeaders(sets ...http.Header) http.Header {
	merged := make(http.Header)
	for _, set := range sets {
		for key, values := range set {
			merged.Del(key)
			for _, value := range values {
				merged.Add(key, value)
			}
		}
	}

	return merged
}

func send(ctx context.Context, request Request, method string, body []byte, contentType string) (*http.Response, error) {
	httpRequest, err := http.NewRequestWithContext(ctx, method, request.URL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%w: build request: %w", model.ErrInvalidArgument, err)
	}

	httpRequest.Header = MergeHeaders(request.Headers)
	if contentType != "" {
		httpRequest.Header.Set("Content-Type", contentType)
	}

	return do(request, httpRequest)
}

func do(request Request, httpRequest *http.Request) (*http.Response, error) {
	client := request.Client
	if client == nil {
		client = http.DefaultClient
	}

	response, err := client.Do(httpRequest)
	if err != nil {
		return nil, transportError(httpRequest.Context(), request.URL, err)
	}

	if response.StatusCode < 200 || response.StatusCode > 299 {
		return nil, statusError(request, response)
	}

	return response, nil
}

func readResponse(response *http.Response, requestBody []byte) (*Response, error) {
	defer closeBody(response.Body)

	body, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	if len(body) > maxResponseBytes {
		return nil, fmt.Errorf("%w: response body exceeds %d bytes", model.ErrInvalidResponse, maxResponseBytes)
	}

	return &Response{
		Headers:     response.Header,
		Body:        body,
		RequestBody: requestBody,
	}, nil
}

// closeBody closes a response body that has been read or abandoned. Its
// error is ignored: the outcome of the call is already decided by then.
func closeBody(body io.Closer) {
	_ = body.Close()
}

func transportError(ctx context.Context, url string, err error) error {
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}

	var timeout interface {
		Timeout() bool
	}
	retryable := errors.As(err, &timeout) && timeout.Timeout()

	return &model.APICallError{
		Message:     "provider request failed",
		URL:         url,
		IsRetryable: retryable,
		Cause:       err,
	}
}
