package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/aarock1234/ai/internal/apierror"
)

// ClientConfig holds the settings for making requests to the Anthropic API.
type ClientConfig struct {
	// BaseURL is the API base URL (e.g. "https://api.anthropic.com/v1").
	BaseURL string

	// APIKey is the Anthropic API key.
	APIKey string

	// APIVersion is the anthropic-version header value.
	APIVersion string

	// HTTPClient is the HTTP client to use. If nil, http.DefaultClient
	// is used.
	HTTPClient *http.Client
}

func (c *ClientConfig) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}

	return http.DefaultClient
}

// DoRequest sends a non-streaming Messages API request and returns the
// decoded response.
func DoRequest(ctx context.Context, cfg *ClientConfig, req *MessageRequest) (*MessageResponse, error) {
	body, err := marshalRequest(req, false)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	setHeaders(httpReq, cfg, req.Headers)

	resp, err := cfg.httpClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, apierror.New(resp.StatusCode, parseErrorMessage(respBody))
	}

	var result MessageResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	return &result, nil
}

// DoStreamRequest sends a streaming Messages API request and returns the
// response body for the caller to consume via DecodeStream. The caller is
// responsible for closing the returned ReadCloser.
func DoStreamRequest(ctx context.Context, cfg *ClientConfig, req *MessageRequest) (io.ReadCloser, error) {
	body, err := marshalRequest(req, true)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	setHeaders(httpReq, cfg, req.Headers)

	resp, err := cfg.httpClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()

		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("reading error response: %w", err)
		}

		return nil, apierror.New(resp.StatusCode, parseErrorMessage(respBody))
	}

	return resp.Body, nil
}

func setHeaders(req *http.Request, cfg *ClientConfig, extra http.Header) {
	req.Header.Set("Content-Type", "application/json")

	if cfg.APIKey != "" {
		req.Header.Set("x-api-key", cfg.APIKey)
	}

	if cfg.APIVersion != "" {
		req.Header.Set("anthropic-version", cfg.APIVersion)
	}

	for key, values := range extra {
		for _, value := range values {
			req.Header.Set(key, value)
		}
	}
}

func parseErrorMessage(body []byte) string {
	var apiErr apiErrorResponse
	if err := json.Unmarshal(body, &apiErr); err == nil && apiErr.Error != nil && apiErr.Error.Message != "" {
		return apiErr.Error.Message
	}

	return string(body)
}

func marshalRequest(req *MessageRequest, stream bool) ([]byte, error) {
	requestBody := *req
	requestBody.Stream = stream

	body, err := json.Marshal(&requestBody)
	if err != nil {
		return nil, fmt.Errorf("marshalling request: %w", err)
	}

	return body, nil
}
