package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/aarock1234/ai/internal/apierror"
)

// ClientConfig holds the settings for making requests to an
// OpenAI-compatible API.
type ClientConfig struct {
	// BaseURL is the API base URL (e.g. "https://api.openai.com/v1").
	BaseURL string

	// APIKey is the bearer token for authentication.
	APIKey string

	// ExtraHeaders are additional HTTP headers sent with every request
	// (e.g. OpenRouter's HTTP-Referer, X-Title).
	ExtraHeaders http.Header

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

// DoRequest sends a non-streaming chat completion request and returns
// the decoded response.
func DoRequest(ctx context.Context, cfg *ClientConfig, req *ChatRequest) (*ChatResponse, error) {
	req.Stream = false

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshalling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	setHeaders(httpReq, cfg)

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
		return nil, apierror.New(resp.StatusCode, string(respBody))
	}

	var result ChatResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	return &result, nil
}

// DoStreamRequest sends a streaming chat completion request and
// returns the response body for the caller to consume via DecodeStream.
// The caller is responsible for closing the returned ReadCloser.
func DoStreamRequest(ctx context.Context, cfg *ClientConfig, req *ChatRequest) (io.ReadCloser, error) {
	req.Stream = true
	req.StreamOptions = &streamOptions{IncludeUsage: true}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshalling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	setHeaders(httpReq, cfg)

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

		return nil, apierror.New(resp.StatusCode, string(respBody))
	}

	return resp.Body, nil
}

// setHeaders applies auth and content-type headers to the request.
func setHeaders(req *http.Request, cfg *ClientConfig) {
	req.Header.Set("Content-Type", "application/json")

	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}

	for key, values := range cfg.ExtraHeaders {
		for _, v := range values {
			req.Header.Set(key, v)
		}
	}
}
