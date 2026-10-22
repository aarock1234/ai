package policyopa

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
)

// HTTPPolicyClientOptions configure an HTTPPolicyClient.
type HTTPPolicyClientOptions struct {
	// URL is the base URL of the OPA server, such as
	// "http://localhost:8181".
	URL string

	// Headers are sent with every request, for example to authenticate with
	// Styra DAS or Enterprise OPA.
	Headers http.Header

	// Client sends the requests. Nil uses http.DefaultClient.
	Client *http.Client
}

// HTTPPolicyClient evaluates policies on a running OPA server through its
// REST Data API.
type HTTPPolicyClient struct {
	options HTTPPolicyClientOptions
}

var _ PolicyClient = (*HTTPPolicyClient)(nil)

// NewHTTPPolicyClient returns a client for the OPA server at options.URL.
func NewHTTPPolicyClient(options HTTPPolicyClientOptions) *HTTPPolicyClient {
	return &HTTPPolicyClient{
		options: options,
	}
}

// opaDataRequest is the body of a Data API query.
type opaDataRequest struct {
	Input any `json:"input,omitempty"`
}

// opaDataResponse is the body of a Data API response. Result is absent when
// the policy is undefined for the input.
type opaDataResponse struct {
	Result json.RawMessage `json:"result"`
}

// opaErrorResponse is the body of a failed Data API request.
type opaErrorResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Evaluate queries the document at path, such as "agent/call/decision",
// with input, and returns its value, or nil when it is undefined.
func (c *HTTPPolicyClient) Evaluate(ctx context.Context, path string, input any) (json.RawMessage, error) {
	request := apicall.Request{
		Client:     c.options.Client,
		URL:        c.dataURL(path),
		Headers:    c.options.Headers,
		ParseError: parseOPAError,
	}

	body := opaDataRequest{
		Input: input,
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, fmt.Errorf("evaluate opa policy %q: %w", path, err)
	}

	var decoded opaDataResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, fmt.Errorf("decode opa policy %q result: %w", path, err)
	}

	return decoded.Result, nil
}

// dataURL returns the Data API URL of the document at path.
func (c *HTTPPolicyClient) dataURL(path string) string {
	base := strings.TrimRight(c.options.URL, "/")

	return base + "/v1/data/" + strings.Trim(path, "/")
}

// parseOPAError returns the message of an OPA error response.
func parseOPAError(body []byte) string {
	var response opaErrorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	return response.Message
}
