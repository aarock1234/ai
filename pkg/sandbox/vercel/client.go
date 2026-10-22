package vercel

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

// API constants.
const (
	defaultBaseURL       = "https://vercel.com/api"
	userAgent            = "ai-sdk-go/sandbox-vercel"
	maxErrorBodyBytes    = 64 << 10
	maxResponseBodyBytes = 32 << 20
)

// APIError is a failed Vercel Sandbox API response.
type APIError struct {
	StatusCode int

	// Code is the API's error code, such as "sandbox_stopping".
	Code string

	// Message is the API's error message.
	Message string

	Body []byte
}

// Error describes the failure.
func (e *APIError) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("vercel sandbox api: status code %d is not ok", e.StatusCode)
	}

	return fmt.Sprintf("vercel sandbox api: status code %d is not ok: %s", e.StatusCode, e.Message)
}

// isAPIStatus reports an API error with the status and, when not empty, the
// error code.
func isAPIStatus(err error, statusCode int, code string) bool {
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != statusCode {
		return false
	}

	return code == "" || apiErr.Code == code
}

// apiClient calls the Vercel Sandbox REST API for one team and project.
type apiClient struct {
	httpClient *http.Client
	baseURL    string

	// oidc refreshes the token before requests when it is a Vercel OIDC
	// token, and is nil otherwise.
	oidc *oidcTokenSource

	// credentialsMu guards the token and team of credentials, which a
	// refresh replaces.
	credentialsMu sync.Mutex
	credentials   Credentials
}

// newAPIClient returns a client for the resolved credentials.
func newAPIClient(credentials Credentials, baseURL string, httpClient *http.Client) *apiClient {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	client := &apiClient{
		httpClient:  httpClient,
		baseURL:     strings.TrimRight(baseURL, "/"),
		credentials: credentials,
	}

	if isOIDCToken(credentials.Token) {
		client.oidc = newOIDCTokenSource(httpClient, credentials.Token)
	}

	return client
}

// isOIDCToken reports whether token is a JWT that names its team, as
// Vercel OIDC tokens do.
func isOIDCToken(token string) bool {
	claims, err := decodeOIDCClaims(token)

	return err == nil && claims.OwnerID != ""
}

// currentCredentials returns the credentials of the next request.
func (c *apiClient) currentCredentials() Credentials {
	c.credentialsMu.Lock()
	defer c.credentialsMu.Unlock()

	return c.credentials
}

// refreshToken replaces an OIDC token that expires within the refresh
// buffer. Failures keep the current token, as upstream does: the request
// then reports the authentication error.
func (c *apiClient) refreshToken(ctx context.Context) {
	if c.oidc == nil {
		return
	}

	current := c.currentCredentials()

	scope := oidcScope{
		teamID:    current.TeamID,
		projectID: current.ProjectID,
	}

	token, err := c.oidc.token(ctx, scope, apiTokenRefreshBuffer)
	if err != nil || token == current.Token {
		return
	}

	c.credentialsMu.Lock()
	defer c.credentialsMu.Unlock()

	c.credentials.Token = token

	if claims, err := decodeOIDCClaims(token); err == nil && claims.OwnerID != "" {
		c.credentials.TeamID = claims.OwnerID
	}
}

// apiRequest describes one API request.
type apiRequest struct {
	method string
	path   string
	query  url.Values

	// body is encoded as JSON unless rawBody is set.
	body        any
	rawBody     []byte
	contentType string
	header      http.Header

	// noRetry sends the request once, for requests that start work that a
	// retry would repeat, such as a streamed command.
	noRetry bool
}

// prepare builds one attempt of a request with a current token.
func (c *apiClient) prepare(ctx context.Context, request apiRequest) (*http.Request, error) {
	c.refreshToken(ctx)

	return c.newRequest(ctx, request)
}

// doJSON sends a request and decodes a JSON response into result.
func (c *apiClient) doJSON(ctx context.Context, request apiRequest, result any) error {
	response, err := c.do(ctx, request)
	if err != nil {
		return err
	}

	defer func() {
		_ = response.Body.Close()
	}()

	body, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBodyBytes))
	if err != nil {
		return fmt.Errorf("read vercel sandbox response: %w", err)
	}

	if result == nil || len(bytes.TrimSpace(body)) == 0 {
		return nil
	}

	if err := json.Unmarshal(body, result); err != nil {
		return fmt.Errorf("decode vercel sandbox response: %w", err)
	}

	return nil
}

// newRequest builds an authenticated request scoped to the team.
func (c *apiClient) newRequest(ctx context.Context, request apiRequest) (*http.Request, error) {
	query := url.Values{}
	for key, values := range request.query {
		query[key] = values
	}

	credentials := c.currentCredentials()
	query.Set("teamId", credentials.TeamID)

	target := c.baseURL + request.path + "?" + query.Encode()

	body, contentType, err := requestBody(request)
	if err != nil {
		return nil, err
	}

	httpRequest, err := http.NewRequestWithContext(ctx, request.method, target, body)
	if err != nil {
		return nil, fmt.Errorf("create vercel sandbox request: %w", err)
	}

	httpRequest.Header.Set("Authorization", "Bearer "+credentials.Token)
	httpRequest.Header.Set("User-Agent", userAgent)

	if contentType != "" {
		httpRequest.Header.Set("Content-Type", contentType)
	}

	for name, values := range request.header {
		for _, value := range values {
			httpRequest.Header.Add(name, value)
		}
	}

	return httpRequest, nil
}

// requestBody encodes the body of a request.
func requestBody(request apiRequest) (io.Reader, string, error) {
	if request.rawBody != nil {
		return bytes.NewReader(request.rawBody), request.contentType, nil
	}

	if request.body == nil {
		return nil, "", nil
	}

	encoded, err := json.Marshal(request.body)
	if err != nil {
		return nil, "", fmt.Errorf("encode vercel sandbox request: %w", err)
	}

	return bytes.NewReader(encoded), "application/json", nil
}

// apiErrorOf reads a failed response into an *APIError.
func apiErrorOf(response *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes))

	var decoded struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}

	_ = json.Unmarshal(body, &decoded)

	apiErr := &APIError{
		StatusCode: response.StatusCode,
		Code:       decoded.Error.Code,
		Message:    decoded.Error.Message,
		Body:       body,
	}

	if response.StatusCode == http.StatusUnauthorized || response.StatusCode == http.StatusForbidden {
		return authenticationError(apiErr)
	}

	return apiErr
}
