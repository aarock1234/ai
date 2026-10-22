package credential

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// defaultRefreshWindow is how long before expiry a token counts as expiring.
const defaultRefreshWindow = 5 * time.Minute

// maxTokenResponseBytes bounds a token endpoint response.
const maxTokenResponseBytes = 1 << 20

// OAuthCredential is a stored OAuth credential.
type OAuthCredential struct {
	AccessToken  string
	RefreshToken string
	ExpiresAt    time.Time
}

// IsAccessTokenExpiringSoon reports whether a token expires within the
// refresh window. A zero window selects five minutes.
func IsAccessTokenExpiringSoon(expiresAt time.Time, now time.Time, refreshWindow time.Duration) bool {
	if refreshWindow <= 0 {
		refreshWindow = defaultRefreshWindow
	}

	return !expiresAt.After(now.Add(refreshWindow))
}

// RequestFormat is the body encoding of a token refresh request.
type RequestFormat string

// Token refresh request formats. The empty format is RequestForm.
const (
	RequestForm RequestFormat = "form"
	RequestJSON RequestFormat = "json"
)

// RefreshOptions configure RefreshOAuthAccessToken.
type RefreshOptions struct {
	TokenURL     string
	ClientID     string
	RefreshToken string

	RequestFormat RequestFormat

	// Header is sent with the request.
	Header http.Header

	// Client sends the request. Nil uses http.DefaultClient. Redirects are
	// never followed.
	Client *http.Client
}

// RefreshResult is a refreshed access token.
type RefreshResult struct {
	AccessToken string
	ExpiresAt   time.Time

	// RefreshToken is the rotated refresh token, empty when the endpoint did
	// not rotate it.
	RefreshToken string
}

// tokenResponse is the part of a token endpoint response the refresh reads.
type tokenResponse struct {
	AccessToken  *string         `json:"access_token"`
	ExpiresIn    *float64        `json:"expires_in"`
	RefreshToken json.RawMessage `json:"refresh_token"`
}

// RefreshOAuthAccessToken exchanges a refresh token for a new access token.
// The expiry comes from expires_in, falling back to the access token's JWT
// exp claim.
func RefreshOAuthAccessToken(ctx context.Context, options RefreshOptions) (RefreshResult, error) {
	request, err := refreshRequest(ctx, options)
	if err != nil {
		return RefreshResult{}, err
	}

	response, err := withoutRedirects(options.Client).Do(request)
	if err != nil {
		return RefreshResult{}, fmt.Errorf("refresh oauth access token: %w", err)
	}

	defer func() {
		_ = response.Body.Close()
	}()

	body, err := io.ReadAll(io.LimitReader(response.Body, maxTokenResponseBytes))
	if err != nil {
		return RefreshResult{}, fmt.Errorf("read oauth token response: %w", err)
	}

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return RefreshResult{}, fmt.Errorf("oauth access token refresh failed with status %d", response.StatusCode)
	}

	return parseTokenResponse(body, time.Now())
}

// errRedirect reports a token endpoint that answered with a redirect.
var errRedirect = errors.New("oauth token endpoint redirects are not followed")

// withoutRedirects returns a copy of client, or of http.DefaultClient when
// client is nil, that does not follow redirects, so that the refresh token
// is never sent to another endpoint.
func withoutRedirects(client *http.Client) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}

	copied := *client
	copied.CheckRedirect = func(*http.Request, []*http.Request) error {
		return errRedirect
	}

	return &copied
}

// refreshRequest builds the token refresh request.
func refreshRequest(ctx context.Context, options RefreshOptions) (*http.Request, error) {
	values := map[string]string{
		"grant_type":    "refresh_token",
		"client_id":     options.ClientID,
		"refresh_token": options.RefreshToken,
	}

	var body []byte
	contentType := "application/x-www-form-urlencoded"

	if options.RequestFormat == RequestJSON {
		encoded, err := json.Marshal(values)
		if err != nil {
			return nil, fmt.Errorf("encode oauth refresh request: %w", err)
		}

		body = encoded
		contentType = "application/json"
	} else {
		form := url.Values{}
		for key, value := range values {
			form.Set(key, value)
		}

		body = []byte(form.Encode())
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, options.TokenURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create oauth refresh request: %w", err)
	}

	request.Header.Set("Content-Type", contentType)

	for name, headerValues := range options.Header {
		for _, value := range headerValues {
			request.Header.Add(name, value)
		}
	}

	return request, nil
}

// parseTokenResponse validates a token endpoint response.
func parseTokenResponse(body []byte, now time.Time) (RefreshResult, error) {
	var response tokenResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return RefreshResult{}, errors.New("oauth access token refresh returned invalid json")
	}

	if response.AccessToken == nil || *response.AccessToken == "" {
		return RefreshResult{}, errors.New("oauth access token refresh response is missing access_token")
	}

	result := RefreshResult{
		AccessToken: *response.AccessToken,
	}

	switch {
	case response.ExpiresIn != nil && *response.ExpiresIn >= 0:
		result.ExpiresAt = now.Add(time.Duration(*response.ExpiresIn * float64(time.Second)))
	default:
		expiresAt, ok := JWTExpiresAt(result.AccessToken)
		if !ok {
			return RefreshResult{}, errors.New("oauth access token refresh response does not include a usable expiry")
		}

		result.ExpiresAt = expiresAt
	}

	if len(response.RefreshToken) == 0 || string(response.RefreshToken) == "null" {
		return result, nil
	}

	var rotated string
	if err := json.Unmarshal(response.RefreshToken, &rotated); err != nil || rotated == "" {
		return RefreshResult{}, errors.New("oauth access token refresh response contains an invalid refresh_token")
	}

	result.RefreshToken = rotated

	return result, nil
}
