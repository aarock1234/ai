package vercel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/harness/credential"
)

// Device sign-in settings of @vercel/sandbox's development credentials.
const (
	deviceSignInScope         = "openid offline_access"
	deviceCodeGrantType       = "urn:ietf:params:oauth:grant-type:device_code"
	deviceTokenRequestTimeout = 10 * time.Second
	slowDownIncrement         = 5 * time.Second
	defaultSignInTimeout      = 5 * time.Minute
	vercelSignInTimeout       = time.Minute
)

// ErrSignInTimedOut reports that the user did not approve a device sign-in
// within its timeout.
var ErrSignInTimedOut = errors.New("authentication flow timed out")

// oauthErrorCodes are the error codes of valid OAuth error responses.
var oauthErrorCodes = []string{
	"invalid_request",
	"invalid_client",
	"invalid_grant",
	"unauthorized_client",
	"unsupported_grant_type",
	"invalid_scope",
	"server_error",
	"authorization_pending",
	"slow_down",
	"access_denied",
	"expired_token",
	"unsupported_token_type",
}

// DeviceAuthorization is a pending device sign-in, which the user approves
// in a browser.
type DeviceAuthorization struct {
	// UserCode is the code the user enters at VerificationURI.
	UserCode string

	// VerificationURI is the page where the user enters UserCode.
	VerificationURI string

	// VerificationURIComplete is the page that approves the sign-in
	// without entering UserCode.
	VerificationURIComplete string

	// ExpiresAt is when the codes expire.
	ExpiresAt time.Time
}

// DeviceSignIn configures the interactive device sign-in to Vercel.
type DeviceSignIn struct {
	// Prompt shows the authorization to the user, for example by printing
	// VerificationURIComplete, or VerificationURI with UserCode. It returns
	// once the authorization is shown; waiting for approval starts then,
	// and an error cancels the sign-in. The package never opens a browser.
	Prompt func(ctx context.Context, authorization DeviceAuthorization) error

	// Timeout bounds the wait for approval. Zero selects 5 minutes, or 1
	// minute when VERCEL_URL is set.
	Timeout time.Duration

	// Logger receives progress notices. Nil uses slog.Default.
	Logger *slog.Logger
}

// SignInOptions configure SignIn.
type SignInOptions struct {
	DeviceSignIn

	// HTTPClient sends the requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client
}

// SignIn signs in to Vercel with the OAuth device authorization flow of the
// Vercel CLI, a port of the development sign-in of @vercel/sandbox, and
// stores the tokens in the CLI's auth.json, where DevelopmentCredentials
// and the OIDC token refresh find them. Prompt is required.
//
// It polls the token endpoint at the interval the authorization server
// asks for, slowing down when asked to and doubling the interval after a
// request timeout, until the user approves or denies the sign-in, the codes
// expire, the timeout elapses (ErrSignInTimedOut) or ctx ends.
//
// Go adaptation: upstream prints the authorization and opens a browser when
// the user presses return. The caller shows it through Prompt instead, and
// auth.json keeps its other members and is replaced atomically.
func SignIn(ctx context.Context, options SignInOptions) error {
	if options.Prompt == nil {
		return errors.New("sign in to vercel: prompt is required")
	}

	httpClient := options.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	server, err := discoverAuthorizationServer(ctx, httpClient)
	if err != nil {
		return fmt.Errorf("sign in to vercel: %w", err)
	}

	authorization, err := requestDeviceAuthorization(ctx, httpClient, server)
	if err != nil {
		return fmt.Errorf("sign in to vercel: %w", err)
	}

	if err := options.Prompt(ctx, authorization.public()); err != nil {
		return fmt.Errorf("sign in to vercel: %w", err)
	}

	timeout := options.signInTimeout()

	pollCtx, cancel := context.WithTimeoutCause(ctx, timeout, signInTimeoutError(timeout))
	defer cancel()

	tokens, err := pollForTokens(pollCtx, devicePoll{
		httpClient:    httpClient,
		tokenEndpoint: server.TokenEndpoint,
		authorization: authorization,
	})
	if err != nil {
		return fmt.Errorf("sign in to vercel: %w", err)
	}

	return storeSignInTokens(tokens)
}

// signInTimeout returns the timeout, defaulting to a shorter one on Vercel.
func (s DeviceSignIn) signInTimeout() time.Duration {
	if s.Timeout > 0 {
		return s.Timeout
	}

	if os.Getenv("VERCEL_URL") != "" {
		return vercelSignInTimeout
	}

	return defaultSignInTimeout
}

func (s DeviceSignIn) logger() *slog.Logger {
	if s.Logger == nil {
		return slog.Default()
	}

	return s.Logger
}

func signInTimeoutError(timeout time.Duration) error {
	return fmt.Errorf(
		"%w after %s; provide a token to avoid the interactive sign-in, or link the project with `vercel link` to refresh oidc tokens automatically",
		ErrSignInTimedOut,
		timeout,
	)
}

// deviceAuthorizationResponse is a device authorization response. All
// members are required.
type deviceAuthorizationResponse struct {
	DeviceCode              string   `json:"device_code"`
	UserCode                string   `json:"user_code"`
	VerificationURI         string   `json:"verification_uri"`
	VerificationURIComplete string   `json:"verification_uri_complete"`
	ExpiresIn               *float64 `json:"expires_in"`
	Interval                *float64 `json:"interval"`
}

func (r deviceAuthorizationResponse) isComplete() bool {
	hasCodes := r.DeviceCode != "" && r.UserCode != ""
	hasURIs := isAbsoluteURL(r.VerificationURI) && isAbsoluteURL(r.VerificationURIComplete)

	return hasCodes && hasURIs && r.ExpiresIn != nil && r.Interval != nil
}

func isAbsoluteURL(value string) bool {
	parsed, err := url.Parse(value)

	return err == nil && parsed.Scheme != "" && parsed.Host != ""
}

// deviceAuthorization is a started device sign-in.
type deviceAuthorization struct {
	deviceCode              string
	userCode                string
	verificationURI         string
	verificationURIComplete string
	expiresAt               time.Time
	interval                time.Duration
}

func (a deviceAuthorization) public() DeviceAuthorization {
	return DeviceAuthorization{
		UserCode:                a.userCode,
		VerificationURI:         a.verificationURI,
		VerificationURIComplete: a.verificationURIComplete,
		ExpiresAt:               a.expiresAt,
	}
}

// requestDeviceAuthorization starts a device sign-in for the Vercel CLI's
// OAuth client.
func requestDeviceAuthorization(ctx context.Context, httpClient *http.Client, server authorizationServer) (deviceAuthorization, error) {
	if server.DeviceAuthorizationEndpoint == "" {
		return deviceAuthorization{}, errors.New("invalid oauth discovery response: missing device_authorization_endpoint")
	}

	form := url.Values{
		"client_id": {
			vercelCLIClientID,
		},
		"scope": {
			deviceSignInScope,
		},
	}

	request, err := newOAuthFormRequest(ctx, server.DeviceAuthorizationEndpoint, form)
	if err != nil {
		return deviceAuthorization{}, fmt.Errorf("create device authorization request: %w", err)
	}

	var response deviceAuthorizationResponse

	var failure oauthErrorResponse

	if _, err := doOAuthJSON(httpClient, request, &response, &failure); err != nil {
		return deviceAuthorization{}, err
	}

	if !response.isComplete() {
		return deviceAuthorization{}, fmt.Errorf(
			"failed to parse device authorization response from the vercel authorization server: %w",
			failure.cause(),
		)
	}

	return deviceAuthorization{
		deviceCode:              response.DeviceCode,
		userCode:                response.UserCode,
		verificationURI:         response.VerificationURI,
		verificationURIComplete: response.VerificationURIComplete,
		expiresAt:               time.Now().Add(seconds(*response.ExpiresIn)),
		interval:                seconds(*response.Interval),
	}, nil
}

func seconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Second))
}

// devicePoll is the wait for the approval of a device sign-in.
type devicePoll struct {
	httpClient    *http.Client
	tokenEndpoint string
	authorization deviceAuthorization
}

// pollForTokens requests the tokens of an approved sign-in until the user
// approves it, the request fails, the codes expire or ctx ends.
func pollForTokens(ctx context.Context, poll devicePoll) (oauthTokens, error) {
	interval := poll.authorization.interval

	for time.Now().Before(poll.authorization.expiresAt) {
		attempt, err := requestDeviceTokens(ctx, poll)

		switch {
		case err != nil && ctx.Err() != nil:
			return oauthTokens{}, context.Cause(ctx)

		case err != nil && isRequestTimeout(err):
			interval *= 2

		case err != nil:
			return oauthTokens{}, err

		case attempt.isApproved():
			return attempt.tokens, nil

		case attempt.failure.Code == "authorization_pending":
			// The user has not approved yet; wait for the same interval.

		case attempt.failure.Code == "slow_down":
			interval += slowDownIncrement

		default:
			return oauthTokens{}, attempt.failure.cause()
		}

		if err := waitForRetry(ctx, interval); err != nil {
			return oauthTokens{}, err
		}
	}

	return oauthTokens{}, errors.New("timed out waiting for authentication; please try again")
}

func isRequestTimeout(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}

	netErr, ok := errors.AsType[net.Error](err)

	return ok && netErr.Timeout()
}

// deviceTokenAttempt is the response to one device access token request:
// tokens, or an OAuth error.
type deviceTokenAttempt struct {
	tokens  oauthTokens
	failure oauthErrorResponse
}

func (a deviceTokenAttempt) isApproved() bool {
	return a.tokens.validate() == nil
}

// requestDeviceTokens requests the tokens of the sign-in once, within the
// request timeout.
func requestDeviceTokens(ctx context.Context, poll devicePoll) (deviceTokenAttempt, error) {
	requestCtx, cancel := context.WithTimeout(ctx, deviceTokenRequestTimeout)
	defer cancel()

	form := url.Values{
		"client_id": {
			vercelCLIClientID,
		},
		"grant_type": {
			deviceCodeGrantType,
		},
		"device_code": {
			poll.authorization.deviceCode,
		},
	}

	request, err := newOAuthFormRequest(requestCtx, poll.tokenEndpoint, form)
	if err != nil {
		return deviceTokenAttempt{}, fmt.Errorf("create device token request: %w", err)
	}

	response, err := poll.httpClient.Do(request)
	if err != nil {
		return deviceTokenAttempt{}, err
	}

	defer func() {
		_ = response.Body.Close()
	}()

	content, err := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes))
	if err != nil {
		return deviceTokenAttempt{}, fmt.Errorf("read device token response: %w", err)
	}

	return decodeDeviceTokenAttempt(content)
}

// decodeDeviceTokenAttempt reads tokens, or else an OAuth error, from a
// JSON response. A response that is neither is an unexpected OAuth error.
func decodeDeviceTokenAttempt(content []byte) (deviceTokenAttempt, error) {
	if !json.Valid(content) {
		return deviceTokenAttempt{}, errors.New("decode device token response: invalid json")
	}

	var attempt deviceTokenAttempt

	if err := json.Unmarshal(content, &attempt.tokens); err == nil && attempt.isApproved() {
		return attempt, nil
	}

	if err := json.Unmarshal(content, &attempt.failure); err != nil {
		attempt.failure = oauthErrorResponse{}
	}

	return attempt, nil
}

// oauthErrorResponse is an OAuth 2.0 error response.
type oauthErrorResponse struct {
	Code        string `json:"error"`
	Description string `json:"error_description"`
	URI         string `json:"error_uri"`
}

// cause describes the error: its code, description and URI, or an
// unexpected response when the code is not an OAuth error code.
func (r oauthErrorResponse) cause() error {
	if !slices.Contains(oauthErrorCodes, r.Code) {
		return errors.New("unexpected response from the vercel authorization server")
	}

	var message strings.Builder

	message.WriteString(r.Code)

	if r.Description != "" {
		fmt.Fprintf(&message, ": %s", r.Description)
	}

	if r.URI != "" {
		fmt.Fprintf(&message, " (%s)", r.URI)
	}

	return errors.New(message.String())
}

func newOAuthFormRequest(ctx context.Context, endpoint string, form url.Values) (*http.Request, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}

	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.Header.Set("User-Agent", userAgent)

	return request, nil
}

// storeSignInTokens writes the tokens of a sign-in to the Vercel CLI's
// auth.json.
func storeSignInTokens(tokens oauthTokens) error {
	configDir := cliConfigDir()

	unlock := credential.LockRefresh(authConfigPath(configDir))
	defer unlock()

	return writeCLIAuthTokens(configDir, tokens)
}
