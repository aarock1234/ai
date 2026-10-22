package vercel

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// apiTokenRefreshBuffer refreshes OIDC tokens that expire within five
// minutes before each API request, as upstream's API client does.
const apiTokenRefreshBuffer = 5 * time.Minute

// oidcClaims are the claims of a Vercel OIDC token that the client reads.
type oidcClaims struct {
	OwnerID   string `json:"owner_id"`
	ProjectID string `json:"project_id"`

	// Expiry is the expiration time in seconds since the epoch, or nil.
	Expiry *float64 `json:"exp"`
}

// decodeOIDCClaims reads the claims of a JWT without verifying it.
func decodeOIDCClaims(token string) (oidcClaims, error) {
	segments := strings.Split(token, ".")
	if len(segments) != 3 {
		return oidcClaims{}, errors.New("invalid vercel oidc token: expected a jwt")
	}

	payload, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(segments[1], "="))
	if err != nil {
		return oidcClaims{}, fmt.Errorf("invalid vercel oidc token: %w", err)
	}

	var claims oidcClaims
	if err := json.Unmarshal(payload, &claims); err != nil {
		return oidcClaims{}, fmt.Errorf("invalid vercel oidc token: %w", err)
	}

	return claims, nil
}

// expiresWithin reports whether the token expires before buffer elapses.
// Tokens without an expiration never expire.
func (c oidcClaims) expiresWithin(buffer time.Duration) bool {
	if c.Expiry == nil {
		return false
	}

	expiry := time.UnixMilli(int64(*c.Expiry * 1000))

	return expiry.Before(time.Now().Add(buffer))
}

// isTokenExpired decodes token and reports whether it expires before
// buffer elapses.
func isTokenExpired(token string, buffer time.Duration) (bool, error) {
	claims, err := decodeOIDCClaims(token)
	if err != nil {
		return false, err
	}

	return claims.expiresWithin(buffer), nil
}

// oidcScope is the team and project of an OIDC token, when known.
type oidcScope struct {
	teamID    string
	projectID string
}

// oidcTokenSource returns current Vercel OIDC tokens, a port of
// @vercel/oidc's getVercelOidcToken: the token of VERCEL_OIDC_TOKEN_FILE or
// VERCEL_OIDC_TOKEN, refreshed through the linked project when it is
// missing or expires. It is safe for concurrent use.
//
// Go adaptation: upstream stores a refreshed token in process.env. The
// source keeps it instead and prefers it over an expired environment token.
type oidcTokenSource struct {
	httpClient *http.Client

	mu      sync.Mutex
	current string
}

func newOIDCTokenSource(httpClient *http.Client, current string) *oidcTokenSource {
	return &oidcTokenSource{
		httpClient: httpClient,
		current:    current,
	}
}

// token returns a token that stays valid for at least buffer, refreshing it
// when needed.
func (s *oidcTokenSource) token(ctx context.Context, scope oidcScope, buffer time.Duration) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	candidates, err := s.candidates()
	if err != nil {
		return "", err
	}

	var invalid error

	for _, candidate := range candidates {
		expired, err := isTokenExpired(candidate, buffer)
		if err != nil {
			invalid = err

			continue
		}

		if !expired {
			s.current = candidate

			return candidate, nil
		}
	}

	refreshed, err := refreshOIDCToken(ctx, oidcRefresh{
		scope:      scope,
		buffer:     buffer,
		httpClient: s.httpClient,
	})
	if err != nil {
		return "", errors.Join(invalid, err)
	}

	s.current = refreshed

	return refreshed, nil
}

// candidates lists the tokens to try in order: the token file when
// VERCEL_OIDC_TOKEN_FILE is set, and otherwise the last token obtained and
// VERCEL_OIDC_TOKEN.
func (s *oidcTokenSource) candidates() ([]string, error) {
	if path := os.Getenv("VERCEL_OIDC_TOKEN_FILE"); path != "" {
		token, err := readOIDCTokenFile(path)
		if err != nil {
			return nil, err
		}

		return []string{
			token,
		}, nil
	}

	var candidates []string

	if s.current != "" {
		candidates = append(candidates, s.current)
	}

	if token := os.Getenv("VERCEL_OIDC_TOKEN"); token != "" && token != s.current {
		candidates = append(candidates, token)
	}

	return candidates, nil
}

func readOIDCTokenFile(path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read vercel oidc token file: %w", err)
	}

	token := strings.TrimSpace(string(content))
	if token == "" {
		return "", fmt.Errorf("the vercel oidc token file is empty: %s", path)
	}

	return token, nil
}

// hasOIDCEnvironment reports whether the environment provides an OIDC
// token directly.
func hasOIDCEnvironment() bool {
	return os.Getenv("VERCEL_OIDC_TOKEN_FILE") != "" || os.Getenv("VERCEL_OIDC_TOKEN") != ""
}
