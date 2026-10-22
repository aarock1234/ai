// Package credential holds the credential helpers harness adapters use to
// authenticate their runtimes with a native CLI subscription: reading and
// writing tokens in the operating system's credential store and in private
// files, decoding JWT expiries and refreshing OAuth access tokens. It ports
// the native-subscription and oauth-access-token utilities of upstream
// @ai-sdk/harness/utils.
package credential

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"strings"
	"time"
)

// ParseJWTPayload decodes the payload of a JWT without verifying it. It
// reports false when token is not a JWT with a JSON object payload.
func ParseJWTPayload(token string) (map[string]any, bool) {
	payload, ok := jwtPayload(token)
	if !ok {
		return nil, false
	}

	var claims map[string]any
	if err := json.Unmarshal(payload, &claims); err != nil || claims == nil {
		return nil, false
	}

	return claims, true
}

// expiryClaims are the claims JWTExpiresAt reads.
type expiryClaims struct {
	Expiry *float64 `json:"exp"`
}

// JWTExpiresAt returns the expiry of a JWT from its exp claim. It reports
// false when the token has no numeric exp claim.
func JWTExpiresAt(token string) (time.Time, bool) {
	payload, ok := jwtPayload(token)
	if !ok {
		return time.Time{}, false
	}

	var claims expiryClaims
	if err := json.Unmarshal(payload, &claims); err != nil || claims.Expiry == nil {
		return time.Time{}, false
	}

	return time.UnixMilli(int64(*claims.Expiry * 1000)), true
}

// jwtPayload returns the decoded payload segment of a JWT.
func jwtPayload(token string) ([]byte, bool) {
	segments := strings.Split(token, ".")
	if len(segments) != 3 {
		return nil, false
	}

	payload, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(segments[1], "="))
	if err != nil {
		return nil, false
	}

	return payload, true
}

// HomeDirectory returns the user's home directory, or empty when it is
// unknown.
func HomeDirectory() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}

	return home
}
