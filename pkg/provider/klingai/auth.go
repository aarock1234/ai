package klingai

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// Environment variables of the credentials.
const (
	apiKeyVariable    = "KLINGAI_API_KEY"
	accessKeyVariable = "KLINGAI_ACCESS_KEY"
	secretKeyVariable = "KLINGAI_SECRET_KEY"
)

// JWT timing of legacy credentials: tokens are valid for 30 minutes and
// from 5 seconds before they are issued, to tolerate clock skew.
const (
	tokenLifetime  = 30 * time.Minute
	tokenClockSkew = 5 * time.Second
)

// missingCredentialsMessage explains both authentication schemes.
const missingCredentialsMessage = "KlingAI API key is missing. Pass it using WithAPIKey " +
	"or the KLINGAI_API_KEY environment variable. Alternatively, pass the " +
	"legacy WithAccessKey and WithSecretKey options or the " +
	"KLINGAI_ACCESS_KEY and KLINGAI_SECRET_KEY environment variables."

type tokenHeader struct {
	Algorithm string `json:"alg"`
	Type      string `json:"typ"`
}

type tokenClaims struct {
	Issuer    string `json:"iss"`
	ExpiresAt int64  `json:"exp"`
	NotBefore int64  `json:"nbf"`
}

// authToken resolves the bearer token. Explicit options take precedence
// over environment variables, and an API key over the access key and
// secret key pair:
//
//  1. WithAPIKey
//  2. WithAccessKey and WithSecretKey
//  3. KLINGAI_API_KEY
//  4. KLINGAI_ACCESS_KEY and KLINGAI_SECRET_KEY
//
// Every credential is trimmed, and a blank credential counts as unset.
func (p *Provider) authToken() (string, error) {
	if p.apiKey != "" {
		return p.apiKey, nil
	}

	if p.accessKey != "" && p.secretKey != "" {
		return signToken(p.accessKey, p.secretKey, p.now())
	}

	if apiKey := credential("", apiKeyVariable); apiKey != "" {
		return apiKey, nil
	}

	return p.legacyToken()
}

// legacyToken signs a JWT with the access key and secret key, each read
// from its option or else its environment variable.
func (p *Provider) legacyToken() (string, error) {
	accessKey := credential(p.accessKey, accessKeyVariable)
	secretKey := credential(p.secretKey, secretKeyVariable)

	if accessKey == "" && secretKey == "" {
		return "", fmt.Errorf("%w: %s", model.ErrLoadAPIKey, missingCredentialsMessage)
	}

	if accessKey == "" {
		return "", missingCredentialError("access key", accessKeyVariable)
	}

	if secretKey == "" {
		return "", missingCredentialError("secret key", secretKeyVariable)
	}

	return signToken(accessKey, secretKey, p.now())
}

// signToken creates an HS256 JWT issued by accessKey.
func signToken(accessKey string, secretKey string, now time.Time) (string, error) {
	header, err := encodeTokenPart(tokenHeader{
		Algorithm: "HS256",
		Type:      "JWT",
	})
	if err != nil {
		return "", err
	}

	claims, err := encodeTokenPart(tokenClaims{
		Issuer:    accessKey,
		ExpiresAt: now.Add(tokenLifetime).Unix(),
		NotBefore: now.Add(-tokenClockSkew).Unix(),
	})
	if err != nil {
		return "", err
	}

	signingInput := header + "." + claims

	mac := hmac.New(sha256.New, []byte(secretKey))
	mac.Write([]byte(signingInput))

	signature := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))

	return signingInput + "." + signature, nil
}

func encodeTokenPart(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("encode klingai token: %w", err)
	}

	return base64.RawURLEncoding.EncodeToString(encoded), nil
}

// credential returns the configured value, which the options store trimmed,
// or else the trimmed environment variable. "" means the credential is
// unset.
func credential(configured string, variable string) string {
	if configured != "" {
		return configured
	}

	return strings.TrimSpace(os.Getenv(variable))
}

// missingCredentialError reports a missing half of the legacy key pair.
func missingCredentialError(description string, variable string) error {
	return fmt.Errorf(
		"%w: KlingAI %s is missing; set %s or pass it as an option",
		model.ErrLoadAPIKey,
		description,
		variable,
	)
}
