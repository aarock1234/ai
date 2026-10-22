package codemode

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"regexp"
	"strings"
	"time"
)

// The continuation token uses the signed codec of upstream's run runtime:
// base64url(canonical JSON envelope) + "." + base64url(HMAC-SHA256), keyed
// with the SHA-256 of the continuation signing key, as upstream's code mode
// derives it.
const (
	runTokenSignatureDomain = "run.continuation.v1\x00"
	runTokenFormat          = "run-continuation"
	runTokenAlgorithm       = "HS256"
	runTokenVersion         = 1
	runTokenClockSkew       = time.Minute
	maxRunTokenBytes        = 32 * 1024 * 1024
)

var base64URLPattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

// runTokenEnvelope is the signed payload of a continuation token.
type runTokenEnvelope struct {
	Algorithm   string `json:"algorithm"`
	ExpiresAtMs int64  `json:"expiresAtMs"`
	Format      string `json:"format"`
	IssuedAtMs  int64  `json:"issuedAtMs"`
	Nonce       string `json:"nonce"`
	State       any    `json:"state"`
	Version     int    `json:"version"`
}

// encodeRunToken signs the replay state of an interrupted invocation.
func encodeRunToken(state runState, security continuationSecurity) (string, error) {
	issuedAt := time.Now().UnixMilli()

	body, err := runCanonicalJSON(runTokenEnvelope{
		Algorithm:   runTokenAlgorithm,
		ExpiresAtMs: issuedAt + runTokenMaxAgeMs(security),
		Format:      runTokenFormat,
		IssuedAtMs:  issuedAt,
		Nonce:       randomHex(16),
		State:       state,
		Version:     runTokenVersion,
	})
	if err != nil {
		return "", newProtocolError("code mode continuation state is not encodable: " + err.Error())
	}

	encodedBody := base64.RawURLEncoding.EncodeToString(body)
	token := encodedBody + "." + runTokenSignature(encodedBody, security)

	if len(token) > maxRunTokenBytes {
		return "", newProtocolError("code mode continuation token exceeds the size limit")
	}

	return token, nil
}

// decodeRunToken verifies a continuation token and returns its replay
// state, which the caller validates against the resumed scope.
func decodeRunToken(token string, security continuationSecurity) (json.RawMessage, error) {
	if len(token) > maxRunTokenBytes {
		return nil, newProtocolError("code mode continuation token exceeds the size limit")
	}

	separator := strings.LastIndexByte(token, '.')
	if separator <= 0 {
		return nil, newProtocolError("code mode continuation token is malformed")
	}

	encodedBody := token[:separator]
	signature := token[separator+1:]

	body, ok := decodeBase64URL(encodedBody)
	if !ok {
		return nil, newProtocolError("code mode continuation payload is malformed")
	}

	if !hmac.Equal([]byte(signature), []byte(runTokenSignature(encodedBody, security))) {
		return nil, newProtocolError("code mode continuation signature is invalid")
	}

	return decodeRunTokenEnvelope(body, encodedBody, security)
}

func decodeRunTokenEnvelope(body []byte, encodedBody string, security continuationSecurity) (json.RawMessage, error) {
	invalid := newProtocolError("code mode continuation envelope is invalid")

	fields, ok := exactObject(body, "algorithm", "expiresAtMs", "format", "issuedAtMs", "nonce", "state", "version")
	if !ok {
		return nil, invalid
	}

	var envelope struct {
		Algorithm string `json:"algorithm"`
		Format    string `json:"format"`
		Nonce     string `json:"nonce"`
		Version   int    `json:"version"`
	}

	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, invalid
	}

	issuedAt, issuedOK := safeInteger(fields["issuedAtMs"])
	expiresAt, expiresOK := safeInteger(fields["expiresAtMs"])

	valid := envelope.Format == runTokenFormat &&
		envelope.Version == runTokenVersion &&
		envelope.Algorithm == runTokenAlgorithm &&
		lowerHex32Pattern.MatchString(envelope.Nonce) &&
		issuedOK &&
		expiresOK &&
		expiresAt > issuedAt &&
		isJSONObject(fields["state"])

	if !valid {
		return nil, invalid
	}

	canonical, err := runCanonicalJSON(json.RawMessage(body))
	if err != nil || base64.RawURLEncoding.EncodeToString(canonical) != encodedBody {
		return nil, newProtocolError("code mode continuation payload is not canonically encoded")
	}

	now := time.Now().UnixMilli()

	switch {
	case expiresAt <= now:
		return nil, newProtocolError("code mode continuation has expired")

	case issuedAt > now+runTokenClockSkew.Milliseconds():
		return nil, newProtocolError("code mode continuation was issued in the future")

	case expiresAt-issuedAt > runTokenMaxAgeMs(security):
		return nil, newProtocolError("code mode continuation lifetime is invalid")
	}

	return fields["state"], nil
}

func runTokenSignature(encodedBody string, security continuationSecurity) string {
	secret := sha256.Sum256(security.signingKey)

	mac := hmac.New(sha256.New, secret[:])
	mac.Write([]byte(runTokenSignatureDomain))
	mac.Write([]byte(encodedBody))

	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func runTokenMaxAgeMs(security continuationSecurity) int64 {
	return max(1, security.maxAge.Milliseconds())
}

// decodeBase64URL decodes unpadded base64url that is canonically encoded.
func decodeBase64URL(encoded string) ([]byte, bool) {
	if !base64URLPattern.MatchString(encoded) {
		return nil, false
	}

	decoded, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil || base64.RawURLEncoding.EncodeToString(decoded) != encoded {
		return nil, false
	}

	return decoded, true
}

func isJSONObject(document json.RawMessage) bool {
	var object map[string]json.RawMessage

	return json.Unmarshal(document, &object) == nil && object != nil
}
