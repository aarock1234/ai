package mcp

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Event signing secrets follow Standard Webhooks: "whsec_" and the base64
// encoding of 24 to 64 random bytes.
const (
	eventSecretPrefix   = "whsec_"
	eventSecretBytes    = 32
	minEventSecretBytes = 24
	maxEventSecretBytes = 64
)

// eventSignatureTolerance bounds the age and clock skew of a signed
// delivery.
const eventSignatureTolerance = 5 * time.Minute

var (
	eventSecretPattern    = regexp.MustCompile(`^[A-Za-z0-9+/]+={0,2}$`)
	eventTimestampPattern = regexp.MustCompile(`^\d+$`)
)

// NewEventSecret returns a random event signing secret.
func NewEventSecret() string {
	secret := make([]byte, eventSecretBytes)
	_, _ = rand.Read(secret)

	return eventSecretPrefix + base64.StdEncoding.EncodeToString(secret)
}

// decodeEventSecret returns the key of a signing secret.
func decodeEventSecret(secret string) ([]byte, error) {
	encoded, hasPrefix := strings.CutPrefix(secret, eventSecretPrefix)
	if !hasPrefix || !eventSecretPattern.MatchString(encoded) {
		return nil, &ClientError{
			Message: "invalid mcp event signing secret",
		}
	}

	key, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, &ClientError{
			Message: "invalid mcp event signing secret",
			Cause:   err,
		}
	}

	if len(key) < minEventSecretBytes || len(key) > maxEventSecretBytes {
		return nil, &ClientError{
			Message: "mcp event signing secrets must contain 24-64 random bytes",
		}
	}

	return key, nil
}

// verifyEventSignature authenticates the exact body bytes of a delivery with
// Standard Webhooks v1 signatures. Any one valid signature of a rotation
// is enough.
func verifyEventSignature(header http.Header, body []byte, secret string) (bool, error) {
	id := header.Get("webhook-id")
	timestamp := header.Get("webhook-timestamp")
	signatures := header.Get("webhook-signature")

	if id == "" || signatures == "" || !isRecentTimestamp(timestamp) {
		return false, nil
	}

	key, err := decodeEventSecret(secret)
	if err != nil {
		return false, err
	}

	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(id + "." + timestamp + "."))
	mac.Write(body)
	expected := mac.Sum(nil)

	for signature := range strings.SplitSeq(signatures, " ") {
		if matchesSignature(signature, expected) {
			return true, nil
		}
	}

	return false, nil
}

// matchesSignature reports whether a "v1,<base64>" signature equals
// expected. Malformed signatures do not match.
func matchesSignature(signature string, expected []byte) bool {
	parts := strings.Split(signature, ",")
	if len(parts) < 2 || parts[0] != "v1" || parts[1] == "" {
		return false
	}

	decoded, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil {
		return false
	}

	return hmac.Equal(decoded, expected)
}

// isRecentTimestamp reports whether timestamp is a Unix time in seconds
// within the tolerance of now.
func isRecentTimestamp(timestamp string) bool {
	if !eventTimestampPattern.MatchString(timestamp) {
		return false
	}

	seconds, err := strconv.ParseFloat(timestamp, 64)
	if err != nil {
		return false
	}

	now := float64(time.Now().UnixNano()) / float64(time.Second)

	return math.Abs(now-seconds) <= eventSignatureTolerance.Seconds()
}
