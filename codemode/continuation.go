package codemode

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"sync"
	"time"
)

const (
	continuationVersion       = 2
	continuationSignatureAlg  = "HMAC-SHA256"
	defaultContinuationMaxAge = time.Hour
	continuationClockSkew     = time.Minute
)

var continuationNoncePattern = regexp.MustCompile(`^[0-9a-fA-F]{32}$`)

// ContinuationSecurity configures how continuations are signed and how long
// they stay valid.
type ContinuationSecurity struct {
	// SigningKey authenticates continuations. Nil uses the process default,
	// which is random unless SetContinuationSigningKey replaced it. Use a
	// stable key to resume continuations in another process.
	SigningKey []byte

	// MaxAge is how long a continuation stays valid. Zero uses the process
	// default, one hour unless SetContinuationSigningKey replaced it.
	MaxAge time.Duration
}

// Continuation is the opaque state of an interrupted invocation. The token
// and the compatibility metadata are authenticated; persist the value
// without reading or modifying its fields.
//
// Its JSON encoding, signature and token match upstream's code mode and
// its run runtime, so a continuation created by either implementation
// resumes in the other when both use the same signing key.
type Continuation struct {
	Version              int                   `json:"version"`
	JS                   string                `json:"js"`
	OuterToolCallID      string                `json:"outerToolCallId"`
	ToolNames            []string              `json:"toolNames"`
	Token                string                `json:"token"`
	PendingInterruptions []PendingInterruption `json:"pendingInterruptions"`
	Resolutions          []PendingResolution   `json:"resolutions"`
	Auth                 ContinuationAuth      `json:"auth"`
}

// ContinuationAuth is the signature metadata of a Continuation.
type ContinuationAuth struct {
	Alg         string `json:"alg"`
	Nonce       string `json:"nonce"`
	IssuedAtMs  int64  `json:"issuedAtMs"`
	ExpiresAtMs int64  `json:"expiresAtMs"`
	Signature   string `json:"signature,omitempty"`
}

// PendingInterruption is one interruption of the batch that an
// interrupted invocation stopped at. It is part of the authenticated
// Continuation state.
type PendingInterruption struct {
	RunInterruptionID string           `json:"runInterruptionId"`
	InterruptID       string           `json:"interruptId"`
	ToolName          string           `json:"toolName"`
	ToolCallID        string           `json:"toolCallId"`
	Input             json.RawMessage  `json:"input,omitempty"`
	Payload           InterruptPayload `json:"payload"`
}

// PendingResolution is a resolution collected while the interruptions of a
// batch are exposed one at a time. It is part of the authenticated
// Continuation state.
type PendingResolution struct {
	RunInterruptionID string `json:"runInterruptionId"`

	// Value is the JSON of the resolution; nil is undefined.
	Value json.RawMessage `json:"value,omitempty"`
}

// continuationSecurity is a ContinuationSecurity with the defaults applied.
type continuationSecurity struct {
	signingKey []byte
	maxAge     time.Duration
}

// continuationDefaults holds the process-wide signing key and maximum age.
var continuationDefaults = struct {
	mu         sync.RWMutex
	signingKey []byte
	maxAge     time.Duration
}{
	signingKey: []byte(rand.Text()),
	maxAge:     defaultContinuationMaxAge,
}

// SetContinuationSigningKey replaces the process-wide default signing key
// and maximum age used when ContinuationSecurity leaves them unset. A nil
// key generates a new random key; a zero maxAge restores one hour.
func SetContinuationSigningKey(key []byte, maxAge time.Duration) error {
	if key == nil {
		key = []byte(rand.Text())
	}

	if len(key) == 0 {
		return errors.New("continuation signing key must not be empty")
	}

	if maxAge < 0 {
		return errors.New("continuation max age must not be negative")
	}

	continuationDefaults.mu.Lock()
	defer continuationDefaults.mu.Unlock()

	continuationDefaults.signingKey = slices.Clone(key)
	continuationDefaults.maxAge = positiveOr(maxAge, defaultContinuationMaxAge)

	return nil
}

func resolveContinuationSecurity(options ContinuationSecurity) (continuationSecurity, error) {
	if options.SigningKey != nil && len(options.SigningKey) == 0 {
		return continuationSecurity{}, errors.New("continuation signing key must not be empty")
	}

	if options.MaxAge < 0 {
		return continuationSecurity{}, errors.New("continuation max age must not be negative")
	}

	continuationDefaults.mu.RLock()
	defer continuationDefaults.mu.RUnlock()

	security := continuationSecurity{
		signingKey: continuationDefaults.signingKey,
		maxAge:     positiveOr(options.MaxAge, continuationDefaults.maxAge),
	}

	if options.SigningKey != nil {
		security.signingKey = options.SigningKey
	}

	return security, nil
}

// signContinuation returns a copy of the continuation with fresh signature
// metadata.
func signContinuation(continuation Continuation, security continuationSecurity) (Continuation, error) {
	nonce := make([]byte, 16)
	_, _ = rand.Read(nonce)

	issuedAt := time.Now()

	signed := cloneContinuation(continuation)
	signed.Auth = ContinuationAuth{
		Alg:         continuationSignatureAlg,
		Nonce:       hex.EncodeToString(nonce),
		IssuedAtMs:  issuedAt.UnixMilli(),
		ExpiresAtMs: issuedAt.Add(security.maxAge).UnixMilli(),
	}

	signature, err := continuationSignature(signed, security.signingKey)
	if err != nil {
		return Continuation{}, err
	}

	signed.Auth.Signature = signature

	return signed, nil
}

// verifyContinuation checks the shape, lifetime and signature of a
// continuation.
func verifyContinuation(continuation Continuation, security continuationSecurity) error {
	if !hasContinuationEnvelope(continuation) {
		return newProtocolError("code mode continuation envelope is malformed")
	}

	if !hasContinuationAuth(continuation.Auth) {
		return newProtocolError("code mode continuation is missing valid signed auth metadata")
	}

	now := time.Now().UnixMilli()

	if continuation.Auth.ExpiresAtMs < now {
		return newProtocolError("code mode continuation has expired")
	}

	if continuation.Auth.IssuedAtMs > now+continuationClockSkew.Milliseconds() {
		return newProtocolError("code mode continuation was issued in the future")
	}

	expected, err := continuationSignature(continuation, security.signingKey)
	if err != nil {
		return err
	}

	if !hmac.Equal([]byte(continuation.Auth.Signature), []byte(expected)) {
		return newProtocolError("code mode continuation signature is invalid")
	}

	return nil
}

func hasContinuationEnvelope(continuation Continuation) bool {
	return continuation.Version == continuationVersion &&
		continuation.Token != "" &&
		len(continuation.PendingInterruptions) > 0
}

func hasContinuationAuth(auth ContinuationAuth) bool {
	return auth.Alg == continuationSignatureAlg &&
		continuationNoncePattern.MatchString(auth.Nonce) &&
		auth.ExpiresAtMs > auth.IssuedAtMs &&
		auth.Signature != ""
}

// continuationSignature signs the canonical JSON of the continuation
// without its signature, as upstream does.
func continuationSignature(continuation Continuation, signingKey []byte) (string, error) {
	unsigned := continuation
	unsigned.Auth.Signature = ""

	payload, err := codeModeCanonicalJSON(unsigned)
	if err != nil {
		return "", fmt.Errorf("encode continuation: %w", err)
	}

	mac := hmac.New(sha256.New, signingKey)
	mac.Write(payload)

	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil)), nil
}

func cloneContinuation(continuation Continuation) Continuation {
	cloned := continuation
	cloned.ToolNames = slices.Clone(continuation.ToolNames)
	cloned.PendingInterruptions = make([]PendingInterruption, len(continuation.PendingInterruptions))
	cloned.Resolutions = make([]PendingResolution, len(continuation.Resolutions))

	for index, pending := range continuation.PendingInterruptions {
		cloned.PendingInterruptions[index] = clonePendingInterruption(pending)
	}

	for index, resolution := range continuation.Resolutions {
		cloned.Resolutions[index] = PendingResolution{
			RunInterruptionID: resolution.RunInterruptionID,
			Value:             slices.Clone(resolution.Value),
		}
	}

	return cloned
}

func clonePendingInterruption(pending PendingInterruption) PendingInterruption {
	cloned := pending
	cloned.Input = slices.Clone(pending.Input)
	cloned.Payload = pending.Payload.clone()

	return cloned
}
