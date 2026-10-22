package ai

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// toolApprovalPayloadVersion separates approval signatures from other uses
// of the secret.
const toolApprovalPayloadVersion = "ai-sdk-tool-approval-v1"

// toolApprovalSignature identifies the approval request being signed.
type toolApprovalSignature struct {
	secret     []byte
	approvalID string
	toolCallID string
	toolName   string
	input      json.RawMessage
}

// signToolApproval returns the HMAC-SHA256 signature of an approval request,
// binding the approval to its tool call and input. It returns the empty
// string when no secret is configured.
func signToolApproval(signature toolApprovalSignature) (string, error) {
	if signature.secret == nil {
		return "", nil
	}

	payload, err := toolApprovalPayload(signature)
	if err != nil {
		return "", err
	}

	return base64.RawURLEncoding.EncodeToString(computeHMAC(signature.secret, payload)), nil
}

// verifyToolApprovalSignature checks a signature. Signatures in the legacy
// newline-separated format are accepted when no identifier contains a
// newline, which made that format ambiguous.
func verifyToolApprovalSignature(signature toolApprovalSignature, encoded string) (bool, error) {
	received, err := decodeBase64URL(encoded)
	if err != nil {
		return false, nil
	}

	payload, err := toolApprovalPayload(signature)
	if err != nil {
		return false, err
	}

	if hmac.Equal(received, computeHMAC(signature.secret, payload)) {
		return true, nil
	}

	if strings.ContainsRune(signature.approvalID+signature.toolCallID+signature.toolName, '\n') {
		return false, nil
	}

	legacyPayload, err := legacyToolApprovalPayload(signature)
	if err != nil {
		return false, err
	}

	return hmac.Equal(received, computeHMAC(signature.secret, legacyPayload)), nil
}

// toolApprovalPayload serializes the signed fields as a JSON array, which
// keeps the field boundaries unambiguous.
func toolApprovalPayload(signature toolApprovalSignature) ([]byte, error) {
	digest, err := canonicalDigest(signature.input)
	if err != nil {
		return nil, err
	}

	return encodeJSONWithoutHTMLEscaping([]string{
		toolApprovalPayloadVersion,
		signature.approvalID,
		signature.toolCallID,
		signature.toolName,
		digest,
	})
}

func legacyToolApprovalPayload(signature toolApprovalSignature) ([]byte, error) {
	digest, err := canonicalDigest(signature.input)
	if err != nil {
		return nil, err
	}

	fields := []string{
		signature.approvalID,
		signature.toolCallID,
		signature.toolName,
		digest,
	}

	return []byte(strings.Join(fields, "\n")), nil
}

func computeHMAC(secret, payload []byte) []byte {
	mac := hmac.New(sha256.New, secret)
	mac.Write(payload)

	return mac.Sum(nil)
}

// decodeBase64URL decodes base64url or standard base64, with or without
// padding.
func decodeBase64URL(encoded string) ([]byte, error) {
	normalized := strings.NewReplacer("+", "-", "/", "_").Replace(strings.TrimRight(encoded, "="))

	return base64.RawURLEncoding.DecodeString(normalized)
}

// canonicalDigest returns the base64url SHA-256 digest of the canonical JSON
// encoding of a document.
func canonicalDigest(document json.RawMessage) (string, error) {
	canonical, err := canonicalJSON(document)
	if err != nil {
		return "", err
	}

	digest := sha256.Sum256(canonical)

	return base64.RawURLEncoding.EncodeToString(digest[:]), nil
}

// canonicalJSON encodes a JSON document with sorted object keys, so that
// equal values produce equal bytes regardless of key order or formatting.
// Numbers keep their exact token, so integers beyond float64 precision stay
// distinct. Empty input encodes as null.
func canonicalJSON(document json.RawMessage) ([]byte, error) {
	var value any
	if len(bytes.TrimSpace(document)) > 0 {
		decoder := json.NewDecoder(bytes.NewReader(document))
		decoder.UseNumber()

		if err := decoder.Decode(&value); err != nil {
			return nil, fmt.Errorf("decode canonical JSON input: %w", err)
		}
	}

	var canonical bytes.Buffer
	if err := writeCanonicalJSON(&canonical, value); err != nil {
		return nil, err
	}

	return canonical.Bytes(), nil
}

func writeCanonicalJSON(buffer *bytes.Buffer, value any) error {
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}

		slices.Sort(keys)

		buffer.WriteByte('{')
		for index, key := range keys {
			if index > 0 {
				buffer.WriteByte(',')
			}

			if err := writeCanonicalJSON(buffer, key); err != nil {
				return err
			}

			buffer.WriteByte(':')

			if err := writeCanonicalJSON(buffer, typed[key]); err != nil {
				return err
			}
		}
		buffer.WriteByte('}')

		return nil

	case []any:
		buffer.WriteByte('[')
		for index, element := range typed {
			if index > 0 {
				buffer.WriteByte(',')
			}

			if err := writeCanonicalJSON(buffer, element); err != nil {
				return err
			}
		}
		buffer.WriteByte(']')

		return nil
	}

	encoded, err := encodeJSONWithoutHTMLEscaping(value)
	if err != nil {
		return err
	}

	buffer.Write(encoded)

	return nil
}

// encodeJSONWithoutHTMLEscaping encodes like JavaScript's JSON.stringify,
// which does not escape <, > and &.
func encodeJSONWithoutHTMLEscaping(value any) ([]byte, error) {
	var buffer bytes.Buffer

	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)

	if err := encoder.Encode(value); err != nil {
		return nil, fmt.Errorf("encode canonical JSON: %w", err)
	}

	return bytes.TrimSuffix(buffer.Bytes(), []byte("\n")), nil
}

// ToolDrift lists the differences between two tool fingerprint maps.
type ToolDrift struct {
	// Added are tools present only in the current fingerprints.
	Added []string

	// Removed are tools present only in the baseline.
	Removed []string

	// Changed are tools whose definition differs.
	Changed []string
}

// FingerprintTools returns a stable digest of the security-relevant fields
// of each tool, keyed by tool name: its description, input schema and title.
// Store a baseline when the tools are first trusted and compare later
// definitions with DetectToolDrift to catch changed tool definitions, for
// example from an MCP server.
func FingerprintTools(tools []Tool) (map[string]string, error) {
	fingerprints := make(map[string]string, len(tools))

	for _, tool := range tools {
		definition, err := toolFingerprintDefinition(tool)
		if err != nil {
			return nil, fmt.Errorf("fingerprint tool %q: %w", tool.Name, err)
		}

		digest, err := canonicalDigest(definition)
		if err != nil {
			return nil, fmt.Errorf("fingerprint tool %q: %w", tool.Name, err)
		}

		fingerprints[tool.Name] = digest
	}

	return fingerprints, nil
}

// toolFingerprint is the part of a tool definition that a fingerprint
// covers.
type toolFingerprint struct {
	Description toolFingerprintDescription `json:"description"`
	InputSchema json.RawMessage            `json:"inputSchema,omitempty"`
	Title       string                     `json:"title,omitempty"`
}

// toolFingerprintDescription has the type "none" for tools without a
// description and "string" otherwise.
type toolFingerprintDescription struct {
	Type  string `json:"type"`
	Value string `json:"value,omitempty"`
}

func toolFingerprintDefinition(tool Tool) (json.RawMessage, error) {
	description := toolFingerprintDescription{
		Type: "none",
	}

	if tool.Description != "" {
		description = toolFingerprintDescription{
			Type:  "string",
			Value: tool.Description,
		}
	}

	encoded, err := json.Marshal(toolFingerprint{
		Description: description,
		InputSchema: tool.InputSchema,
		Title:       tool.Title,
	})
	if err != nil {
		return nil, fmt.Errorf("encode definition: %w", err)
	}

	return encoded, nil
}

// DetectToolDrift compares fingerprints from FingerprintTools with a
// baseline. The lists are sorted by tool name.
func DetectToolDrift(current, baseline map[string]string) ToolDrift {
	var drift ToolDrift

	for name, digest := range current {
		baselineDigest, known := baseline[name]

		switch {
		case !known:
			drift.Added = append(drift.Added, name)
		case digest != baselineDigest:
			drift.Changed = append(drift.Changed, name)
		}
	}

	for name := range baseline {
		if _, present := current[name]; !present {
			drift.Removed = append(drift.Removed, name)
		}
	}

	slices.Sort(drift.Added)
	slices.Sort(drift.Removed)
	slices.Sort(drift.Changed)

	return drift
}
