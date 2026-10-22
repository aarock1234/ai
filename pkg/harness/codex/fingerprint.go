package codex

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
)

// fingerprintPrefix marks fingerprints of canonical, key-sorted JSON.
// Fingerprints without it come from an order-sensitive legacy format.
const fingerprintPrefix = "v2:"

// fingerprintTurnConfiguration hashes the instructions and tools a thread
// runs with, as canonical JSON with sorted object keys.
func fingerprintTurnConfiguration(instructions string, tools []harness.ToolSpec) (string, error) {
	canonicalTools := make([]any, 0, len(tools))

	for _, tool := range tools {
		canonical, err := canonicalTool(tool)
		if err != nil {
			return "", err
		}

		canonicalTools = append(canonicalTools, canonical)
	}

	var canonicalInstructions any
	if instructions != "" {
		canonicalInstructions = instructions
	}

	encoded, err := canonicalJSON(map[string]any{
		"instructions": canonicalInstructions,
		"tools":        canonicalTools,
	})
	if err != nil {
		return "", fmt.Errorf("fingerprint codex turn configuration: %w", err)
	}

	sum := sha256.Sum256(encoded)

	return fingerprintPrefix + hex.EncodeToString(sum[:]), nil
}

// canonicalTool returns a tool as a JSON value whose objects encode with
// sorted keys.
func canonicalTool(tool harness.ToolSpec) (map[string]any, error) {
	canonical := map[string]any{
		"name": tool.Name,
	}

	if tool.Description != "" {
		canonical["description"] = tool.Description
	}

	if len(tool.InputSchema) > 0 {
		decoder := json.NewDecoder(bytes.NewReader(tool.InputSchema))
		decoder.UseNumber()

		var schema any
		if err := decoder.Decode(&schema); err != nil {
			return nil, fmt.Errorf("decode input schema of tool %q: %w", tool.Name, err)
		}

		canonical["inputSchema"] = schema
	}

	return canonical, nil
}

// canonicalJSON encodes value as JSON.stringify does for these values: map
// keys sorted and no HTML escaping.
func canonicalJSON(value any) ([]byte, error) {
	var buffer bytes.Buffer

	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)

	if err := encoder.Encode(value); err != nil {
		return nil, err
	}

	return []byte(strings.TrimSuffix(buffer.String(), "\n")), nil
}
