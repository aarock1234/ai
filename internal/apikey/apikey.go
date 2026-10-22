// Package apikey resolves provider API keys from an explicit setting or the
// environment, reporting a missing key with one consistent error message.
package apikey

import (
	"fmt"
	"os"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// Load returns configured when it is set, otherwise the value of the first
// set environment variable in variables. A missing key returns an error
// wrapping model.ErrLoadAPIKey that names every variable.
func Load(configured string, variables ...string) (string, error) {
	apiKey := Lookup(configured, variables...)
	if apiKey == "" {
		return "", missingKeyError(variables)
	}

	return apiKey, nil
}

// Lookup returns configured when it is set, otherwise the value of the first
// set environment variable in variables, or "" when none is set. Use Load
// unless the caller has further credential fallbacks.
func Lookup(configured string, variables ...string) string {
	if configured != "" {
		return configured
	}

	for _, variable := range variables {
		if value := os.Getenv(variable); value != "" {
			return value
		}
	}

	return ""
}

// missingKeyError reports that neither the setting nor any of variables
// provides a key.
func missingKeyError(variables []string) error {
	if len(variables) == 0 {
		return fmt.Errorf("%w: use WithAPIKey", model.ErrLoadAPIKey)
	}

	if len(variables) == 1 {
		return fmt.Errorf("%w: set %s or use WithAPIKey", model.ErrLoadAPIKey, variables[0])
	}

	return fmt.Errorf(
		"%w: set %s, or use WithAPIKey",
		model.ErrLoadAPIKey,
		strings.Join(variables, " or "),
	)
}
