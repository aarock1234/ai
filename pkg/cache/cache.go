// Package cache provides file-based intermediate result caching.
// Cache keys are derived from step name, input ID, and prompt hash
// so that changing a prompt template invalidates cached results.
package cache

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Cache stores intermediate step results on disk.
type Cache struct {
	dir string
}

// New creates a Cache rooted at the given directory.
func New(dir string) *Cache {
	return &Cache{dir: dir}
}

// Get retrieves a cached result. Returns nil, nil if not found.
func (c *Cache) Get(stepName, inputID, promptHash string) (json.RawMessage, error) {
	data, err := os.ReadFile(c.path(stepName, inputID, promptHash))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading cache: %w", err)
	}

	return data, nil
}

// Set stores a result in the cache.
func (c *Cache) Set(stepName, inputID, promptHash string, data json.RawMessage) error {
	p := c.path(stepName, inputID, promptHash)

	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return fmt.Errorf("creating cache dir: %w", err)
	}

	return os.WriteFile(p, data, 0o644)
}

// Hash returns a hex-encoded SHA-256 hash of the given content.
// Used to compute prompt hashes for cache keys.
func Hash(content string) string {
	h := sha256.Sum256([]byte(content))

	return fmt.Sprintf("%x", h)
}

func (c *Cache) path(stepName, inputID, promptHash string) string {
	h := sha256.Sum256([]byte(inputID + "|" + promptHash))
	key := fmt.Sprintf("%x", h[:8])

	return filepath.Join(c.dir, stepName, key+".json")
}
