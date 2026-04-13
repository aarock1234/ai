package ai

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"math"
)

// GenerateID returns a random hexadecimal identifier.
func GenerateID() (string, error) {
	return CreateIDGenerator(16)()
}

// CreateIDGenerator returns a reusable ID generator for fixed-size IDs.
func CreateIDGenerator(size int) func() (string, error) {
	if size < 1 {
		size = 16
	}

	return func() (string, error) {
		buf := make([]byte, size)
		if _, err := rand.Read(buf); err != nil {
			return "", fmt.Errorf("generate id: %w", err)
		}

		return hex.EncodeToString(buf), nil
	}
}

// CosineSimilarity computes the cosine similarity between two vectors.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}

	var (
		dot   float64
		normA float64
		normB float64
	)

	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
