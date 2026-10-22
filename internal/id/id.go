// Package id generates random identifiers for generated content.
package id

import (
	"crypto/rand"
	"math/big"
)

const (
	alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	size     = 16
)

// New returns a random 16-character alphanumeric identifier, prefixed with
// "prefix-" when prefix is not empty.
func New(prefix string) string {
	value := random()
	if prefix == "" {
		return value
	}

	return prefix + "-" + value
}

func random() string {
	limit := big.NewInt(int64(len(alphabet)))
	characters := make([]byte, size)

	for index := range characters {
		position, err := rand.Int(rand.Reader, limit)
		if err != nil {
			// crypto/rand does not fail on supported platforms.
			panic(err)
		}

		characters[index] = alphabet[position.Int64()]
	}

	return string(characters)
}
